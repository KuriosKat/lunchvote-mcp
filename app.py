import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import httpx
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
import contextlib

from mcp.server.fastmcp import FastMCP

# -----------------------------
# 환경변수
# -----------------------------
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "c8aa5a99a9b1489f3104b321c028b58f").strip()
DB_PATH = os.getenv("DB_PATH", "lunchvote.db")

if not KAKAO_REST_API_KEY:
    # 서버 실행은 되지만, 실제 호출 시 에러를 내도록 둘 수도 있어.
    # 초보자 입장에선 "키 설정 안 하면 바로 알려주는" 게 낫기 때문에 여기서 막음.
    raise RuntimeError("환경변수 KAKAO_REST_API_KEY 가 필요합니다. (카카오 REST API 키)")

# -----------------------------
# DB (SQLite, 로컬 파일)
# -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with db() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            group_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS group_members (
            group_id TEXT,
            user_id TEXT,
            PRIMARY KEY (group_id, user_id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bans (
            scope TEXT,          -- 'user' or 'group'
            scope_id TEXT,       -- user_id or group_id
            keyword TEXT,
            created_at TEXT,
            PRIMARY KEY (scope, scope_id, keyword)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recents (
            user_id TEXT,
            keyword TEXT,
            eaten_at TEXT,
            PRIMARY KEY (user_id, keyword, eaten_at)
        )
        """)
        conn.commit()

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def make_id(prefix: str) -> str:
    # 초간단 ID (운영에선 UUID 권장)
    return f"{prefix}_{int(datetime.utcnow().timestamp())}"

# -----------------------------
# Kakao Local API 호출
# -----------------------------
KAKAO_BASE = "https://dapi.kakao.com"

async def kakao_category_search(
    x: float,
    y: float,
    category_group_code: str = "FD6",
    radius: int = 1500,
    size: int = 15,
    sort: str = "distance",
) -> List[Dict[str, Any]]:
    """
    카카오 로컬 API: 카테고리로 장소 검색
    - category_group_code: FD6(음식점), CE7(카페) 등
    """
    url = f"{KAKAO_BASE}/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {
        "category_group_code": category_group_code,
        "x": x,
        "y": y,
        "radius": radius,
        "size": size,
        "sort": sort,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("documents", [])

# -----------------------------
# 추천 로직
# -----------------------------
def load_bans(group_id: str, user_id: str) -> List[str]:
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT keyword FROM bans WHERE scope='group' AND scope_id=?", (group_id,))
        group_bans = [row["keyword"] for row in cur.fetchall()]
        cur.execute("SELECT keyword FROM bans WHERE scope='user' AND scope_id=?", (user_id,))
        user_bans = [row["keyword"] for row in cur.fetchall()]
    # 중복 제거
    return sorted(set(group_bans + user_bans))

def load_recent_keywords(user_id: str, within_days: int = 3) -> List[str]:
    cutoff = datetime.utcnow() - timedelta(days=within_days)
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT keyword, eaten_at FROM recents WHERE user_id=?",
            (user_id,),
        )
        out = []
        for row in cur.fetchall():
            try:
                t = datetime.fromisoformat(row["eaten_at"])
            except Exception:
                continue
            if t >= cutoff:
                out.append(row["keyword"])
    return sorted(set(out))

def matches_any(text: str, keywords: List[str]) -> bool:
    text = (text or "").lower()
    for kw in keywords:
        if kw.lower() in text:
            return True
    return False

def format_place(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "place_name": p.get("place_name"),
        "distance_m": int(p.get("distance") or 0) if str(p.get("distance") or "").isdigit() else p.get("distance"),
        "category_name": p.get("category_name"),
        "address_name": p.get("address_name"),
        "road_address_name": p.get("road_address_name"),
        "phone": p.get("phone"),
        "place_url": p.get("place_url"),
        "x": p.get("x"),
        "y": p.get("y"),
    }

# -----------------------------
# MCP 서버 정의
# - Streamable HTTP는 프로덕션 권장 / stateless_http + json_response 추천이 문서에 있음
# -----------------------------
mcp = FastMCP(
    "LunchVote MCP",
    stateless_http=True,
    json_response=True,
)

@mcp.tool()
def group_create(name: str) -> Dict[str, str]:
    """새 밥모임(투표권자 그룹)을 만든다."""
    group_id = make_id("grp")
    with db() as conn:
        conn.execute(
            "INSERT INTO groups(group_id, name, created_at) VALUES(?,?,?)",
            (group_id, name, now_iso()),
        )
        conn.commit()
    return {"group_id": group_id}

@mcp.tool()
def group_add_member(group_id: str, user_id: str) -> Dict[str, str]:
    """그룹에 멤버를 추가한다."""
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO group_members(group_id, user_id) VALUES(?,?)",
            (group_id, user_id),
        )
        conn.commit()
    return {"ok": "true"}

@mcp.tool()
def ban_add(scope: str, scope_id: str, keyword: str) -> Dict[str, str]:
    """
    밴 키워드를 추가한다.
    - scope: 'user' 또는 'group'
    - scope_id: user_id 또는 group_id
    예) scope='group', scope_id='grp_...', keyword='치킨'
    """
    if scope not in ("user", "group"):
        raise ValueError("scope는 'user' 또는 'group' 이어야 합니다.")
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bans(scope, scope_id, keyword, created_at) VALUES(?,?,?,?)",
            (scope, scope_id, keyword, now_iso()),
        )
        conn.commit()
    return {"ok": "true"}

@mcp.tool()
def ban_remove(scope: str, scope_id: str, keyword: str) -> Dict[str, str]:
    """밴 키워드를 제거한다."""
    with db() as conn:
        conn.execute(
            "DELETE FROM bans WHERE scope=? AND scope_id=? AND keyword=?",
            (scope, scope_id, keyword),
        )
        conn.commit()
    return {"ok": "true"}

@mcp.tool()
def recent_add(user_id: str, keyword: str, eaten_at_iso: Optional[str] = None) -> Dict[str, str]:
    """
    최근 먹은 것을 기록한다 (ex: '국밥', '초밥', '파스타').
    eaten_at_iso 미지정 시 현재 시각(UTC) 사용.
    """
    ts = eaten_at_iso or now_iso()
    with db() as conn:
        conn.execute(
            "INSERT INTO recents(user_id, keyword, eaten_at) VALUES(?,?,?)",
            (user_id, keyword, ts),
        )
        conn.commit()
    return {"ok": "true"}

@mcp.tool()
async def recommend_restaurants(
    group_id: str,
    user_id: str,
    x: float,
    y: float,
    radius: int = 1500,
    limit: int = 5,
    recent_days: int = 3,
) -> Dict[str, Any]:
    """
    위치(x=경도, y=위도) 기반으로 근처 음식점을 추천한다.
    - 그룹 밴 + 유저 밴
    - 유저 최근먹은 키워드(기본 3일)도 제외
    """
    bans = load_bans(group_id, user_id)
    recents = load_recent_keywords(user_id, within_days=recent_days)
    candidates = await kakao_category_search(x=x, y=y, category_group_code="FD6", radius=radius, size=30)

    filtered = []
    for p in candidates:
        cat = p.get("category_name", "") or ""
        name = p.get("place_name", "") or ""
        # 밴/최근 키워드가 가게명 또는 카테고리에 포함되면 제외
        if matches_any(name, bans) or matches_any(cat, bans):
            continue
        if matches_any(name, recents) or matches_any(cat, recents):
            continue
        filtered.append(format_place(p))

    return {
        "input": {"group_id": group_id, "user_id": user_id, "x": x, "y": y, "radius": radius},
        "excluded": {"bans": bans, "recent_keywords": recents},
        "results": filtered[: max(1, min(limit, 20))],
        "note": "결과가 비면 radius를 늘리거나(예: 3000), 밴/최근 키워드를 줄여보세요.",
    }

# -----------------------------
# ASGI 앱으로 띄우기 (/mcp)
# - 공식 문서 예제대로 Starlette에 mount
# - 브라우저/플랫폼에서 header 읽기 위해 CORS에서 Mcp-Session-Id 노출 권장
# -----------------------------
@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield

init_db()

app = Starlette(
    routes=[
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

app = CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
