# 맛집 추천 AI Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LangGraph 기반 맛집 추천 ReAct Agent를 구현한다. 5개 패턴(ReAct, Tool Use, Plan-and-Solve, Memory, Reflection)이 self-hosted Phoenix 트레이스에 또렷이 보이고, 채점용 시나리오 "전주 객사 근처 친구랑 저녁..."를 실행하면 추천 3곳과 trace.md가 산출된다.

**Architecture:** `load_memory → planner → react_agent ↔ tool_node → aggregator → reflector → finalizer → save_memory` 8개 노드 그래프. 도구 4개(Kakao geocode/local, Naver local, Google Places)는 LLM이 자율 호출, aggregator는 결정론적 코드(dedup/score/filter), reflector는 최대 2회 재시도. 메모리는 SQLite, 트레이스는 `phoenix.otel.register(auto_instrument=True)`로 LangChain instrumentor가 자동 캡쳐.

**Tech Stack:** Python 3.11+, uv, LangGraph 0.2+, langchain-openai, OpenAI `gpt-4o-mini`, langgraph-checkpoint-sqlite, arize-phoenix-otel, openinference-instrumentation-langchain, httpx, pydantic, rapidfuzz, rich, pytest + respx (httpx mock) + pytest-mock.

**Source spec:** [`docs/superpowers/specs/2026-05-28-restaurant-agent-design.md`](../specs/2026-05-28-restaurant-agent-design.md)

---

## Task 0: 프로젝트 부트스트랩 (uv + 의존성 + .env.example + README 스텁)

**Files:**
- Create: `pyproject.toml` (uv init)
- Create: `.python-version` (uv init)
- Create: `.env.example`
- Create: `README.md`
- Modify: `.gitignore` (확인만)

- [ ] **Step 1: uv 초기화**

Run: `uv init --no-readme --bare` (현재 디렉터리, 기존 .gitignore 보존)
Expected: `pyproject.toml` + `.python-version` 생성, 기존 파일 영향 없음.

- [ ] **Step 2: 런타임 의존성 추가**

Run:
```bash
uv add \
  "langgraph>=0.2" \
  "langchain-openai>=0.2" \
  "langgraph-checkpoint-sqlite>=2.0" \
  "openai>=1.50" \
  "arize-phoenix-otel" \
  "openinference-instrumentation-langchain" \
  "httpx>=0.27" \
  "pydantic>=2.7" \
  "python-dotenv>=1.0" \
  "rapidfuzz>=3.9" \
  "rich>=13.7"
```
Expected: `pyproject.toml`에 `dependencies` 갱신, `uv.lock` 생성.

- [ ] **Step 3: 개발 의존성 추가**

Run:
```bash
uv add --dev pytest pytest-mock respx
```
Expected: `[dependency-groups].dev`에 추가.

- [ ] **Step 4: `.env.example` 생성**

Write `/home/rheon/Desktop/projects/OSS/Assign4/.env.example`:
```bash
# === LLM (필수) ===
OPENAI_API_KEY=sk-...

# === 지역/맛집 검색 ===
# 필수: 발급 → https://developers.kakao.com (REST API key)
KAKAO_REST_API_KEY=

# 선택: 발급 → https://developers.naver.com (검색 → 지역)
NAVER_CLIENT_ID=
NAVER_CLIENT_SECRET=

# 선택: 발급 → https://console.cloud.google.com (Places API New)
GOOGLE_PLACES_API_KEY=

# === Phoenix (트레이스, 선택) ===
PHOENIX_COLLECTOR_ENDPOINT=https://phoenix.rheon.kr/api/collect
PHOENIX_API_KEY=
PHOENIX_PROJECT_NAME=restaurant-recommender-agent
# PHOENIX_DISABLED=1   # 트레이스 끄기
# PHOENIX_DEBUG=1      # OTel 디버그 로그
```

- [ ] **Step 5: README 스텁 생성**

Write `/home/rheon/Desktop/projects/OSS/Assign4/README.md`:
```markdown
# 맛집 추천 AI Agent (OSS Assign4)

LangGraph 기반 ReAct Agent. 자세한 설계는 [docs/superpowers/specs/](docs/superpowers/specs/2026-05-28-restaurant-agent-design.md).

## 실행
```bash
uv sync
cp .env.example .env && $EDITOR .env
uv run python scripts/seed_memory.py
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

> 채점자 안내: 필수 키는 `OPENAI_API_KEY`, `KAKAO_REST_API_KEY` 두 개. Naver/Google/Phoenix는 없어도 동작.
```

- [ ] **Step 6: 의존성 설치 확인**

Run: `uv sync`
Expected: `Resolved N packages` + `.venv/` 생성, 에러 없음.

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml uv.lock .python-version .env.example README.md
git commit -m "chore: bootstrap uv project with deps, .env.example, README stub"
```

---

## Task 1: Phoenix telemetry 모듈

**Files:**
- Create: `src/__init__.py`
- Create: `src/observability/__init__.py`
- Create: `src/observability/telemetry.py`
- Create: `tests/__init__.py`
- Create: `tests/observability/__init__.py`
- Create: `tests/observability/test_telemetry.py`

- [ ] **Step 1: 빈 패키지 파일 생성**

```bash
mkdir -p src/observability tests/observability
touch src/__init__.py src/observability/__init__.py tests/__init__.py tests/observability/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

Write `tests/observability/test_telemetry.py`:
```python
import os
from unittest.mock import patch

import pytest

from src.observability import telemetry


@pytest.fixture(autouse=True)
def reset_state():
    telemetry._provider = None
    yield
    telemetry._provider = None


def test_init_is_idempotent(monkeypatch):
    monkeypatch.setenv("PHOENIX_API_KEY", "k")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.rheon.kr/api/collect")
    with patch.object(telemetry, "register") as m:
        m.return_value.force_flush = lambda: None
        telemetry.init_telemetry()
        telemetry.init_telemetry()
    assert m.call_count == 1


def test_init_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")
    with patch.object(telemetry, "register") as m:
        telemetry.init_telemetry()
    m.assert_not_called()


def test_init_skips_when_key_missing(monkeypatch):
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    with patch.object(telemetry, "register") as m:
        telemetry.init_telemetry()
    m.assert_not_called()


def test_flush_safe_when_uninitialized():
    # Should not raise even if init was never called
    telemetry.flush_telemetry()
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/observability/test_telemetry.py -v`
Expected: `ImportError` 또는 `AttributeError` (모듈/함수 없음).

- [ ] **Step 4: telemetry.py 구현**

Write `src/observability/telemetry.py`:
```python
"""Phoenix OTel bootstrap. Call init_telemetry() exactly once at process start
BEFORE importing langchain/langgraph so auto-instrumentation wraps them."""

from __future__ import annotations
import os
from typing import Any

from phoenix.otel import register

_provider: Any = None


def init_telemetry() -> None:
    global _provider
    if _provider is not None:
        return
    if os.getenv("PHOENIX_DISABLED") == "1":
        return
    api_key = os.getenv("PHOENIX_API_KEY")
    if not api_key:
        # No key → silently skip; agent still runs without traces.
        return

    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.rheon.kr/api/collect"
    )
    project = os.getenv("PHOENIX_PROJECT_NAME", "restaurant-recommender-agent")

    _provider = register(
        project_name=project,
        endpoint=endpoint,
        headers={"authorization": f"Bearer {api_key}"},
        auto_instrument=True,
    )
    print(
        f"[phoenix] telemetry initialized → project=\"{project}\" "
        f"endpoint={endpoint} (auth)"
    )


def flush_telemetry() -> None:
    """Force BatchSpanProcessor to ship queued spans before process exit."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception:  # noqa: BLE001 — flush errors must not crash the CLI
        pass
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/observability/test_telemetry.py -v`
Expected: 4 passed.

- [ ] **Step 6: 커밋**

```bash
git add src/observability/ src/__init__.py tests/observability/ tests/__init__.py
git commit -m "feat(telemetry): Phoenix OTel bootstrap with kill switch and idempotent init"
```

---

## Task 2: 메모리 스토어 + 시드 스크립트

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/store.py`
- Create: `scripts/seed_memory.py`
- Create: `tests/memory/__init__.py`
- Create: `tests/memory/test_store.py`

- [ ] **Step 1: 빈 패키지 디렉터리 + 데이터 디렉터리**

```bash
mkdir -p src/memory tests/memory scripts data
touch src/memory/__init__.py tests/memory/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

Write `tests/memory/test_store.py`:
```python
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


def test_user_profile_set_get(store):
    store.set_user_profile("disliked_categories", ["해물", "회"])
    assert store.get_user_profile("disliked_categories") == ["해물", "회"]


def test_user_profile_missing_returns_default(store):
    assert store.get_user_profile("nope", default="x") == "x"


def test_visit_history_append_and_recent(store):
    now = datetime(2026, 5, 28, 19, 0, 0)
    store.append_visit("백송갈비", "한식", visited_at=now - timedelta(days=1))
    store.append_visit("스시오마카세", "일식", visited_at=now - timedelta(days=3))
    recent = store.get_recent_visits(within_days=1, now=now)
    assert [v["name"] for v in recent] == ["백송갈비"]


def test_visit_history_within_7_days(store):
    now = datetime(2026, 5, 28, 19, 0, 0)
    store.append_visit("백송갈비", "한식", visited_at=now - timedelta(days=1))
    store.append_visit("스시오마카세", "일식", visited_at=now - timedelta(days=3))
    recent = store.get_recent_visits(within_days=7, now=now)
    assert {v["name"] for v in recent} == {"백송갈비", "스시오마카세"}


def test_all_profile_keys(store):
    store.set_user_profile("a", 1)
    store.set_user_profile("b", "two")
    assert store.all_user_profile() == {"a": 1, "b": "two"}
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/memory/test_store.py -v`
Expected: `ImportError`.

- [ ] **Step 4: 스토어 구현**

Write `src/memory/store.py`:
```python
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profile (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visit_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  category TEXT,
  visited_at TEXT NOT NULL,
  source TEXT
);
CREATE INDEX IF NOT EXISTS visit_history_visited_at_idx
  ON visit_history(visited_at);
"""


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- user_profile ---
    def set_user_profile(self, key: str, value: Any) -> None:
        now = datetime.now().isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO user_profile(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now),
            )

    def get_user_profile(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute(
                "SELECT value FROM user_profile WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def all_user_profile(self) -> dict[str, Any]:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM user_profile").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # --- visit_history ---
    def append_visit(
        self,
        name: str,
        category: str | None,
        visited_at: datetime | None = None,
        source: str = "recommended",
    ) -> None:
        ts = (visited_at or datetime.now()).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO visit_history(name,category,visited_at,source) VALUES(?,?,?,?)",
                (name, category, ts, source),
            )

    def get_recent_visits(
        self, within_days: int, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        ref = (now or datetime.now()).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT name,category,visited_at,source FROM visit_history "
                "WHERE julianday(?) - julianday(visited_at) <= ? "
                "ORDER BY visited_at DESC",
                (ref, within_days),
            ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/memory/test_store.py -v`
Expected: 5 passed.

- [ ] **Step 6: 시드 스크립트 작성**

Write `scripts/seed_memory.py`:
```python
"""Seed the agent's memory with a demo user profile and visit history,
so the Memory pattern is observable from the very first run.

Idempotent — re-running replaces profile keys and skips duplicate visits."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from src.memory.store import MemoryStore

DB_PATH = Path("data/agent_memory.db")

PROFILE = {
    "disliked_categories": ["해물", "회"],
    "default_budget": "moderate",
    "notes": "친구와의 저녁은 적당한 가격대의 한식을 선호",
}

DEMO_VISITS = [
    {"name": "백송갈비 객사점", "category": "한식", "days_ago": 1},
    {"name": "전주 콩나물국밥 본점", "category": "한식", "days_ago": 3},
]


def main() -> None:
    store = MemoryStore(DB_PATH)

    for k, v in PROFILE.items():
        store.set_user_profile(k, v)

    existing_names = {v["name"] for v in store.get_recent_visits(within_days=30)}
    now = datetime.now()
    added = 0
    for v in DEMO_VISITS:
        if v["name"] in existing_names:
            continue
        store.append_visit(
            v["name"], v["category"],
            visited_at=now - timedelta(days=v["days_ago"]),
            source="seed",
        )
        added += 1

    print(f"[seed] profile keys: {list(store.all_user_profile().keys())}")
    print(f"[seed] visits added: {added}, total recent (7d): "
          f"{len(store.get_recent_visits(within_days=7))}")
    print(f"[seed] db: {DB_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 시드 스크립트 동작 확인**

Run: `uv run python scripts/seed_memory.py`
Expected:
```
[seed] profile keys: ['disliked_categories', 'default_budget', 'notes']
[seed] visits added: 2, total recent (7d): 2
[seed] db: data/agent_memory.db
```
Run again: `uv run python scripts/seed_memory.py`
Expected: `visits added: 0` (idempotent).

- [ ] **Step 8: 커밋**

```bash
git add src/memory/ tests/memory/ scripts/seed_memory.py
git commit -m "feat(memory): SQLite store with user_profile + visit_history + seed script"
```

---

## Task 3: 공유 타입 (Restaurant, Plan, AgentState)

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/types.py`
- Create: `src/agent/state.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/test_types.py`

- [ ] **Step 1: 패키지 디렉터리**

```bash
mkdir -p src/agent tests/agent
touch src/agent/__init__.py tests/agent/__init__.py
```

- [ ] **Step 2: 타입 테스트 작성**

Write `tests/agent/test_types.py`:
```python
from src.agent.types import Restaurant, Plan, AggregationWeights, DEFAULT_WEIGHTS


def test_restaurant_minimum_fields():
    r = Restaurant(id="k_1", name="가게", source="kakao")
    assert r.id == "k_1"
    assert r.rating is None
    assert r.source_count == 1


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.model_dump().values()) - 1.0) < 1e-9


def test_plan_parses_json():
    plan = Plan.model_validate({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance"},
        "naver": {"sort": "comment"},
        "google": {
            "included_type": "restaurant",
            "price_levels": ["PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"],
            "min_rating": 4.0,
        },
        "post_filters": {
            "exclude_categories": ["해물", "회"],
            "exclude_visited_within_days": 1,
            "k": 3,
        },
        "weights": {"rating": 0.45, "review": 0.30, "distance": 0.05,
                    "match": 0.10, "price": 0.10},
    })
    assert plan.weights.rating == 0.45
    assert plan.post_filters.k == 3
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `uv run pytest tests/agent/test_types.py -v`
Expected: ImportError.

- [ ] **Step 4: 타입 구현**

Write `src/agent/types.py`:
```python
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Source = Literal["kakao", "naver", "google"]


class Restaurant(BaseModel):
    id: str
    name: str
    source: Source
    category: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    distance_m: Optional[float] = None
    rating: Optional[float] = None              # 0..5 (Google)
    review_count: Optional[int] = None
    price_level: Optional[int] = None           # 0..4 (Google)
    phone: Optional[str] = None
    url: Optional[str] = None
    source_count: int = 1                       # set by aggregator after dedup
    score: Optional[float] = None               # set by aggregator


class AggregationWeights(BaseModel):
    rating: float = 0.35
    review: float = 0.25
    distance: float = 0.15
    match: float = 0.15
    price: float = 0.10


DEFAULT_WEIGHTS = AggregationWeights()


class KakaoParams(BaseModel):
    category_group_code: str = "FD6"            # FD6=음식점, CE7=카페
    radius_m: int = 800
    sort: Literal["distance", "accuracy"] = "distance"
    size: int = 15


class NaverParams(BaseModel):
    sort: Literal["random", "comment"] = "comment"


class GoogleParams(BaseModel):
    included_type: str = "restaurant"
    price_levels: Optional[list[str]] = None    # e.g. ["PRICE_LEVEL_MODERATE"]
    min_rating: Optional[float] = None
    open_now: Optional[bool] = None
    language_code: str = "ko"


class PostFilters(BaseModel):
    exclude_categories: list[str] = Field(default_factory=list)
    exclude_visited_within_days: int = 1
    k: int = 3


class Plan(BaseModel):
    region_query: str
    needs_geocoding: bool = True
    food_keywords: list[str] = Field(default_factory=list)   # ex: ["저녁", "한식"]
    kakao: KakaoParams = Field(default_factory=KakaoParams)
    naver: NaverParams = Field(default_factory=NaverParams)
    google: GoogleParams = Field(default_factory=GoogleParams)
    post_filters: PostFilters = Field(default_factory=PostFilters)
    weights: AggregationWeights = Field(default_factory=AggregationWeights)
    clarification_needed: list[str] = Field(default_factory=list)
```

- [ ] **Step 5: AgentState 구현**

Write `src/agent/state.py`:
```python
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.agent.types import Plan, Restaurant


class AgentState(TypedDict, total=False):
    # input
    query: str

    # memory
    user_profile: dict[str, Any]
    recent_visits: list[dict[str, Any]]

    # planning
    plan: Plan

    # search / aggregation
    candidates: list[Restaurant]        # raw from tools (with duplicates)
    aggregated: list[Restaurant]        # after dedup/merge/score/filter

    # reflection
    reflection_count: int
    reflection_passed: bool
    reflection_reason: str

    # output
    final_recommendation: list[Restaurant]
    final_text: str

    # ReAct conversation (auto-merged by add_messages)
    messages: Annotated[list[AnyMessage], add_messages]

    # bookkeeping for trace
    trace_log: list[dict[str, Any]]
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `uv run pytest tests/agent/test_types.py -v`
Expected: 3 passed.

- [ ] **Step 7: 커밋**

```bash
git add src/agent/types.py src/agent/state.py src/agent/__init__.py tests/agent/
git commit -m "feat(agent): shared Pydantic types (Restaurant, Plan, Weights) + AgentState"
```

---

## Task 4: 도구 — geocode (Kakao 주소 검색)

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/_http.py`
- Create: `src/tools/geocode.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_geocode.py`

- [ ] **Step 1: 패키지 디렉터리**

```bash
mkdir -p src/tools tests/tools
touch src/tools/__init__.py tests/tools/__init__.py
```

- [ ] **Step 2: 공용 HTTP 헬퍼 (모킹 친화)**

Write `src/tools/_http.py`:
```python
"""Shared httpx Client factory. Centralized so respx can patch one place
and tools share connection settings."""

from __future__ import annotations

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


def make_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)
```

- [ ] **Step 3: 실패하는 테스트 작성**

Write `tests/tools/test_geocode.py`:
```python
import httpx
import pytest
import respx

from src.tools.geocode import geocode


KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@respx.mock
def test_geocode_success(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "place_name": "전주 객사",
                        "address_name": "전북 전주시 완산구 중앙동3가",
                        "x": "127.1489",
                        "y": "35.8186",
                    }
                ]
            },
        )
    )
    result = geocode.invoke({"region": "전주 객사"})
    assert result["lat"] == pytest.approx(35.8186)
    assert result["lng"] == pytest.approx(127.1489)
    assert "전주" in result["matched_address"]


@respx.mock
def test_geocode_not_found(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(
        return_value=httpx.Response(200, json={"documents": []})
    )
    result = geocode.invoke({"region": "외계행성 뮤뮤성"})
    assert result["error"] == "region_not_found"
    assert "suggestions" in result


@respx.mock
def test_geocode_http_error(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(return_value=httpx.Response(500))
    result = geocode.invoke({"region": "전주 객사"})
    assert result["error"] == "kakao_http_error"


def test_geocode_missing_key(monkeypatch):
    monkeypatch.delenv("KAKAO_REST_API_KEY", raising=False)
    result = geocode.invoke({"region": "전주 객사"})
    assert result["error"] == "missing_api_key"
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `uv run pytest tests/tools/test_geocode.py -v`
Expected: ImportError.

- [ ] **Step 5: geocode 구현**

Write `src/tools/geocode.py`:
```python
from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from src.tools._http import make_client

KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@tool
def geocode(region: str) -> dict[str, Any]:
    """지역명·랜드마크·주소를 위경도 좌표로 변환한다.

    Args:
        region: 한국어 지역 표현 ("전주 객사", "홍대 정문", "서울시청" 등).
    Returns:
        성공: {"lat":..., "lng":..., "matched_address":...}
        실패: {"error":..., "message":..., "suggestions":[...]}
    """
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        return {"error": "missing_api_key",
                "message": "KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다."}

    try:
        with make_client() as client:
            resp = client.get(
                KAKAO_KEYWORD_URL,
                params={"query": region, "size": 5},
                headers={"Authorization": f"KakaoAK {api_key}"},
            )
    except Exception as e:  # noqa: BLE001
        return {"error": "kakao_network_error", "message": str(e)}

    if resp.status_code >= 400:
        return {"error": "kakao_http_error", "status": resp.status_code}

    docs = resp.json().get("documents", [])
    if not docs:
        return {
            "error": "region_not_found",
            "message": f"'{region}'에 해당하는 장소를 찾지 못했습니다.",
            "suggestions": [
                "지역명을 더 구체적으로 ('전주 객사' → '전주시 완산구 객사길')",
                "근처 랜드마크나 행정구역명으로 다시 입력해주세요",
            ],
        }

    top = docs[0]
    return {
        "lat": float(top["y"]),
        "lng": float(top["x"]),
        "matched_address": top.get("address_name") or top.get("road_address_name", ""),
        "matched_name": top.get("place_name", ""),
    }
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `uv run pytest tests/tools/test_geocode.py -v`
Expected: 4 passed.

- [ ] **Step 7: 커밋**

```bash
git add src/tools/__init__.py src/tools/_http.py src/tools/geocode.py tests/tools/
git commit -m "feat(tools): geocode tool (Kakao keyword search) with graceful errors"
```

---

## Task 5: 도구 — search_kakao_local

**Files:**
- Create: `src/tools/kakao_local.py`
- Create: `tests/tools/test_kakao_local.py`

- [ ] **Step 1: 테스트 작성**

Write `tests/tools/test_kakao_local.py`:
```python
import httpx
import pytest
import respx

from src.tools.kakao_local import search_kakao_local

URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@respx.mock
def test_returns_restaurants(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "id": "1",
                        "place_name": "백송갈비 객사점",
                        "category_name": "음식점 > 한식 > 육류,고기",
                        "address_name": "전북 전주시 완산구 ...",
                        "road_address_name": "전북 전주시 ...",
                        "x": "127.1489", "y": "35.8186",
                        "distance": "120",
                        "phone": "063-...",
                        "place_url": "http://place.map.kakao.com/1",
                    }
                ]
            },
        )
    )
    out = search_kakao_local.invoke(
        {"query": "저녁", "lat": 35.8186, "lng": 127.1489, "radius_m": 800}
    )
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비 객사점"
    assert r["source"] == "kakao"
    assert r["distance_m"] == 120
    assert r["category"] == "한식"


@respx.mock
def test_empty_results(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(return_value=httpx.Response(200, json={"documents": []}))
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["count"] == 0
    assert out["results"] == []


def test_missing_key(monkeypatch):
    monkeypatch.delenv("KAKAO_REST_API_KEY", raising=False)
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["error"] == "missing_api_key"


@respx.mock
def test_http_error_returns_empty_not_raises(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(return_value=httpx.Response(503))
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["error"] == "kakao_http_error"
    assert out["results"] == []
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/tools/test_kakao_local.py -v`
Expected: ImportError.

- [ ] **Step 3: 구현**

Write `src/tools/kakao_local.py`:
```python
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def _category_leaf(category_name: str | None) -> str | None:
    """'음식점 > 한식 > 육류,고기' → '한식' (두 번째 레벨이 가장 유용)."""
    if not category_name:
        return None
    parts = [p.strip() for p in category_name.split(">")]
    return parts[1] if len(parts) >= 2 else parts[-1]


@tool
def search_kakao_local(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 800,
    category_group_code: str = "FD6",
    sort: Literal["distance", "accuracy"] = "distance",
    size: int = 15,
) -> dict[str, Any]:
    """카카오 로컬 API로 좌표 반경 안의 음식점/카페를 검색한다.

    Args:
        query: 키워드 ("저녁", "비빔밥", "파스타").
        lat, lng: 검색 중심 좌표.
        radius_m: 반경(미터, 최대 20000).
        category_group_code: FD6=음식점, CE7=카페/디저트.
        sort: distance | accuracy.
        size: 최대 15.
    """
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        return {"error": "missing_api_key", "results": [], "count": 0}

    params = {
        "query": query,
        "x": lng, "y": lat,
        "radius": radius_m,
        "category_group_code": category_group_code,
        "sort": sort,
        "size": min(size, 15),
    }
    try:
        with make_client() as client:
            resp = client.get(URL, params=params,
                              headers={"Authorization": f"KakaoAK {api_key}"})
    except Exception as e:  # noqa: BLE001
        return {"error": "kakao_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "kakao_http_error", "status": resp.status_code,
                "results": [], "count": 0}

    docs = resp.json().get("documents", [])
    results = [{
        "id": f"k_{d.get('id', '')}",
        "name": d.get("place_name", ""),
        "source": "kakao",
        "category": _category_leaf(d.get("category_name")),
        "address": d.get("road_address_name") or d.get("address_name"),
        "lat": float(d["y"]) if d.get("y") else None,
        "lng": float(d["x"]) if d.get("x") else None,
        "distance_m": int(d["distance"]) if d.get("distance") else None,
        "phone": d.get("phone") or None,
        "url": d.get("place_url"),
    } for d in docs]
    return {"results": results, "count": len(results)}
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
uv run pytest tests/tools/test_kakao_local.py -v
git add src/tools/kakao_local.py tests/tools/test_kakao_local.py
git commit -m "feat(tools): search_kakao_local with FD6/CE7 category filter"
```

---

## Task 6: 도구 — search_naver_local

**Files:**
- Create: `src/tools/naver_local.py`
- Create: `tests/tools/test_naver_local.py`

- [ ] **Step 1: 테스트 작성**

Write `tests/tools/test_naver_local.py`:
```python
import httpx
import respx

from src.tools.naver_local import search_naver_local

URL = "https://openapi.naver.com/v1/search/local.json"


@respx.mock
def test_returns_results(monkeypatch):
    monkeypatch.setenv("NAVER_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "secret")
    respx.get(URL).mock(return_value=httpx.Response(200, json={
        "items": [{
            "title": "<b>백송</b>갈비",
            "category": "한식>육류,고기",
            "address": "전북 전주시 완산구",
            "roadAddress": "전북 전주시 ...",
            "mapx": "1271489000", "mapy": "358186000",
            "link": "http://...",
        }],
    }))
    out = search_naver_local.invoke({"query": "전주 객사 맛집"})
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비"            # HTML 태그 제거
    assert r["source"] == "naver"
    assert r["category"] == "한식"            # leaf 추출


def test_missing_keys(monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)
    out = search_naver_local.invoke({"query": "x"})
    assert out["error"] == "missing_api_key"
    assert out["results"] == []


@respx.mock
def test_http_error(monkeypatch):
    monkeypatch.setenv("NAVER_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "secret")
    respx.get(URL).mock(return_value=httpx.Response(500))
    out = search_naver_local.invoke({"query": "x"})
    assert out["error"] == "naver_http_error"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/tools/test_naver_local.py -v`

- [ ] **Step 3: 구현**

Write `src/tools/naver_local.py`:
```python
from __future__ import annotations

import os
import re
from typing import Any, Literal

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://openapi.naver.com/v1/search/local.json"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(text: str | None) -> str:
    return _TAG_RE.sub("", text or "")


def _category_leaf(category: str | None) -> str | None:
    if not category:
        return None
    return category.split(">")[0].strip()


@tool
def search_naver_local(
    query: str,
    sort: Literal["random", "comment"] = "comment",
) -> dict[str, Any]:
    """네이버 지역검색 API. display=5로 고정(API 제한).
    블로그 리뷰 풍부도 신호로 사용. sort='comment'면 리뷰 많은 곳 우선."""
    cid = os.getenv("NAVER_CLIENT_ID")
    cs = os.getenv("NAVER_CLIENT_SECRET")
    if not (cid and cs):
        return {"error": "missing_api_key", "results": [], "count": 0}

    try:
        with make_client() as client:
            resp = client.get(URL, params={"query": query, "display": 5, "sort": sort},
                              headers={"X-Naver-Client-Id": cid,
                                       "X-Naver-Client-Secret": cs})
    except Exception as e:  # noqa: BLE001
        return {"error": "naver_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "naver_http_error", "status": resp.status_code,
                "results": [], "count": 0}

    items = resp.json().get("items", [])
    results = []
    for i, it in enumerate(items):
        # Naver mapx/mapy are TM coords scaled by 10^7 → convert
        try:
            lng = float(it["mapx"]) / 10_000_000
            lat = float(it["mapy"]) / 10_000_000
        except (KeyError, TypeError, ValueError):
            lng = lat = None
        results.append({
            "id": f"n_{i}_{_strip(it.get('title',''))}",
            "name": _strip(it.get("title", "")),
            "source": "naver",
            "category": _category_leaf(it.get("category")),
            "address": it.get("roadAddress") or it.get("address"),
            "lat": lat, "lng": lng,
            "url": it.get("link"),
        })
    return {"results": results, "count": len(results)}
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
uv run pytest tests/tools/test_naver_local.py -v
git add src/tools/naver_local.py tests/tools/test_naver_local.py
git commit -m "feat(tools): search_naver_local (display=5 fixed, blog review signal)"
```

---

## Task 7: 도구 — search_google_places

**Files:**
- Create: `src/tools/google_places.py`
- Create: `tests/tools/test_google_places.py`

- [ ] **Step 1: 테스트 작성**

Write `tests/tools/test_google_places.py`:
```python
import httpx
import respx

from src.tools.google_places import search_google_places

URL = "https://places.googleapis.com/v1/places:searchText"


@respx.mock
def test_returns_results(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "places": [{
            "id": "ChIJxxx",
            "displayName": {"text": "백송갈비"},
            "formattedAddress": "전북 전주시 ...",
            "location": {"latitude": 35.8186, "longitude": 127.1489},
            "rating": 4.4,
            "userRatingCount": 218,
            "priceLevel": "PRICE_LEVEL_MODERATE",
            "primaryType": "korean_restaurant",
        }],
    }))
    out = search_google_places.invoke({
        "query": "전주 객사 저녁", "lat": 35.81, "lng": 127.14,
        "min_rating": 4.0, "price_levels": ["PRICE_LEVEL_MODERATE"],
    })
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비"
    assert r["rating"] == 4.4
    assert r["review_count"] == 218
    assert r["price_level"] == 2     # MODERATE → 2


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    out = search_google_places.invoke({"query": "x", "lat": 0, "lng": 0})
    assert out["error"] == "missing_api_key"


@respx.mock
def test_http_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(429))
    out = search_google_places.invoke({"query": "x", "lat": 0, "lng": 0})
    assert out["error"] == "google_http_error"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/tools/test_google_places.py -v`

- [ ] **Step 3: 구현**

Write `src/tools/google_places.py`:
```python
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.rating,places.userRatingCount,places.priceLevel,places.primaryType,"
    "places.websiteUri,places.nationalPhoneNumber"
)
PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


@tool
def search_google_places(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 800,
    included_type: str = "restaurant",
    price_levels: Optional[list[str]] = None,
    min_rating: Optional[float] = None,
    open_now: Optional[bool] = None,
    language_code: str = "ko",
) -> dict[str, Any]:
    """Google Places (New) Text Search. 평점·가격대를 API 단에서 거른다."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "missing_api_key", "results": [], "count": 0}

    body: dict[str, Any] = {
        "textQuery": query,
        "languageCode": language_code,
        "includedType": included_type,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius_m),
            }
        },
    }
    if price_levels:
        body["priceLevels"] = price_levels
    if min_rating is not None:
        body["minRating"] = min_rating
    if open_now is not None:
        body["openNow"] = open_now

    try:
        with make_client() as client:
            resp = client.post(URL, json=body, headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": FIELD_MASK,
                "Content-Type": "application/json",
            })
    except Exception as e:  # noqa: BLE001
        return {"error": "google_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "google_http_error", "status": resp.status_code,
                "results": [], "count": 0}

    places = resp.json().get("places", [])
    results = []
    for p in places:
        loc = p.get("location") or {}
        results.append({
            "id": f"g_{p.get('id', '')}",
            "name": (p.get("displayName") or {}).get("text", ""),
            "source": "google",
            "category": p.get("primaryType"),
            "address": p.get("formattedAddress"),
            "lat": loc.get("latitude"),
            "lng": loc.get("longitude"),
            "rating": p.get("rating"),
            "review_count": p.get("userRatingCount"),
            "price_level": PRICE_LEVEL_MAP.get(p.get("priceLevel")),
            "phone": p.get("nationalPhoneNumber"),
            "url": p.get("websiteUri"),
        })
    return {"results": results, "count": len(results)}
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
uv run pytest tests/tools/test_google_places.py -v
git add src/tools/google_places.py tests/tools/test_google_places.py
git commit -m "feat(tools): search_google_places (priceLevels/minRating API-side filter)"
```

---

## Task 8: Aggregator (dedup → merge → score → filter)

**Files:**
- Create: `src/agent/nodes/__init__.py`
- Create: `src/agent/nodes/aggregator.py`
- Create: `tests/agent/nodes/__init__.py`
- Create: `tests/agent/nodes/test_aggregator.py`

- [ ] **Step 1: 패키지 디렉터리**

```bash
mkdir -p src/agent/nodes tests/agent/nodes
touch src/agent/nodes/__init__.py tests/agent/nodes/__init__.py
```

- [ ] **Step 2: 테스트 작성**

Write `tests/agent/nodes/test_aggregator.py`:
```python
from datetime import datetime, timedelta

from src.agent.nodes.aggregator import aggregator_node
from src.agent.types import Plan, Restaurant


def _r(**kw):
    base = dict(id=kw.pop("id", "x"), name=kw.pop("name", "X"),
                source=kw.pop("source", "kakao"))
    return Restaurant(**base, **kw).model_dump()


def test_dedup_merges_same_restaurant_from_multiple_sources():
    state = {
        "plan": Plan(region_query="전주 객사").model_dump(),
        "candidates": [
            _r(id="k_1", name="백송갈비 객사점", source="kakao",
               category="한식", lat=35.8186, lng=127.1489),
            _r(id="g_1", name="백송갈비", source="google",
               rating=4.4, review_count=200, price_level=2,
               lat=35.8186, lng=127.1489),
            _r(id="n_1", name="백송갈비 본점", source="naver",
               review_count=150, lat=35.8186, lng=127.1489),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    agg = out["aggregated"]
    assert len(agg) == 1
    r = agg[0]
    assert r["source_count"] == 3
    assert r["rating"] == 4.4
    assert r["category"] == "한식"
    # review_count = max(naver=150, google=200) = 200
    assert r["review_count"] == 200


def test_excludes_disliked_categories():
    state = {
        "plan": Plan(
            region_query="전주",
            post_filters={"exclude_categories": ["해물"], "k": 3,
                          "exclude_visited_within_days": 1},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="해물탕집", source="kakao", category="해물", rating=4.5),
            _r(id="k_2", name="한식당", source="kakao", category="한식", rating=4.2),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "해물탕집" not in names
    assert "한식당" in names


def test_excludes_recent_within_window():
    now = datetime.now()
    state = {
        "plan": Plan(
            region_query="전주",
            post_filters={"exclude_visited_within_days": 1, "k": 3,
                          "exclude_categories": []},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="어제먹은집", source="kakao", category="한식", rating=4.5),
            _r(id="k_2", name="새로운집", source="kakao", category="한식", rating=4.2),
        ],
        "recent_visits": [
            {"name": "어제먹은집", "category": "한식",
             "visited_at": (now - timedelta(hours=20)).isoformat(),
             "source": "seed"},
        ],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "어제먹은집" not in names
    assert "새로운집" in names


def test_top_k_ordered_by_score():
    state = {
        "plan": Plan(region_query="전주",
                     post_filters={"k": 2, "exclude_categories": [],
                                   "exclude_visited_within_days": 1}).model_dump(),
        "candidates": [
            _r(id="k_1", name="A", source="kakao", category="한식",
               rating=3.0, review_count=10, distance_m=100, price_level=2),
            _r(id="k_2", name="B", source="kakao", category="한식",
               rating=4.8, review_count=500, distance_m=200, price_level=2),
            _r(id="k_3", name="C", source="kakao", category="한식",
               rating=4.0, review_count=100, distance_m=150, price_level=2),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    assert len(out["aggregated"]) == 2
    assert out["aggregated"][0]["name"] == "B"  # 최고점
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/agent/nodes/test_aggregator.py -v`
Expected: ImportError.

- [ ] **Step 4: aggregator 구현**

Write `src/agent/nodes/aggregator.py`:
```python
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from src.agent.types import (
    AggregationWeights,
    DEFAULT_WEIGHTS,
    Plan,
    Restaurant,
)


_PAREN_RE = re.compile(r"[\(\[].*?[\)\]]")
_PUNCT_RE = re.compile(r"[\s·\-,./]+")


def _normalize_name(name: str) -> str:
    """공백/괄호 부속어/구두점 제거 후 NFKC 정규화."""
    s = _PAREN_RE.sub("", name)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower().strip()


def _haversine_m(a_lat, a_lng, b_lat, b_lng) -> float | None:
    if None in (a_lat, a_lng, b_lat, b_lng):
        return None
    R = 6_371_000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def _same_restaurant(a: dict, b: dict, name_threshold: int = 85,
                     coord_threshold_m: float = 50.0) -> bool:
    name_sim = fuzz.token_set_ratio(_normalize_name(a["name"]),
                                    _normalize_name(b["name"]))
    if name_sim < name_threshold:
        return False
    dist = _haversine_m(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
    # 좌표 없으면 이름 유사도만으로 판단 (Naver가 일부 좌표 누락)
    return dist is None or dist <= coord_threshold_m


def _merge(group: list[dict]) -> dict:
    """소스별 강점에 따라 메타데이터 머지. group은 같은 식당 후보."""
    by_source = {r["source"]: r for r in group}
    kakao = by_source.get("kakao")
    naver = by_source.get("naver")
    google = by_source.get("google")
    head = kakao or naver or google or group[0]

    merged = dict(head)
    # name/address/category — Kakao 우선
    for k in ("name", "address", "category"):
        merged[k] = (kakao or {}).get(k) or (naver or {}).get(k) or (google or {}).get(k)
    # rating/price_level — Google 우선
    merged["rating"] = (google or {}).get("rating") or (kakao or {}).get("rating")
    merged["price_level"] = (google or {}).get("price_level")
    # review_count = max
    rev = [r.get("review_count") for r in group if r.get("review_count")]
    merged["review_count"] = max(rev) if rev else None
    # distance — kakao가 들고 있는 게 보통 가장 정확
    merged["distance_m"] = (kakao or {}).get("distance_m") or (google or {}).get("distance_m")
    # 좌표
    for k in ("lat", "lng"):
        merged[k] = (kakao or {}).get(k) or (google or {}).get(k) or (naver or {}).get(k)
    merged["source_count"] = len(by_source)
    merged["id"] = "+".join(sorted(by_source.keys())) + ":" + (merged["name"] or "")
    merged["source"] = head["source"]  # primary
    return merged


def _dedup_and_merge(raw: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for r in raw:
        if not r.get("name"):
            continue
        for g in groups:
            if _same_restaurant(g[0], r):
                g.append(r); break
        else:
            groups.append([r])
    return [_merge(g) for g in groups]


def _norm_clip(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _score(r: dict, w: AggregationWeights) -> float:
    rating = _norm_clip(r.get("rating") or 0.0, 0.0, 5.0)
    review = _norm_clip(math.log1p(r.get("review_count") or 0), 0.0, math.log1p(10_000))
    dist = r.get("distance_m")
    distance = _norm_clip(1 / ((dist or 1_000) + 50), 0.0, 1 / 50)
    match = (r.get("source_count") or 1) / 3
    price = _norm_clip(float(r.get("price_level") or 2), 0.0, 4.0)
    return (
        w.rating * rating
        + w.review * review
        + w.distance * distance
        + w.match * match
        - w.price * price
    )


def _is_recently_visited(name: str, recent: list[dict]) -> bool:
    target = _normalize_name(name)
    return any(_normalize_name(v["name"]) == target for v in recent)


def aggregator_node(state: dict[str, Any]) -> dict[str, Any]:
    """결정론적 코드 노드. LLM 호출 없음."""
    plan = Plan.model_validate(state["plan"])
    raw = state.get("candidates", [])
    user_dislikes = set(plan.post_filters.exclude_categories) | set(
        (state.get("user_profile") or {}).get("disliked_categories", [])
    )
    weights = plan.weights or DEFAULT_WEIGHTS

    merged = _dedup_and_merge(raw)

    # filter
    recent = state.get("recent_visits", [])
    filtered = []
    excluded_by_recency = 0
    excluded_by_category = 0
    for r in merged:
        if r.get("category") in user_dislikes:
            excluded_by_category += 1
            continue
        if _is_recently_visited(r["name"], recent):
            excluded_by_recency += 1
            continue
        filtered.append(r)

    # score + sort
    for r in filtered:
        r["score"] = _score(r, weights)
    filtered.sort(key=lambda r: r["score"], reverse=True)

    top_k = filtered[: plan.post_filters.k]
    return {
        "aggregated": [Restaurant.model_validate(r).model_dump() for r in top_k],
        "trace_log": state.get("trace_log", []) + [{
            "node": "aggregator",
            "raw_count": len(raw),
            "merged_count": len(merged),
            "excluded_by_category": excluded_by_category,
            "excluded_by_recency": excluded_by_recency,
            "kept": len(top_k),
        }],
    }
```

- [ ] **Step 5: 통과 확인 + 커밋**

```bash
uv run pytest tests/agent/nodes/test_aggregator.py -v
git add src/agent/nodes/__init__.py src/agent/nodes/aggregator.py tests/agent/nodes/
git commit -m "feat(agent): aggregator node (dedup/merge/score/filter, deterministic)"
```

---

## Task 9: 메모리 노드 (load_memory, save_memory)

**Files:**
- Create: `src/agent/nodes/load_memory.py`
- Create: `src/agent/nodes/save_memory.py`
- Create: `tests/agent/nodes/test_memory_nodes.py`

- [ ] **Step 1: 테스트 작성**

Write `tests/agent/nodes/test_memory_nodes.py`:
```python
from pathlib import Path

from src.agent.nodes.load_memory import load_memory_node
from src.agent.nodes.save_memory import save_memory_node
from src.memory.store import MemoryStore


def test_load_memory_pulls_profile_and_recent_visits(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["해물"])
    store.append_visit("어제집", "한식")

    out = load_memory_node({"query": "x"}, store=store, recency_days=7)
    assert out["user_profile"]["disliked_categories"] == ["해물"]
    assert len(out["recent_visits"]) == 1


def test_save_memory_appends_recommendations(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    state = {
        "final_recommendation": [
            {"name": "A", "category": "한식"},
            {"name": "B", "category": "한식"},
        ],
    }
    out = save_memory_node(state, store=store)
    assert out["trace_log"][-1]["saved"] == 2
    visits = store.get_recent_visits(within_days=1)
    assert {v["name"] for v in visits} == {"A", "B"}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/agent/nodes/test_memory_nodes.py -v`

- [ ] **Step 3: load_memory 구현**

Write `src/agent/nodes/load_memory.py`:
```python
from __future__ import annotations

from typing import Any

from src.memory.store import MemoryStore


def load_memory_node(state: dict[str, Any], *,
                     store: MemoryStore,
                     recency_days: int = 7) -> dict[str, Any]:
    """user_profile + recent_visits (지난 N일)을 state에 주입."""
    profile = store.all_user_profile()
    recent = store.get_recent_visits(within_days=recency_days)
    return {
        "user_profile": profile,
        "recent_visits": recent,
        "trace_log": state.get("trace_log", []) + [{
            "node": "load_memory",
            "profile_keys": list(profile.keys()),
            "recent_count": len(recent),
        }],
    }
```

- [ ] **Step 4: save_memory 구현**

Write `src/agent/nodes/save_memory.py`:
```python
from __future__ import annotations

from typing import Any

from src.memory.store import MemoryStore


def save_memory_node(state: dict[str, Any], *,
                     store: MemoryStore) -> dict[str, Any]:
    """final_recommendation을 visit_history에 append."""
    recs = state.get("final_recommendation") or []
    for r in recs:
        store.append_visit(
            name=r.get("name", ""),
            category=r.get("category"),
            source="recommended",
        )
    return {
        "trace_log": state.get("trace_log", []) + [{
            "node": "save_memory",
            "saved": len(recs),
        }],
    }
```

- [ ] **Step 5: 통과 확인 + 커밋**

```bash
uv run pytest tests/agent/nodes/test_memory_nodes.py -v
git add src/agent/nodes/load_memory.py src/agent/nodes/save_memory.py tests/agent/nodes/test_memory_nodes.py
git commit -m "feat(agent): load_memory + save_memory nodes"
```

---

## Task 10: planner 노드 + 프롬프트

**Files:**
- Create: `src/agent/prompts.py`
- Create: `src/agent/nodes/planner.py`
- Create: `tests/agent/nodes/test_planner.py`

- [ ] **Step 1: 프롬프트 모듈**

Write `src/agent/prompts.py`:
```python
"""모든 LLM 노드의 시스템 프롬프트. 한 곳에 모아두어 튜닝 용이."""

PLANNER_SYSTEM = """\
당신은 한국 맛집 추천 에이전트의 '계획 수립' 노드다.
사용자의 자연어 요청과 메모리(선호/비선호/최근 방문)를 받아,
검색 도구를 어떻게 부를지 JSON으로만 응답하라.

응답 스키마 (모두 필수, 일부는 null/기본값 허용):
{
  "region_query": str,                    // geocode 입력
  "needs_geocoding": bool,
  "food_keywords": [str, ...],            // 검색 키워드
  "kakao": {"category_group_code":"FD6"|"CE7", "radius_m":int, "sort":"distance"|"accuracy", "size":int},
  "naver": {"sort":"random"|"comment"},
  "google": {"included_type":str, "price_levels":[str]|null,
             "min_rating":float|null, "open_now":bool|null, "language_code":"ko"},
  "post_filters": {"exclude_categories":[str], "exclude_visited_within_days":int, "k":int},
  "weights": {"rating":float, "review":float, "distance":float, "match":float, "price":float},
  "clarification_needed": [str, ...]      // 입력이 너무 모호하면 채워라
}

가이드:
- "너무 비싸지 않다" → google.price_levels=["PRICE_LEVEL_INEXPENSIVE","PRICE_LEVEL_MODERATE"]
- "리뷰가 좋은" → weights.rating ↑(0.40+), weights.review ↑(0.30+), google.min_rating=4.0
- "친구랑 저녁/모임" → kakao.category_group_code="FD6", food_keywords에 "저녁"/"모임" 포함
- "디저트/카페" → kakao.category_group_code="CE7"
- "걸어서 N분" → radius_m = N*70 (보수적)
- 사용자가 명시 안 한 값은 합리적 기본값으로
- 메모리의 disliked_categories는 반드시 post_filters.exclude_categories에 머지
- weights 합은 1.0 근처여야 한다 (price는 마이너스 가중)
"""

REFLECTOR_SYSTEM = """\
당신은 한국 맛집 추천 에이전트의 '자가검토' 노드다.
계획(plan)과 종합 결과(aggregated)를 보고 다음을 확인하라.

체크리스트:
1. 후보 수 >= post_filters.k
2. 모두 가격 조건 충족 (price_level <= max(price_levels의 숫자))
3. 모두 평점 조건 충족 (rating >= min_rating, 둘 다 있을 때만)
4. 비선호 카테고리와 충돌 없음
5. 최근 방문 윈도우와 충돌 없음

응답은 JSON만:
{
  "passed": bool,
  "reason": str,                 // 한국어로 간단히
  "suggested_relaxation": {      // passed=false일 때만
    "kakao_radius_delta_m": int|null,    // 예: +400
    "google_min_rating_delta": float|null, // 예: -0.3
    "google_drop_price_filter": bool,
    "post_filters_exclude_categories_remove": [str]
  } | null
}
"""

FINALIZER_SYSTEM = """\
당신은 한국 맛집 추천 에이전트의 '최종 답변' 노드다.
입력으로 사용자 원본 요청 + 추천 후보(top-k)를 받는다.
각 후보에 대해 1-2문장의 추천 이유를 자연스러운 한국어로 작성하라.

출력 형식 (마크다운):
1. **<식당명>** (<카테고리>, ★<rating>, 도보 <km/m>)
   <이유 한 문장>
2. ...
3. ...

마지막에 적용된 가정(예산/평점 등)이 있다면 짧게 명시. 광고성 표현 금지.
"""
```

- [ ] **Step 2: 테스트 작성 (LLM 모킹)**

Write `tests/agent/nodes/test_planner.py`:
```python
import json
from unittest.mock import MagicMock

from src.agent.nodes.planner import planner_node
from src.agent.types import Plan


def test_planner_returns_validated_plan():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "food_keywords": ["저녁"],
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance", "size": 15},
        "naver": {"sort": "comment"},
        "google": {
            "included_type": "restaurant",
            "price_levels": ["PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"],
            "min_rating": 4.0, "open_now": None, "language_code": "ko",
        },
        "post_filters": {"exclude_categories": ["해물"],
                         "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.45, "review": 0.30, "distance": 0.05,
                    "match": 0.10, "price": 0.10},
        "clarification_needed": [],
    }))

    state = {
        "query": "전주 객사 근처...",
        "user_profile": {"disliked_categories": ["해물"]},
        "recent_visits": [],
    }
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.region_query == "전주 객사"
    assert "해물" in plan.post_filters.exclude_categories


def test_planner_force_merges_user_dislikes_even_if_llm_forgets():
    """LLM이 깜빡해도 코드가 user_profile.disliked_categories를 머지."""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "region_query": "전주",
        "needs_geocoding": True,
        "food_keywords": [],
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance", "size": 15},
        "naver": {"sort": "comment"},
        "google": {"included_type": "restaurant", "price_levels": None,
                   "min_rating": None, "open_now": None, "language_code": "ko"},
        "post_filters": {"exclude_categories": [],  # LLM이 빼먹음
                         "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                    "match": 0.15, "price": 0.10},
        "clarification_needed": [],
    }))

    state = {"query": "x",
             "user_profile": {"disliked_categories": ["해물", "회"]},
             "recent_visits": []}
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert set(plan.post_filters.exclude_categories) >= {"해물", "회"}
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/agent/nodes/test_planner.py -v`

- [ ] **Step 4: planner 구현**

Write `src/agent/nodes/planner.py`:
```python
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import PLANNER_SYSTEM
from src.agent.types import Plan


def planner_node(state: dict[str, Any], *, llm) -> dict[str, Any]:
    """LLM에 plan JSON 받기. user_profile.disliked_categories는 코드로 강제 머지."""
    user_msg = (
        f"사용자 요청: {state.get('query', '')}\n"
        f"사용자 프로필: {json.dumps(state.get('user_profile', {}), ensure_ascii=False)}\n"
        f"최근 방문(요약): "
        f"{[v.get('name') for v in state.get('recent_visits', [])]}\n"
        "위 정보를 토대로 계획 JSON만 응답하라."
    )
    resp = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    raw = resp.content
    plan_dict = json.loads(raw)

    # ── 코드 가드레일: 비선호 카테고리 강제 머지 ──
    user_dislikes = set(
        (state.get("user_profile") or {}).get("disliked_categories", []) or []
    )
    pf = plan_dict.setdefault("post_filters", {})
    existing = set(pf.get("exclude_categories") or [])
    pf["exclude_categories"] = sorted(existing | user_dislikes)

    plan = Plan.model_validate(plan_dict)
    return {
        "plan": plan.model_dump(),
        "trace_log": state.get("trace_log", []) + [{
            "node": "planner",
            "plan": plan.model_dump(),
        }],
    }
```

- [ ] **Step 5: 통과 확인 + 커밋**

```bash
uv run pytest tests/agent/nodes/test_planner.py -v
git add src/agent/prompts.py src/agent/nodes/planner.py tests/agent/nodes/test_planner.py
git commit -m "feat(agent): planner node (LLM JSON plan + dislikes guardrail)"
```

---

## Task 11: react_agent + reflector + finalizer 노드

**Files:**
- Create: `src/agent/nodes/react_agent.py`
- Create: `src/agent/nodes/reflector.py`
- Create: `src/agent/nodes/finalizer.py`
- Create: `tests/agent/nodes/test_reflector.py`
- Create: `tests/agent/nodes/test_finalizer.py`

- [ ] **Step 1: react_agent 노드 — ReAct 루프**

Write `src/agent/nodes/react_agent.py`:
```python
"""React loop: LLM이 도구를 자율 호출. LangGraph의 create_react_agent를 활용.

검색 도구만 노출 (geocode/kakao/naver/google). 결과는 messages에 누적되고,
이후 코드가 ToolMessage를 파싱해서 candidates로 평탄화한다."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import create_react_agent

from src.agent.types import Plan

REACT_SYSTEM = """\
당신은 한국 맛집 검색 에이전트다. 사용자의 plan(JSON)을 받아 도구를 호출해서
후보 식당 풀을 모은다.

핵심 규칙:
1. plan.needs_geocoding == true 면 geocode를 가장 먼저 호출.
2. geocode 결과의 lat/lng로 search_kakao_local과 search_google_places를 호출 (병렬 가능).
3. search_naver_local은 좌표 없이 query만 받는다.
4. 도구 호출은 plan의 파라미터를 그대로 사용. 임의로 값 바꾸지 말 것.
5. 모든 검색이 끝나면 도구 호출 없이 'DONE'이라고만 답변하라.
"""


def make_react_agent(llm, tools: list, recursion_limit: int = 16) -> Callable:
    """create_react_agent를 노드 함수로 래핑."""
    agent = create_react_agent(model=llm, tools=tools)

    def node(state: dict[str, Any]) -> dict[str, Any]:
        plan = Plan.model_validate(state["plan"])
        first_msg = HumanMessage(content=(
            f"plan: {json.dumps(plan.model_dump(), ensure_ascii=False)}\n"
            f"사용자 원본: {state.get('query', '')}\n"
            "이 plan대로 도구를 호출해서 후보를 모아라."
        ))
        result = agent.invoke(
            {"messages": [SystemMessage(content=REACT_SYSTEM), first_msg]},
            {"recursion_limit": recursion_limit},
        )
        msgs = result["messages"]
        candidates = _extract_candidates(msgs)
        return {
            "messages": msgs,
            "candidates": candidates,
            "trace_log": state.get("trace_log", []) + [{
                "node": "react_agent",
                "messages_count": len(msgs),
                "candidates_count": len(candidates),
            }],
        }

    return node


def _extract_candidates(messages: list) -> list[dict]:
    """ToolMessage(검색 결과)에서 results 평탄화."""
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            payload = json.loads(m.content) if isinstance(m.content, str) else m.content
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "results" in payload:
            out.extend(payload["results"])
    return out
```

- [ ] **Step 2: reflector 노드 + 테스트**

Write `tests/agent/nodes/test_reflector.py`:
```python
import json
from unittest.mock import MagicMock

from src.agent.nodes.reflector import reflector_node


def test_passes_when_enough_candidates():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "passed": True, "reason": "조건 충족", "suggested_relaxation": None,
    }))
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1}},
        "aggregated": [{"name": f"r{i}"} for i in range(3)],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["reflection_passed"] is True
    assert out["reflection_count"] == 1


def test_fail_returns_relaxation_and_bumps_count():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "passed": False, "reason": "후보 부족",
        "suggested_relaxation": {
            "kakao_radius_delta_m": 400,
            "google_min_rating_delta": -0.3,
            "google_drop_price_filter": False,
            "post_filters_exclude_categories_remove": [],
        },
    }))
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1},
                 "kakao": {"radius_m": 800, "category_group_code": "FD6",
                           "sort": "distance", "size": 15},
                 "naver": {"sort": "comment"},
                 "google": {"included_type": "restaurant",
                            "price_levels": None, "min_rating": 4.0,
                            "open_now": None, "language_code": "ko"},
                 "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                             "match": 0.15, "price": 0.10},
                 "needs_geocoding": True, "food_keywords": [],
                 "clarification_needed": []},
        "aggregated": [],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["reflection_passed"] is False
    assert out["reflection_count"] == 1
    # plan was relaxed
    assert out["plan"]["kakao"]["radius_m"] == 1200
    assert abs(out["plan"]["google"]["min_rating"] - 3.7) < 1e-9
```

Write `src/agent/nodes/reflector.py`:
```python
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import REFLECTOR_SYSTEM
from src.agent.types import Plan

MAX_REFLECTION = 2


def _apply_relaxation(plan_dict: dict[str, Any], relax: dict[str, Any]) -> dict[str, Any]:
    p = dict(plan_dict)
    if relax.get("kakao_radius_delta_m"):
        p["kakao"] = {**p["kakao"], "radius_m":
                      int(p["kakao"]["radius_m"]) + int(relax["kakao_radius_delta_m"])}
    if relax.get("google_min_rating_delta") and p["google"].get("min_rating") is not None:
        p["google"] = {**p["google"],
                       "min_rating": float(p["google"]["min_rating"]) +
                       float(relax["google_min_rating_delta"])}
    if relax.get("google_drop_price_filter"):
        p["google"] = {**p["google"], "price_levels": None}
    drops = set(relax.get("post_filters_exclude_categories_remove") or [])
    if drops:
        p["post_filters"] = {
            **p["post_filters"],
            "exclude_categories": [
                c for c in p["post_filters"]["exclude_categories"] if c not in drops
            ],
        }
    return p


def reflector_node(state: dict[str, Any], *, llm) -> dict[str, Any]:
    plan = Plan.model_validate(state["plan"])
    aggregated = state.get("aggregated", [])
    count = int(state.get("reflection_count", 0))

    user_msg = (
        f"plan: {json.dumps(plan.model_dump(), ensure_ascii=False)}\n"
        f"aggregated_count: {len(aggregated)}\n"
        f"aggregated: {json.dumps(aggregated, ensure_ascii=False)[:2000]}\n"
        "위를 보고 체크리스트 평가 JSON만 응답하라."
    )
    resp = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    eval_ = json.loads(resp.content)

    new_count = count + 1
    out: dict[str, Any] = {
        "reflection_count": new_count,
        "reflection_passed": bool(eval_.get("passed")),
        "reflection_reason": eval_.get("reason", ""),
        "trace_log": state.get("trace_log", []) + [{
            "node": "reflector",
            "passed": bool(eval_.get("passed")),
            "reason": eval_.get("reason", ""),
            "reflection_count": new_count,
        }],
    }

    # passed=false이고 아직 여유 있으면 plan 완화
    if not eval_.get("passed") and new_count < MAX_REFLECTION \
            and eval_.get("suggested_relaxation"):
        out["plan"] = _apply_relaxation(plan.model_dump(),
                                        eval_["suggested_relaxation"])
    return out


def should_retry(state: dict[str, Any]) -> str:
    """conditional edge에서 사용. retry|finalize 분기."""
    if state.get("reflection_passed"):
        return "finalize"
    if int(state.get("reflection_count", 0)) >= MAX_REFLECTION:
        return "finalize"
    return "retry"
```

- [ ] **Step 3: finalizer 노드 + 테스트**

Write `tests/agent/nodes/test_finalizer.py`:
```python
from unittest.mock import MagicMock

from src.agent.nodes.finalizer import finalizer_node


def test_finalizer_builds_text_and_final_recommendation():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=(
        "1. **A** (한식, ★4.5, 도보 3분)\n   친구와 가기 좋은 가성비 한식\n"
        "2. **B** (...)\n3. **C** (...)\n\n적용 가정: 가격대 모더레이트"
    ))
    state = {
        "query": "전주 객사 근처...",
        "aggregated": [
            {"name": "A", "category": "한식", "rating": 4.5,
             "distance_m": 200, "price_level": 2, "review_count": 100,
             "source_count": 2, "id": "x", "source": "kakao", "score": 0.7},
            {"name": "B", "category": "한식", "rating": 4.3,
             "distance_m": 300, "price_level": 2, "review_count": 80,
             "source_count": 1, "id": "y", "source": "kakao", "score": 0.6},
            {"name": "C", "category": "한식", "rating": 4.2,
             "distance_m": 400, "price_level": 2, "review_count": 60,
             "source_count": 1, "id": "z", "source": "kakao", "score": 0.5},
        ],
    }
    out = finalizer_node(state, llm=fake_llm)
    assert len(out["final_recommendation"]) == 3
    assert "1." in out["final_text"]
```

Write `src/agent/nodes/finalizer.py`:
```python
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import FINALIZER_SYSTEM


def finalizer_node(state: dict[str, Any], *, llm) -> dict[str, Any]:
    aggregated = state.get("aggregated", [])
    user_msg = (
        f"사용자 원본 요청: {state.get('query', '')}\n"
        f"추천 후보 (top {len(aggregated)}):\n"
        f"{json.dumps(aggregated, ensure_ascii=False, indent=2)}"
    )
    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    return {
        "final_recommendation": aggregated,
        "final_text": resp.content,
        "trace_log": state.get("trace_log", []) + [{
            "node": "finalizer",
            "k": len(aggregated),
        }],
    }
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
uv run pytest tests/agent/nodes/test_reflector.py tests/agent/nodes/test_finalizer.py -v
git add src/agent/nodes/react_agent.py src/agent/nodes/reflector.py src/agent/nodes/finalizer.py tests/agent/nodes/test_reflector.py tests/agent/nodes/test_finalizer.py
git commit -m "feat(agent): react_agent loop + reflector (relaxation) + finalizer"
```

---

## Task 12: 그래프 조립 (`src/agent/graph.py`)

**Files:**
- Create: `src/agent/graph.py`
- Create: `tests/agent/test_graph.py`

- [ ] **Step 1: 테스트 작성 (모든 LLM/도구 모킹, 그래프 토폴로지 검증)**

Write `tests/agent/test_graph.py`:
```python
"""그래프 빌더 스모크 테스트: 노드/엣지 토폴로지 + 한 번 invoke 가능한지."""

from unittest.mock import MagicMock

from src.agent.graph import build_graph
from src.memory.store import MemoryStore


def test_graph_compiles_and_invokes(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")

    fake_llm = MagicMock()
    # planner returns minimal valid plan
    fake_llm.invoke.return_value = MagicMock(content=(
        '{"region_query":"x","needs_geocoding":false,"food_keywords":[],'
        '"kakao":{"category_group_code":"FD6","radius_m":800,"sort":"distance","size":15},'
        '"naver":{"sort":"comment"},'
        '"google":{"included_type":"restaurant","price_levels":null,"min_rating":null,'
        '"open_now":null,"language_code":"ko"},'
        '"post_filters":{"exclude_categories":[],"exclude_visited_within_days":1,"k":3},'
        '"weights":{"rating":0.35,"review":0.25,"distance":0.15,"match":0.15,"price":0.10},'
        '"clarification_needed":[]}'
    ))

    # react_agent를 통째로 스텁: 빈 candidates만 채워주는 함수로 대체
    def stub_react_node(state):
        return {"candidates": []}

    store = MemoryStore(tmp_path / "g.db")
    graph = build_graph(
        llm=fake_llm,
        store=store,
        react_node_override=stub_react_node,
    )
    # passed=true 응답 시퀀스 (reflector에서 즉시 통과)
    fake_llm.invoke.side_effect = [
        # planner
        fake_llm.invoke.return_value,
        # reflector
        MagicMock(content='{"passed":true,"reason":"ok","suggested_relaxation":null}'),
        # finalizer
        MagicMock(content="1. ...\n2. ...\n3. ..."),
    ]

    result = graph.invoke({"query": "test"})
    assert "final_text" in result
    assert isinstance(result.get("trace_log"), list)
```

- [ ] **Step 2: 그래프 구현**

Write `src/agent/graph.py`:
```python
from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.aggregator import aggregator_node
from src.agent.nodes.finalizer import finalizer_node
from src.agent.nodes.load_memory import load_memory_node
from src.agent.nodes.planner import planner_node
from src.agent.nodes.react_agent import make_react_agent
from src.agent.nodes.reflector import reflector_node, should_retry
from src.agent.nodes.save_memory import save_memory_node
from src.agent.state import AgentState
from src.memory.store import MemoryStore
from src.tools.geocode import geocode
from src.tools.google_places import search_google_places
from src.tools.kakao_local import search_kakao_local
from src.tools.naver_local import search_naver_local


def build_graph(
    *,
    llm,
    store: MemoryStore,
    checkpointer: Optional[SqliteSaver] = None,
    recency_days: int = 7,
    react_node_override: Optional[Callable] = None,
):
    """그래프 컴파일. react_node_override는 테스트에서 LLM 의존 제거용."""

    react_node = react_node_override or make_react_agent(
        llm=llm,
        tools=[geocode, search_kakao_local, search_naver_local, search_google_places],
    )

    g = StateGraph(AgentState)
    g.add_node("load_memory", lambda s: load_memory_node(s, store=store,
                                                         recency_days=recency_days))
    g.add_node("planner", lambda s: planner_node(s, llm=llm))
    g.add_node("react_agent", react_node)
    g.add_node("aggregator", aggregator_node)
    g.add_node("reflector", lambda s: reflector_node(s, llm=llm))
    g.add_node("finalizer", lambda s: finalizer_node(s, llm=llm))
    g.add_node("save_memory", lambda s: save_memory_node(s, store=store))

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "planner")
    g.add_edge("planner", "react_agent")
    g.add_edge("react_agent", "aggregator")
    g.add_edge("aggregator", "reflector")
    g.add_conditional_edges("reflector", should_retry, {
        "retry": "react_agent",
        "finalize": "finalizer",
    })
    g.add_edge("finalizer", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
```

- [ ] **Step 3: 통과 확인 + 커밋**

```bash
uv run pytest tests/agent/test_graph.py -v
git add src/agent/graph.py tests/agent/test_graph.py
git commit -m "feat(agent): StateGraph assembly (8 nodes + conditional retry edge)"
```

---

## Task 13: Rich CLI 렌더러 + trace.md 생성기

**Files:**
- Create: `src/ui/__init__.py`
- Create: `src/ui/renderer.py`
- Create: `tests/ui/__init__.py`
- Create: `tests/ui/test_renderer.py`

- [ ] **Step 1: 패키지 디렉터리**

```bash
mkdir -p src/ui tests/ui
touch src/ui/__init__.py tests/ui/__init__.py
```

- [ ] **Step 2: 테스트 작성**

Write `tests/ui/test_renderer.py`:
```python
from src.ui.renderer import render_trace_md


def test_render_trace_md_includes_each_node():
    trace = [
        {"node": "load_memory", "profile_keys": ["disliked_categories"], "recent_count": 1},
        {"node": "planner", "plan": {"region_query": "전주 객사"}},
        {"node": "react_agent", "messages_count": 8, "candidates_count": 18},
        {"node": "aggregator", "raw_count": 18, "merged_count": 12,
         "excluded_by_category": 2, "excluded_by_recency": 1, "kept": 3},
        {"node": "reflector", "passed": True, "reason": "ok",
         "reflection_count": 1},
        {"node": "finalizer", "k": 3},
        {"node": "save_memory", "saved": 3},
    ]
    md = render_trace_md(
        query="전주 객사 근처...",
        final_text="1. A\n2. B\n3. C",
        trace_log=trace,
    )
    for node in ("load_memory", "planner", "react_agent", "aggregator",
                 "reflector", "finalizer", "save_memory"):
        assert node in md
    assert "전주 객사" in md
    assert "## Final" in md
```

- [ ] **Step 3: 구현**

Write `src/ui/renderer.py`:
```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def announce_query(query: str) -> None:
    console.print(Rule(style="cyan"))
    console.print(Panel(query, title="[bold cyan]User Query", border_style="cyan"))


def log_node(name: str, summary: str) -> None:
    console.print(f"[green]▸[/] [bold]{name}[/]  {summary}")


def announce_recommendation(text: str) -> None:
    console.print(Rule(style="magenta"))
    console.print(Panel(text, title="[bold magenta]Recommendation",
                        border_style="magenta"))


def announce_trace_url(url: str | None) -> None:
    if url:
        console.print(f"[dim][phoenix] trace: {url}[/]")


def render_trace_md(*, query: str, final_text: str,
                    trace_log: list[dict[str, Any]]) -> str:
    """제출용 trace.md 본문 생성."""
    lines = [
        "# Execution Trace",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## User Query",
        "",
        f"> {query}",
        "",
        "## Node Trace",
        "",
        "| # | Node | Summary |",
        "|--|--|--|",
    ]
    for i, entry in enumerate(trace_log, 1):
        node = entry.get("node", "?")
        summary = {k: v for k, v in entry.items() if k != "node"}
        lines.append(f"| {i} | `{node}` | `{json.dumps(summary, ensure_ascii=False)}` |")

    lines += ["", "## Final", "", final_text, ""]
    return "\n".join(lines)


def write_trace_md(*, query: str, final_text: str,
                   trace_log: list[dict[str, Any]],
                   out_dir: Path = Path("docs/traces")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"trace_{ts}.md"
    path.write_text(
        render_trace_md(query=query, final_text=final_text, trace_log=trace_log),
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
uv run pytest tests/ui/test_renderer.py -v
git add src/ui/ tests/ui/
git commit -m "feat(ui): Rich console helpers + trace.md generator"
```

---

## Task 14: main.py + 종단 스모크 실행

**Files:**
- Create: `main.py`

- [ ] **Step 1: main.py 작성**

Write `/home/rheon/Desktop/projects/OSS/Assign4/main.py`:
```python
"""맛집 추천 에이전트 CLI.

사용:
    uv run python main.py "전주 객사 근처에서 친구랑 저녁..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Phoenix 트레이스는 LangChain import 전에 초기화해야 자동 instrumentation이 잡힘.
from src.observability.telemetry import init_telemetry, flush_telemetry  # noqa: E402
init_telemetry()

from langchain_openai import ChatOpenAI  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402
from src.ui.renderer import (  # noqa: E402
    announce_query,
    announce_recommendation,
    announce_trace_url,
    log_node,
    write_trace_md,
)


DEFAULT_QUERY = (
    "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. "
    "너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
)


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY 가 .env에 없습니다.", file=sys.stderr)
        return 2
    if not os.getenv("KAKAO_REST_API_KEY"):
        print("[warn] KAKAO_REST_API_KEY 누락 — geocode/kakao 검색이 비활성됩니다.",
              file=sys.stderr)

    announce_query(query)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    store = MemoryStore(Path("data/agent_memory.db"))
    graph = build_graph(llm=llm, store=store)

    try:
        result = graph.invoke({"query": query})
    except Exception as e:  # noqa: BLE001
        print(f"[error] graph invoke failed: {e}", file=sys.stderr)
        flush_telemetry()
        return 1

    # 콘솔에 노드별 한 줄 요약
    for entry in result.get("trace_log", []):
        node = entry.get("node", "?")
        rest = {k: v for k, v in entry.items() if k != "node"}
        log_node(node, str(rest))

    final_text = result.get("final_text", "(no answer)")
    announce_recommendation(final_text)

    trace_path = write_trace_md(
        query=query, final_text=final_text,
        trace_log=result.get("trace_log", []),
    )
    print(f"[trace] saved: {trace_path}")

    project = os.getenv("PHOENIX_PROJECT_NAME", "restaurant-recommender-agent")
    base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").replace("/api/collect", "")
    if base and os.getenv("PHOENIX_API_KEY"):
        announce_trace_url(f"{base}/projects/{project}")

    flush_telemetry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 전체 테스트 스위트 실행**

Run: `uv run pytest -v`
Expected: 모든 테스트 통과 (Task 1~13의 합).

- [ ] **Step 3: 모킹 없이 종단 실행 — 시나리오**

전제: `.env`에 `OPENAI_API_KEY`, `KAKAO_REST_API_KEY` 입력됨.

Run:
```bash
uv run python scripts/seed_memory.py
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```
Expected:
- `[phoenix]` 또는 키 누락 경고 한 줄
- `User Query` 패널
- `▸ load_memory ... ▸ planner ... ▸ react_agent ... ▸ aggregator ... ▸ reflector ... ▸ finalizer ... ▸ save_memory` 순차 로그
- `Recommendation` 패널 3곳
- `[trace] saved: docs/traces/trace_<ts>.md`
- (Phoenix 키 있으면) trace URL

- [ ] **Step 4: 산출물 확인**

Run: `ls -la docs/traces/ data/`
Expected: `trace_*.md` 1건 이상, `agent_memory.db`.

- [ ] **Step 5: 커밋**

```bash
git add main.py docs/traces/
git commit -m "feat: main.py CLI entry + end-to-end smoke run with trace.md output"
```

---

## Task 15: README 완성 (시연 시나리오 5종 + Phoenix 안내)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README 본문 보강**

Replace `/home/rheon/Desktop/projects/OSS/Assign4/README.md` 내용을 다음으로 교체:
```markdown
# 맛집 추천 AI Agent (OSS Assign4)

LangGraph 기반 ReAct 에이전트. 사용자 요청을 분석해 외부 API(Kakao Local / Naver Search / Google Places)를 호출하고, 결과를 종합·검증해 추천 3곳을 생성한다.

자세한 설계: [`docs/superpowers/specs/2026-05-28-restaurant-agent-design.md`](docs/superpowers/specs/2026-05-28-restaurant-agent-design.md)

## 적용된 Agentic Design Pattern (5개)

| 패턴 | 위치 | 동작 |
|---|---|---|
| **ReAct** | `src/agent/nodes/react_agent.py` | Thought→Action→Observation 자율 루프 |
| **Tool Use** | `src/tools/*` (geocode, kakao, naver, google) | 4개 도구 병렬 호출 |
| **Plan-and-Solve** | `src/agent/nodes/planner.py` | 입력+메모리를 구조화 JSON plan으로 분해 |
| **Memory** | `src/agent/nodes/{load,save}_memory.py` + SQLite | 선호/비선호 + 방문 기록 |
| **Reflection** | `src/agent/nodes/reflector.py` | 결과 자가검증, 부족 시 plan 완화 후 재시도 (max 2) |

## 실행

```bash
# 1. 의존성
uv sync

# 2. 환경변수
cp .env.example .env
$EDITOR .env   # OPENAI_API_KEY, KAKAO_REST_API_KEY 필수

# 3. 메모리 시드 (Memory 패턴 시연용)
uv run python scripts/seed_memory.py

# 4. 메인 시나리오 실행
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

## API 키

| 키 | 필수 여부 | 발급 |
|---|---|---|
| `OPENAI_API_KEY` | 필수 | platform.openai.com |
| `KAKAO_REST_API_KEY` | 필수 | developers.kakao.com |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | 선택 | developers.naver.com (검색 → 지역) |
| `GOOGLE_PLACES_API_KEY` | 선택 | console.cloud.google.com (Places API New) |
| `PHOENIX_*` | 선택 | self-hosted 인스턴스, 없으면 트레이스만 안 보냄 |

키가 일부만 있어도 동작합니다 — 빠진 도구는 비활성화되고 plan이 자동 조정됩니다.

## 시연 시나리오

```bash
# 메인 (스펙 명시)
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."

# 에러 처리 시연
uv run python main.py "외계행성 뮤뮤성 맛집"           # → region_not_found
uv run python main.py "맛집"                          # → 모호 → clarification_needed
uv run python main.py "전주 객사 회 맛집"             # → 메모리 비선호로 0건 → reflector 완화
PHOENIX_DISABLED=1 uv run python main.py "..."        # → 트레이스 없이 빠른 실행
```

## 출력물

- **콘솔**: Rich 기반 노드별 진행 상황
- **`docs/traces/trace_<ts>.md`**: 노드/도구/입력/결과/duration 마크다운 표 (제출용)
- **Phoenix UI**: `https://phoenix.rheon.kr/projects/restaurant-recommender-agent` — span tree, 패턴별로 또렷이 분리됨

## 테스트

```bash
uv run pytest -v
```

## 파일 구조 (요약)

```
src/
├── observability/telemetry.py     # Phoenix init/flush (LangChain import 전 호출)
├── agent/
│   ├── graph.py                   # StateGraph 조립
│   ├── state.py, types.py         # AgentState / Restaurant / Plan
│   ├── prompts.py                 # planner/reflector/finalizer 시스템 프롬프트
│   └── nodes/{load_memory,planner,react_agent,aggregator,
│              reflector,finalizer,save_memory}.py
├── tools/{geocode,kakao_local,naver_local,google_places}.py
├── memory/store.py                # SQLite (user_profile + visit_history)
└── ui/renderer.py                 # Rich 콘솔 + trace.md
```
```

- [ ] **Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: README with patterns map, demo scenarios, key matrix"
```

---

## 자기 검토 체크리스트 (구현 시작 전 필독)

- [ ] `init_telemetry()`는 `from langchain_openai import ...` **이전** 라인에 있어야 한다 (main.py에 명시됨).
- [ ] `MemoryStore`는 thread-safe하지 않다. 단일 프로세스/단일 스레드 CLI에서만 안전.
- [ ] `react_agent`가 recursion_limit를 초과하면 LangGraph는 `GraphRecursionError`를 던진다 — Task 14의 `main.py` try/except가 잡는다.
- [ ] `reflector`의 `MAX_REFLECTION=2`와 `should_retry`가 일관되게 동작하는지 graph 테스트가 검증한다.
- [ ] 도구 4개는 모두 환경변수 누락 시 raise 대신 `{"error": "missing_api_key", "results": [], "count": 0}` 반환 — react_agent가 안전하게 다음 단계 진행 가능.
- [ ] `.env`는 `.gitignore`에 이미 있다 (Task 0 첫 커밋).
- [ ] `data/agent_memory.db`도 `.gitignore`됨. 시드는 README 안내로 채점자가 직접 실행.
