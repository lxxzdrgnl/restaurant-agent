# MATZIP — 맛집 추천 AI Agent

```text
[phoenix] telemetry initialized → project="restaurant" endpoint=https://phoenix.rheon.kr/api/collect (auth)

███╗   ███╗ █████╗ ████████╗███████╗██╗██████╗
████╗ ████║██╔══██╗╚══██╔══╝╚══███╔╝██║██╔══██╗
██╔████╔██║███████║   ██║     ███╔╝ ██║██████╔╝
██║╚██╔╝██║██╔══██║   ██║    ███╔╝  ██║██╔═══╝
██║ ╚═╝ ██║██║  ██║   ██║   ███████╗██║██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝

LangGraph ReAct Agent for restaurant recommendation
Model: gpt-4o-mini · type /help for commands

현재 메모리: 비선호: 해물, 회 · 기본 예산: moderate · 최근 7일 방문 13건
🍽  어떤 맛집을 찾으시나요? (지역 + 음식 종류를 알려주시면 좋아요. 예: "홍대 근처 디저트 카페")
```

LangGraph 기반 **ReAct 맛집 추천 에이전트**입니다. 사용자가 자연어로 맛집 추천을 요청하면, 에이전트가 Kakao · Naver · Google Places API를 병렬 호출하여 결과를 집계하고 최적의 추천을 생성합니다. 추천 결과는 Rich 콘솔, 로컬 trace 파일, Phoenix 분산 트레이싱 UI 세 곳에 동시 기록됩니다.

---

## 빠른 시작

Python **3.13** 권장. 둘 중 하나 골라서 실행하세요.

### 옵션 A — uv (권장, 빠름)

```bash
uv sync                                                # 의존성 설치
cp .env.example .env && $EDITOR .env                   # API 키 입력 (OPENAI + KAKAO 필수)
uv run python scripts/seed_memory.py                   # Memory 패턴 시연용 메모리 시드 (한 번만)
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

### 옵션 B — pip + venv

```bash
python3.13 -m venv .venv && source .venv/bin/activate  # 가상환경
pip install -r requirements.txt                        # 의존성 설치
cp .env.example .env && $EDITOR .env                   # API 키 입력
python scripts/seed_memory.py                          # 메모리 시드 (한 번만)
python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

### 대화형 모드 (선택)

```bash
uv run python chat.py        # 또는: python chat.py
```

ASCII 로고 + ❯ chevron 프롬프트 + 슬래시 커맨드 자동완성(`/help`, `/memory`, `/allow ...`) + ↑↓ 히스토리 + multi-turn clarification 지원. 자세한 사용법은 아래 [10. 대화형 모드](#10-대화형-모드-chatpy) 참고.

### 필수 API 키 (2개)

- `OPENAI_API_KEY` — platform.openai.com
- `KAKAO_REST_API_KEY` — developers.kakao.com (앱 → 앱 키 → REST API 키)

선택 키(`NAVER_*`, `GOOGLE_PLACES_API_KEY`, `PHOENIX_*`)는 없어도 동작합니다. 자세한 발급 안내는 [11. API 키 매트릭스](#11-api-키-매트릭스) 참고.

---

## 목차

1. [적용된 Agentic Design Pattern 5개](#1-적용된-agentic-design-pattern-5개)
2. [아키텍처 다이어그램](#2-아키텍처-다이어그램)
3. [도구 4개 명세](#3-도구-4개-명세)
4. [종합 전략 (aggregator)](#4-종합-전략-aggregator)
5. [메모리 모델](#5-메모리-모델)
6. [에러 처리](#6-에러-처리)
7. [Phoenix 통합](#7-phoenix-통합)
8. [디렉터리 구조](#8-디렉터리-구조)
9. [실행 흐름](#9-실행-흐름)
10. [대화형 모드 (chat.py)](#10-대화형-모드-chatpy)
11. [API 키 매트릭스](#11-api-키-매트릭스)
12. [시연 시나리오](#12-시연-시나리오)
13. [트레이스 산출물](#13-트레이스-산출물)
14. [테스트](#14-테스트)
15. [부록](#15-부록)


---

## 1. 적용된 Agentic Design Pattern 5개

본 프로젝트는 5개의 Agentic Design Pattern을 구현합니다.

| 패턴 | 위치 | 동작 |
|---|---|---|
| **ReAct** | `src/agent/nodes/react_agent.py` | LangGraph `create_react_agent`로 Thought→Action→Observation 자율 루프 (최대 8 turn). 에이전트가 어떤 도구를 어떤 인자로 호출할지 스스로 결정. |
| **Tool Use** | `src/tools/{geocode,kakao_local,naver_local,google_places}.py` | LangChain `@tool` 데코레이터 + OpenAI parallel function calling. 4개 도구를 ReAct 루프 안에서 병렬로 호출 가능. |
| **Plan-and-Solve** | `src/agent/nodes/planner.py` | react_agent 진입 전, LLM이 사용자 입력·메모리 컨텍스트를 분석해 JSON plan(지역, 반경, 음식 카테고리, 가중치 override 등)으로 분해. 사용자 비선호(dislikes)를 가드레일로 강제 머지. |
| **Memory** | `src/agent/nodes/{load,save}_memory.py` + `src/agent/nodes/preference_extractor.py` + `src/memory/store.py` | SQLite 영구 저장소. `user_profile` 테이블로 선호/비선호를 기억하고, `visit_history` 테이블로 방문 이력을 기록해 재방문 억제. **`preference_extractor`** 노드가 매 turn마다 user query를 LLM으로 분석해 새 비선호 추가·제거(`add_disliked`/`remove_disliked`) 및 방문 등록(`log_visits`)을 자동 추출하여 MemoryStore에 반영하고 state를 즉시 갱신. |
| **Reflection** | `src/agent/nodes/reflector.py` | 집계 결과를 LLM이 자가 검증(추천 수 충분한지, 품질 기준 충족했는지). 미충족 시 plan을 완화(반경 확대, 최저 평점 하향)하여 react_agent로 최대 2회 재귀. |

> 각 패턴이 어느 파일에서 실제로 동작하는지 위 표에 매핑되어 있습니다.

---

## 2. 아키텍처 다이어그램

```
START
  │
  ▼
load_memory ─────────────── SQLite에서 user_profile / visit_history 로드
  │                          (Plan-and-Solve · Memory 패턴 준비)
  ▼
preference_extractor ──────  LLM이 user query에서 새 선호/방문 추출, store에 반영, state 갱신
  │                          • add_disliked   : 비선호 카테고리 추가
  │                          • remove_disliked: 비선호 카테고리 제거
  │                          • log_visits     : 방문 이력 자동 등록
  ▼
planner ─────────────────── LLM이 JSON plan으로 분해 (Plan-and-Solve 패턴)
  │                          • 지역명, 반경, 카테고리, 인원, 예산
  │                          • 사용자 비선호 가드레일 강제 머지
  │                          • 가중치 override 결정
  │
  ├─ clarification_needed != [] ──────────────────────────────► finalizer (직행)
  │                                                              (비선호 충돌 안내)
  ▼
react_agent ◄──┐   LangGraph create_react_agent (ReAct 패턴)
  │             │   Thought → Action → Observation 루프 (max 8 turn)
  ▼             │
tool_node ──────┘   병렬 도구 호출 (Tool Use 패턴)
  │                  • geocode        : 지역명 → 좌표
  │                  • kakao_local    : Kakao Local API
  │                  • naver_local    : Naver Search API
  │                  • google_places  : Google Places API
  ▼
aggregator ───────── 결정론적 집계 (ReAct 루프 외부)
  │                   dedup → metadata merge → 가중치 점수 → 필터 → 정렬
  ▼
reflector ─────────── LLM 자가 검증 (Reflection 패턴)
  │                    passed=true/false 판단 + plan 실제 변화 검증 (no-op relaxation 차단)
  │
  ├─ passed=true ──────────────────────────────────────────────┐
  │                                                             │
  └─ passed=false (결과 부족) ─► react_agent (재탐색, plan 완화) │
       최대 2회 retry                                           │
       count ≥ 2 또는 retry 소진 시 ──────────────────────────►│
                                                               ▼
                                                          finalizer
                                                          │  normal | clarification | relaxed 3가지 mode
                                                          ▼
                                                       save_memory
                                                          │  visit_history 자동 append
                                                          ▼
                                                          END
```

**조건 분기 요약**

| 분기 조건 | 다음 노드 |
|---|---|
| planner: `clarification_needed` != [] | `finalizer` (직행, 검색 스킵) |
| reflector: `passed=true` | `finalizer` |
| reflector: `passed=false` AND retry < 2 | `react_agent` (plan 완화 후 재탐색) |
| reflector: `passed=false` AND retry ≥ 2 | `finalizer` (베스트 결과로 강제 종료) |

---

## 3. 도구 4개 명세

| 도구 이름 | 강점 | 핵심 파라미터 |
|---|---|---|
| **geocode** (`src/tools/geocode.py`) | 한국어 지역명·랜드마크를 위/경도로 변환. Kakao Geocoding 기반. 실패 시 `region_not_found` 에러 구조체 반환. | `query: str` (지역명 또는 주소) |
| **kakao_local** (`src/tools/kakao_local.py`) | 카카오맵 기반 주변 음식점 검색. 반경 지정, 카테고리 코드 지원. 국내 커버리지 최강. | `lat, lng: float`, `radius: int` (m), `category_group_code: str`, `query: str`, `size: int` |
| **naver_local** (`src/tools/naver_local.py`) | 네이버 블로그·리뷰 수 데이터 포함. 블로그 멘션 수를 리뷰 가중치 입력으로 활용. | `query: str`, `display: int`, `sort: str` (`random`\|`comment`) |
| **google_places** (`src/tools/google_places.py`) | 국제 평점(1–5)과 user_ratings_total 제공. 다국어 리뷰 집계. 세 API 중 평점 신뢰도 가장 높음. | `query: str`, `lat: float`, `lng: float`, `radius_m: int`, `included_type: str`, `price_levels: list[str] \| None`, `min_rating: float \| None`, `open_now: bool \| None`, `language_code: str` |

> 선택 키(`NAVER_*`, `GOOGLE_PLACES_API_KEY`) 누락 시 해당 도구는 자동 비활성됩니다. planner가 plan을 조정하고, 나머지 도구로 graceful degradation.

---

## 4. 종합 전략 (aggregator)

`src/agent/nodes/aggregator.py`는 react_agent가 수집한 이종(異種) API 결과를 **결정론적 5단계**로 처리합니다.

### 4-1. 5단계 파이프라인

1. **Dedup** — 동일 식당 중복 제거. 이름 유사도(Levenshtein 80 이상) + 좌표 거리(50m 이내) 두 조건 AND로 판정. 중복 시 더 풍부한 메타데이터를 가진 레코드 보존.
2. **Metadata merge** — 살아남은 레코드에 다른 소스의 필드를 병합. 예: Kakao에서 주소·전화번호, Google에서 평점·리뷰 수, Naver에서 블로그 멘션 수.
3. **가중치 점수 계산** — 아래 공식으로 0–1 스칼라 점수 산출.
4. **필터** — `plan.min_rating` 미만 평점 제거, `plan.exclude_visited` 목록(메모리) 제거.
5. **정렬** — 점수 내림차순 정렬 후 상위 `plan.k`개 반환.

### 4-2. 점수 공식

```
score = w_rating   · normalize(rating)
      + w_review   · normalize(log(review_count + 1))
      + w_distance · (1 / (distance_m + 1))   # 가까울수록 ↑
      + w_match    · (source_count / 3)        # 여러 API 동시 등장 → 신뢰 보너스
      - w_price    · price_level               # 가격대 (마이너스 가중)

#### 가격대(`price_level`) — 1~4 등급 ↔ KRW 매핑

외부 API에는 정확한 가격 정보가 없습니다 (Google Places가 1~4 추상 등급만 제공, Kakao·Naver는 가격 정보 없음). planner와 finalizer가 사용하는 매핑:

| price_level | Google 표기 | 대략 1인분 KRW | 예시 |
|---|---|---|---|
| 1 | `INEXPENSIVE`      | **1만원 이하**    | 분식, 국밥, 학식, 패스트푸드 |
| 2 | `MODERATE`         | **1–2만원**       | 일반 식당, 백반, 한식당 |
| 3 | `EXPENSIVE`        | **2–4만원**       | 고급 식당, 갈비집 |
| 4 | `VERY_EXPENSIVE`   | **4만원 이상**    | 파인다이닝, 호텔 |

- planner LLM이 사용자 KRW 표현(예: "만오천원 이하")을 위 등급으로 변환해 `post_filters.max_price_level`에 설정합니다.
- `aggregator`가 `price_level > max_price_level`인 후보를 점수 무관 **강제 제외** (해물·회 비선호 차단과 동일한 hard cutoff).
- `finalizer`가 추천 출력에 KRW 추정값을 함께 표기 (예: "💰 1–3만원").
- 후보의 `price_level`이 `null`인 경우(Kakao·Naver) 차단하지 않고 통과 — 정보 부재를 위반으로 간주하지 않음.
```

### 4-3. 가중치 기본값

| 가중치 키 | 기본값 | 의미 |
|---|---|---|
| `rating` | **0.35** | Google/Kakao 평점 (최대 기여) |
| `review` | **0.25** | 리뷰·블로그 멘션 수 (log 변환 후 정규화) |
| `distance` | **0.15** | 요청 지점으로부터의 거리 |
| `match` | **0.15** | 복수 API에서 동시 등장 시 신뢰 보너스 |
| `price` | **0.10** | 가격대 (높을수록 점수 감산) |

> **planner override**: planner가 사용자 의도에 따라 가중치를 override할 수 있습니다.
> 예시: `"리뷰 좋은 곳 위주로"` → planner가 `rating: 0.45, review: 0.30`으로 조정.
> `"가까운 곳 우선"` → `distance: 0.40`으로 조정.

---

## 5. 메모리 모델

`src/memory/store.py` — SQLite 영구 저장소 (`data/agent_memory.db`)

### 5-1. 두 테이블

**`user_profile`** — 사용자 장기 선호·비선호 저장

```sql
CREATE TABLE user_profile (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    category    TEXT,   -- 음식 카테고리 (예: "한식", "일식")
    sentiment   TEXT,   -- "like" | "dislike"
    value       TEXT,   -- 구체적 항목 (예: "매운 음식", "조용한 분위기")
    updated_at  TEXT    -- ISO 8601
);
```

**`visit_history`** — 방문 이력 시계열 기록

```sql
CREATE TABLE visit_history (
    id           INTEGER PRIMARY KEY,
    user_id      TEXT    NOT NULL,
    place_id     TEXT    NOT NULL,  -- Kakao/Google place ID
    place_name   TEXT,
    visited_at   TEXT,   -- ISO 8601
    rating_given REAL    -- 사용자가 준 평점 (선택)
);
```

### 5-2. 시드 스크립트

최초 실행 전 아래 명령으로 데모용 선호/비선호 및 방문 이력을 심습니다.

```bash
uv run python scripts/seed_memory.py
```

시드 내용 예시: "매운 음식 비선호", "조용한 분위기 선호", 최근 방문 3곳 기록.

### 5-3. 재방문 억제 로직 (`exclude_visited_within_days=1`)

```python
# load_memory.py 핵심 로직
cutoff = datetime.now() - timedelta(days=exclude_visited_within_days)  # 기본값 1
recently_visited = store.get_visited_after(user_id, cutoff)
plan["exclude_visited"] = [v.place_id for v in recently_visited]
```

**"어제 방문만 차단"** 정책: 기본값 `exclude_visited_within_days=1`이므로 어제 방문한 식당은 오늘 추천에서 제외됩니다. 오래전 방문 장소는 다시 추천될 수 있습니다. planner 또는 사용자 설정으로 이 값을 변경할 수 있습니다.

### 5-4. 자동 메모리 갱신 (preference_extractor)

`src/agent/nodes/preference_extractor.py` — `load_memory` 직후, `planner` 직전에 실행되는 LLM 기반 추출 노드입니다.

매 turn마다 user query를 LLM이 분석하여 세 가지 동작을 자동 수행합니다.

| 추출 패턴 | 동작 | 예시 표현 |
|---|---|---|
| `add_disliked` | 비선호 카테고리를 `user_profile`에 즉시 추가 | "나 회 싫어", "곱창은 별로야", "해산물 못 먹어" |
| `remove_disliked` | 기존 비선호 카테고리를 `user_profile`에서 제거 | "사실 매운 음식 괜찮아", "고수 이제 먹을 수 있어" |
| `log_visits` | 언급된 식당을 `visit_history`에 자동 등록 | "백송갈비 객사점 어제 갔어", "거기 지난주에 갔었는데" |

추출 결과는 MemoryStore에 즉시 반영된 뒤 `AgentState`(`dislikes`, `recent_visits`)도 갱신합니다. 따라서 같은 turn의 `planner`가 이미 최신 비선호·방문 정보를 받아서 plan을 수립하게 됩니다.

추출할 내용이 없는 일반 질의(예: "전주 맛집 추천해줘")에서는 LLM이 no-op으로 판단하고 state 변경 없이 통과합니다.

---

## 6. 에러 처리

| 케이스 | 처리 |
|---|---|
| **지역 못 찾음** (geocode 실패) | `{"error": "region_not_found", "suggestions": [...]}` 구조체 반환. react_agent가 LLM 판단으로 사용자에게 대체 지역명을 요청하거나 suggestions 중 하나로 재시도. |
| **API 호출 실패** (timeout / 5xx) | 해당 도구만 빈 결과 + 에러 기록. 다른 도구로 계속 진행. 활성 도구 3개가 모두 실패했을 때만 user-facing 에러로 전파. |
| **검색 결과 0건** | reflector가 `passed=false`로 감지 → `suggested_relaxation` (반경 +300m, min_rating −0.3)을 plan에 반영 후 react_agent 재시도 (최대 2회). |
| **모호한 요청** | planner JSON에 `"clarification_needed": true` + 질문 목록 포함 → react_agent가 LLM 판단으로 사용자에게 추가 정보 요청 메시지 생성. |
| **조건 부족** (반경·k 미지정 등) | planner가 합리적 기본값 보강: 반경 800m, k=3, 정렬 기준=리뷰순. 채워진 가정은 최종 응답에 명시 ("반경 800m로 탐색했습니다"). |

---

## 7. Phoenix 통합

`src/observability/telemetry.py`

Phoenix 분산 트레이싱으로 모든 LangGraph 노드 · LLM 호출 · 도구 실행이 span으로 자동 캡처됩니다.

- **endpoint**: `PHOENIX_COLLECTOR_ENDPOINT` 환경변수 (기본: `https://phoenix.rheon.kr`)
- **project_name**: `restaurant` (Phoenix UI에서 `https://phoenix.rheon.kr/projects/restaurant` 확인)

### 4가지 통합 패턴

| 패턴 | 설명 |
|---|---|
| **단일 idempotent init** | `init_telemetry()`는 프로세스 내 최초 1회만 실행. LangChain import 전에 호출해야 auto-instrumentation이 모든 span을 캡처함. 이미 초기화된 경우 noop. |
| **Kill switch** | `PHOENIX_DISABLED=1` 환경변수로 telemetry 전체 비활성화. 빠른 실행이나 오프라인 환경에서 유용. |
| **시작 로그** | 초기화 성공 시 `[phoenix] telemetry initialized → https://phoenix.rheon.kr/projects/restaurant` 한 줄 출력. Phoenix 연결 상태를 즉시 확인 가능. |
| **Force flush** | 단발 CLI 종료 직전 `flush_telemetry()` 호출. `BatchSpanProcessor`의 기본 5초 대기를 우회해 span 유실 방지. |

```bash
# Phoenix 없이 빠르게 실행 (트레이스 산출 없음)
PHOENIX_DISABLED=1 uv run python main.py "전주 객사 근처 맛집 추천"
```

### Phoenix UI trace 구조

self-hosted Phoenix UI(`https://phoenix.rheon.kr/projects/restaurant`)에서 실행 1회당 **LangGraph 전체 trace 1개**가 생성됩니다. 해당 trace를 펼치면 모든 노드(load_memory → preference_extractor → planner → react_agent → tool_node → aggregator → reflector → finalizer → save_memory), LLM 호출, 도구 호출이 **하나의 root span 아래 자식 span**으로 계층적으로 묶여 있습니다.

LangChain instrumentor가 LangGraph 내부 도구를 별도 root span으로도 export하는 부작용(도구별 1-span trace가 메인 trace와 분리되어 나타나는 현상)은 `src/observability/telemetry.py`의 `_install_root_filter`로 해결합니다. `SpanProcessor.on_end`를 instance-level monkey-patch하여 root span 이름이 "LangGraph"인 것만 export를 허용하고, 나머지 root span(도구 단독 trace)은 export 전에 drop합니다. 자식 span(도구·LLM·노드)은 모두 보존됩니다.

> Phoenix UI에서 trace를 클릭 → 전체 span 트리를 펼쳐서 9개 노드와 도구/LLM 호출이 모두 하나의 trace에 포함되어 있음을 확인할 수 있습니다.

---

## 8. 디렉터리 구조

```
restaurant-agent/
├── pyproject.toml              # 의존성 선언 (uv 관리)
├── uv.lock                     # 재현 가능한 잠금 파일
├── .python-version             # Python 버전 고정
├── .env                        # 실제 키 (gitignored)
├── .env.example                # 키 템플릿 (커밋됨)
├── .gitignore
├── README.md                   # 이 파일 (단일 진실 공급원)
├── main.py                     # CLI 진입점
├── scripts/
│   └── seed_memory.py          # 데모 메모리 시드 스크립트
├── src/
│   ├── observability/
│   │   └── telemetry.py        # Phoenix init / flush
│   ├── agent/
│   │   ├── graph.py            # LangGraph StateGraph 조립
│   │   ├── state.py            # AgentState 타입 정의
│   │   ├── types.py            # Restaurant / Plan 데이터 클래스
│   │   ├── prompts.py          # planner / reflector / finalizer 시스템 프롬프트
│   │   └── nodes/
│   │       ├── load_memory.py          # SQLite → AgentState 로드
│   │       ├── preference_extractor.py # LLM이 query에서 선호/방문 자동 추출 (Memory)
│   │       ├── planner.py              # LLM JSON plan 생성 (Plan-and-Solve)
│   │       ├── react_agent.py          # create_react_agent (ReAct)
│   │       ├── aggregator.py           # 결정론적 집계
│   │       ├── reflector.py            # LLM 자가 검증 (Reflection)
│   │       ├── finalizer.py            # 추천 결과 한국어 포맷 (normal/clarification/relaxed)
│   │       └── save_memory.py          # visit_history 저장
│   ├── tools/
│   │   ├── geocode.py          # 지역명 → 좌표
│   │   ├── kakao_local.py      # Kakao Local API
│   │   ├── naver_local.py      # Naver Search API
│   │   └── google_places.py    # Google Places API
│   ├── memory/
│   │   └── store.py            # SQLite CRUD (user_profile + visit_history)
│   └── ui/
│       └── renderer.py         # Rich 콘솔 출력 + trace.md 저장
├── data/
│   └── agent_memory.db         # gitignored, 시드 후 자동 생성
├── docs/
│   ├── superpowers/
│   │   ├── specs/              # 상세 설계 문서 (브레인스토밍 산출)
│   │   └── plans/              # 구현 계획 문서
│   └── traces/                 # 실행 결과 trace_<timestamp>.md
└── tests/                      # 60+ pytest 테스트
```

---

## 9. 실행 흐름

### 9-1. 단계별 명령

| 단계 | 명령 | 언제 필요한가 |
|---|---|---|
| **1. 의존성 설치** | `uv sync` | 최초 clone 직후 1회만 |
| **2. 환경변수 설정** | `cp .env.example .env` + `$EDITOR .env` | 최초 키 입력 시 1회만 |
| **3. 메모리 시드** | `uv run python scripts/seed_memory.py` | 최초 1회 (Memory 패턴 시연용). 이미 실행했으면 불필요. |
| **4. 에이전트 실행** | `uv run python main.py "질문"` | 추천이 필요할 때마다 |

> **매번 4단계를 다 실행할 필요가 없습니다.** 1–3단계는 환경 준비 단계이며, 이미 완료된 경우 4단계만 실행하면 됩니다.

### 9-2. 최초 설정 (클론 직후)

```bash
# 1) 의존성 설치
uv sync

# 2) 환경변수 입력
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY, KAKAO_REST_API_KEY 입력 (필수)
# Naver, Google, Phoenix 키는 선택사항

# 3) 데모 메모리 시드 (선호/비선호 및 방문 이력 데이터 생성)
uv run python scripts/seed_memory.py
```

### 9-3. 매번 실행

```bash
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

---

## 10. 대화형 모드 (chat.py)

한 세션 안에서 여러 번 추천을 받을 수 있는 REPL 모드도 제공합니다. **Memory 패턴이 살아 움직이는 것을 가장 직관적으로 보여줍니다** — 한 번 추천받으면 그 식당이 visit_history에 들어가고, 다음 추천에서 planner가 자동으로 제외합니다.

시작 시 ASCII 로고와 함께 현재 메모리 요약("어떤 맛집 찾으시나요?")이 출력됩니다.

### 실행

```bash
uv run python chat.py
```

### 슬래시 커맨드

| 명령 | 설명 |
|---|---|
| `/help`, `/?` | 도움말 출력 |
| `/memory` | 현재 user_profile + 최근 방문 기록 출력 |
| `/memory clear` | 전체 메모리 초기화 (user_profile + visit_history 삭제) |
| `/memory dislike add <category>` | 비선호 카테고리 직접 추가 (예: `/memory dislike add 회`) |
| `/memory dislike remove <category>` | 비선호 카테고리 직접 제거 |
| `/allow <category>` | 이번 세션에서만 해당 비선호 카테고리를 일시 해제 |
| `/allow clear` | 세션 허용 카테고리 목록 초기화 |
| `/clear` | 화면 클리어 |
| `/quit`, `/exit`, `/q` | 종료 (Ctrl-D / Ctrl-C 도 가능) |

### multi-turn clarification (yes/no 흐름)

planner가 메모리의 비선호 카테고리와 user query 사이의 충돌을 감지하면(예: "회" 비선호인데 "회 맛집 추천" 요청), `clarification_needed`를 설정하고 finalizer가 "추가 정보가 필요합니다" 메시지를 출력합니다. 다음 turn에서:

- 사용자가 `yes` / `응` / `ok` 등을 입력하면 → 충돌 카테고리가 자동으로 `/allow`에 등록되고 원래 query가 재실행됩니다.
- 사용자가 `no` / `아니` 등을 입력하면 → 충돌을 해소하고 새 query를 대기합니다.

```
> 전주 객사 회 맛집 추천해줘
[추가 정보 필요] 메모리에 '회'가 비선호로 등록되어 있습니다.
회 맛집을 포함해서 찾아드릴까요? (yes/no)

> yes
[이번 세션에서만 '회' 비선호 해제 → 원래 query 재실행]
[추천 3곳: ...]
```

### 예시 세션

```
> 전주 객사 근처에서 친구랑 저녁 먹기 좋은 곳
[추천 3곳: 백송갈비, 한식당, ...]
[trace] saved: docs/traces/trace_xxxx.md

> /memory
User Profile
  • disliked_categories: ['해물', '회']
Recent Visits (7일 이내)
  • 백송갈비 (한식) @ 2026-05-29T...
  • 한식당 (한식) @ 2026-05-29T...
  • ...

> 다른 한식집 또 추천해줘
[planner가 자동으로 위 식당들 제외 → 새로운 3곳]
```

각 turn은 `main.py`와 동일하게 자체 trace 파일을 생성합니다. main.py는 단발 실행용, chat.py는 시연·개발용으로 사용하세요.

---

## 11. API 키 매트릭스

| 키 | 필수 여부 | 환경변수 | 발급 위치 |
|---|---|---|---|
| OpenAI | **✅ 필수** | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| Kakao Local | **✅ 필수** | `KAKAO_REST_API_KEY` | [developers.kakao.com](https://developers.kakao.com) |
| Naver Search | ⭕ 선택 | `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | [developers.naver.com](https://developers.naver.com) |
| Google Places | ⭕ 선택 | `GOOGLE_PLACES_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) |
| Phoenix | ⭕ 선택 | `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` | self-hosted ([phoenix.rheon.kr](https://phoenix.rheon.kr)) |

**선택 키 누락 시 동작**: 해당 도구가 자동 비활성화되고 planner가 plan을 조정합니다 (graceful degradation). 필수 키 2개(OpenAI + Kakao)만 있으면 기본 동작이 보장됩니다.

---

## 12. 시연 시나리오

### 시나리오 1 — 메인 (대표 시나리오)

```bash
uv run python main.py "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
```

**예상 동작**:
- `load_memory`: 시드된 선호/비선호(예: 매운 음식 비선호) 로드
- `planner`: 지역=전주 객사, 반경=800m, k=3, 가격대=low/medium, 가중치 review↑로 override
- `react_agent`: geocode(전주 객사) → kakao_local + naver_local + google_places 병렬 호출
- `aggregator`: 결과 dedup·집계 → 상위 3곳 선정
- `reflector`: passed=true → `finalizer`로 이동
- 출력: Rich 콘솔에 추천 3곳 + 사유, trace 파일 생성, Phoenix span 전송

### 시나리오 2 — 존재하지 않는 지역

```bash
uv run python main.py "외계행성 뮤뮤성 맛집 추천해줘"
```

**예상 동작**:
- `geocode` 실패 → `{"error": "region_not_found", "suggestions": []}` 반환
- react_agent가 "요청하신 지역을 찾을 수 없습니다. 한국 내 지역명을 입력해 주세요." 메시지 생성
- 정상 종료 (에러 없이 사용자 친화적 안내)

### 시나리오 3 — 모호한 요청

```bash
uv run python main.py "맛집"
```

**예상 동작**:
- `planner` JSON에 `"clarification_needed": true` 포함
  - 필요 정보: 지역, 음식 종류, 인원 등
- react_agent가 "어느 지역의 맛집을 찾으시나요? 음식 종류도 알려주시면 더 정확한 추천이 가능합니다." 응답 생성

### 시나리오 4 — 메모리 비선호로 0건 → reflector 완화

```bash
uv run python main.py "전주 객사 회 맛집 찾아줘"
```

**예상 동작** (시드 메모리에 "회" 비선호 등록된 경우):
- `planner`: "회" 카테고리를 비선호로 강제 필터 적용
- `aggregator`: 필터 후 0건 → `reflector`: passed=false
- `reflector`: `suggested_relaxation` 생성 → 반경 확대, min_rating 하향, **비선호 완화 여부 재검토**
- 재탐색 후 결과 있으면 "고객님의 비선호 조건을 일부 완화하여 결과를 찾았습니다" 안내와 함께 추천

### 시나리오 5 — preference_extractor 자동 방문 등록

```bash
uv run python main.py "전주 객사 한식 추천해줘. 백송갈비 객사점은 어제 갔어"
```

**예상 동작**:
- `preference_extractor`: user query에서 `log_visits=["백송갈비 객사점"]` 자동 추출
  - 백송갈비 객사점이 `visit_history`에 즉시 등록됨 (logged_visits=1)
- 갱신된 state를 받은 `planner`: 백송갈비 객사점을 `exclude_visited` 목록에 포함
- `aggregator`: 백송갈비 객사점 필터 아웃
- 최종 추천 결과에서 백송갈비 객사점 제외 — "다음 추천에서 제외됩니다"와 함께 안내

### 시나리오 6 — Phoenix 비활성화 (빠른 실행)

```bash
PHOENIX_DISABLED=1 uv run python main.py "전주 객사 근처 맛집 추천해줘"
```

**예상 동작**:
- `[phoenix] telemetry disabled` 로그 출력
- Phoenix span 전송 없이 동일한 추천 로직 실행
- 네트워크 지연 없이 약간 더 빠른 실행
- 트레이스는 콘솔 + `docs/traces/trace_<ts>.md`에만 기록

---

## 13. 트레이스 산출물

실행 1회당 세 곳에 동시 기록됩니다.

| 산출물 | 위치 | 내용 |
|---|---|---|
| **Rich 콘솔** | 터미널 stdout | 컬러 패널로 추천 목록, 각 식당 이름·평점·거리·추천 사유, 노드 실행 흐름 요약 |
| **Trace 파일** | `docs/traces/trace_<timestamp>.md` | 마크다운 형식, 각 노드 입출력, 도구 호출 로그, 최종 추천 목록. 오프라인 검토용. |
| **Phoenix UI** | `https://phoenix.rheon.kr/projects/restaurant` | LangGraph 전체 그래프 span, LLM 토큰 사용량, 도구 응답 시간, 재시도 여부. `PHOENIX_DISABLED=1`이면 전송 안 됨. |

---

## 14. 테스트

```bash
uv run pytest -v
```

`tests/` 디렉터리에 60개 이상의 pytest 테스트가 포함되어 있습니다. 각 노드(load_memory, preference_extractor, planner, react_agent, aggregator, reflector, finalizer, save_memory), 도구(geocode, kakao_local, naver_local, google_places), 메모리 store, 에러 처리 케이스, chat.py 슬래시 커맨드 및 multi-turn clarification 흐름을 단위·통합 수준에서 검증합니다.

---

## 15. 부록

더 깊은 설계(브레인스토밍 과정, 상세 데이터 스키마, 노드별 프롬프트 설계 근거, 트레이드오프 논의)는 `docs/superpowers/specs/2026-05-28-restaurant-agent-design.md`를 참고하세요.
