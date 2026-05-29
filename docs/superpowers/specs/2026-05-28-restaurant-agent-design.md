# 맛집 추천 AI Agent — 설계 명세

**프로젝트**: 사용자 조건에 맞는 맛집을 찾아주는 Agentic AI (OSS Assign4)
**작성일**: 2026-05-28
**원본 스펙**: [`/spec.md`](../../../spec.md)

> 본 문서는 **결정사항과 원칙** 중심이다. 디테일(재시도 횟수, 인덱스, 프롬프트 문구 등)은 구현 단계에서 더 나은 방식을 자유롭게 선택한다.

---

## 1. 목표

사용자의 자연어 요청을 분석해서 도구(외부 API)를 직접 호출하고, 결과를 종합·검증해 **최종 추천 3곳**을 생성하는 ReAct 기반 에이전트. Phoenix span으로 각 패턴 동작이 트레이스에 또렷이 보여야 한다.

**채점용 시나리오** (스펙 명시):
> "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. 너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."

## 2. 결정사항 요약

| 항목 | 결정 |
|---|---|
| 언어/패키지 | Python + `uv` |
| 프레임워크 | LangGraph (SqliteSaver checkpointer) |
| LLM | OpenAI `gpt-4o-mini` |
| 데이터 소스 | Kakao Local + Naver Search Local + Google Places (New) |
| 적용 패턴 (5개) | ReAct (필수) + Tool Use + Plan-and-Solve + Memory + Reflection |
| 메모리 저장소 | SQLite (`data/agent_memory.db`) |
| CLI/UI | Rich (참고: `dexter-phoenix-pii-guard/src/cli.ts`) |
| 트레이스 | 콘솔(Rich) + `docs/traces/trace_<ts>.md` + Phoenix UI 3중 기록 |
| Phoenix | self-hosted `https://phoenix.rheon.kr` |

## 3. 아키텍처

```
START
  ↓
load_memory   ← SQLite에서 user_profile / visit_history 로드
  ↓
planner       ← LLM이 입력+메모리를 JSON plan으로 분해 (Plan-and-Solve)
  ↓
react_agent ↔ tool_node      ← LLM이 도구 자율 호출 (ReAct + Tool Use)
  ↓
aggregator    ← 결정론적 코드: dedup → metadata merge → 가중치 점수 → 필터
  ↓
reflector     ← LLM이 결과 자가검증, 부족 시 react_agent로 되돌아감 (max 2)
  ↓
finalizer     ← 상위 k=3 + 추천 사유 한국어 포맷
  ↓
save_memory   ← visit_history에 추천 결과 append
  ↓
END
```

**상태(State)**: `query, plan, user_profile, visit_history, candidates, aggregated, reflection_count, final_recommendation, messages`

## 4. 5개 패턴 매핑 (어디서 어떻게 동작하나)

| 패턴 | 위치 | 핵심 동작 | Phoenix span |
|---|---|---|---|
| **ReAct** | `react_agent` 노드 | Thought→Action→Observation 루프, max 8 turn | `node.react_agent` + 자식 `llm.openai`×N + `tool.*`×M |
| **Tool Use** | `tools/*` 4개 도구 | LangChain `@tool` + parallel function calling | `tool.{geocode,kakao,naver,google}` |
| **Plan-and-Solve** | `planner` 노드 | LLM이 `response_format=json_object`로 구조화 plan 출력 | `node.planner` + `plan` JSON attribute |
| **Memory** | `load_memory` / `save_memory` + SQLite | 영구 user_profile + 시계열 visit_history | `node.load_memory` (`user_profile`, `visit_history_count` attr) |
| **Reflection** | `reflector` 노드 | 체크리스트 자가평가 → 부족 시 plan 완화 후 재시도 | `node.reflector` (`passed`, `reflection_count` attr) |

채점자가 Phoenix UI에서 단일 trace 펼치면 5개 패턴이 모두 다른 span으로 보임.

## 5. 도구 명세 (4개)

각 도구는 LangChain `@tool` + Pydantic 스키마. 환경변수에서 API 키 누락 시 빈 결과 + 경고 (graceful degradation).

| 도구 | 강점 | 핵심 파라미터 |
|---|---|---|
| `geocode(region)` | 지역명 → 좌표 | Kakao 주소검색 사용 |
| `search_kakao_local` | 한국 카테고리/거리 정확 | `category_group_code` (FD6/CE7), `radius_m`, `sort` |
| `search_naver_local` | 블로그 리뷰 수 신호 | `sort=comment`, display=5 고정(API 제한) |
| `search_google_places` | 평점·가격대 강점 | `included_type`, `price_levels`, `min_rating`, `open_now`, `language_code=ko` |

`filter_by_criteria`는 도구가 아니라 `aggregator` 노드로 분리.

## 6. 종합 전략 (aggregator 노드)

`react_agent`가 모은 raw 후보를 결정론적으로 처리. LLM 호출 없음.

1. **Dedup**: `rapidfuzz.token_set_ratio(name) ≥ 85 AND haversine ≤ 50m`
2. **Metadata merge** (소스별 강점 우선):
   - `name/address/category` ← Kakao
   - `rating/price_level` ← Google
   - `review_count` ← `max(naver.blog, google.user_ratings_total)`
   - `source_count` ← 등장 API 수 (1~3, 신뢰 시그널)
3. **가중치 점수**:
   ```
   score = w_rating·rating + w_review·log1p(reviews) + w_distance·(1/dist)
         + w_match·(source_count/3) - w_price·price_level
   ```
   기본값 `{rating:0.35, review:0.25, distance:0.15, match:0.15, price:0.10}`.
   **planner가 사용자 의도에 따라 override** (예: "리뷰 좋은 곳" → rating/review 비중 ↑).
4. **필터**: 메모리 비선호 카테고리 / `exclude_visited_within_days` 윈도우 / 평점·가격 안전망
5. **정렬**: score 내림차순 → top `k=3`

## 7. 메모리 모델

```sql
user_profile (key TEXT PK, value JSON, updated_at TIMESTAMP)
  -- 예: ('disliked_categories', '["해물","회"]')

visit_history (id INTEGER PK, name TEXT, category TEXT, visited_at TIMESTAMP, source TEXT)
  -- "어제 먹은 것만" 제외가 기본 (exclude_visited_within_days=1)
  -- planner가 사용자 요청에 따라 윈도우 조정 ("최근 일주일 안 먹은 거" → 7)
```

- visit_history는 **기록**이지 자동 블랙리스트가 아니다. 윈도우 안만 일시 제외.
- 첫 실행부터 메모리 패턴이 시연되도록 `scripts/seed_memory.py` 제공.

## 8. 에러 처리 (원칙)

스펙 명시 5가지 케이스 모두 **Agent가 Observation으로 받고 대안 제시**.

| 케이스 | 처리 |
|---|---|
| geocode 실패 | tool이 `{error, suggestions}` 반환 → react_agent가 대체 지역명 요청 |
| API 호출 실패 | 해당 도구만 빈 결과 + Observation 기록, 나머지로 진행. 전부 실패 시에만 user-facing 에러 |
| 결과 0건 | reflector가 감지 → `suggested_relaxation`으로 반경/평점 완화 후 재시도 (max 2) |
| 모호한 요청 | planner JSON에 `clarification_needed` → react_agent가 추가 질문 |
| 조건 부족 | planner가 기본값 보강 + 적용 가정을 응답에 명시 |

## 9. Phoenix 통합

- **endpoint**: `https://phoenix.rheon.kr/api/collect`
- **인증**: `.env`의 `PHOENIX_API_KEY` → Bearer 헤더
- **프로젝트**: `restaurant-recommender-agent`
- **자동 instrumentation**: `phoenix.otel.register(auto_instrument=True)` + `openinference-instrumentation-langchain` → LangGraph 노드/도구/LLM 모두 자동 span

참고 프로젝트 `dexter-phoenix-pii-guard/src/observability/telemetry.ts`에서 4가지 패턴 차용:
1. **단일 idempotent `init_telemetry()`** — 최초 1회만, LangChain import 전 호출
2. **`PHOENIX_DISABLED=1` 킬 스위치**
3. **시작 로그 1줄** — endpoint/project/auth 상태 출력
4. **`flush_telemetry()`** — 단발 CLI 종료 직전 force flush (BatchSpanProcessor 5초 대기 우회)

## 10. CLI / 트레이스 출력

- **콘솔**: Rich 기반. 참고 프로젝트 `cli.ts`의 `renderEvent` / `summarizeToolResult` 패턴 차용. 도구 호출 트리 + 진행 스피너 + 완료 요약.
- **trace.md**: 노드/도구/입력/결과/duration을 markdown 표로. 채점 첨부용.
- **Phoenix UI**: 콘솔 마지막 줄에 trace URL 출력 → 클릭으로 span tree 확인.

## 11. 디렉터리 구조

```
restaurant-agent/
├── pyproject.toml, uv.lock
├── .env, .env.example, .gitignore
├── README.md
├── main.py                            # init_telemetry → run → flush
├── scripts/
│   └── seed_memory.py
├── src/
│   ├── observability/telemetry.py     # Phoenix init/flush
│   ├── agent/
│   │   ├── graph.py                   # StateGraph 정의
│   │   ├── state.py                   # AgentState TypedDict
│   │   ├── nodes/{load_memory,planner,react_agent,aggregator,
│   │   │           reflector,finalizer,save_memory}.py
│   │   └── prompts.py
│   ├── tools/{geocode,kakao_local,naver_local,google_places}.py
│   ├── memory/store.py
│   └── ui/renderer.py                 # Rich 트레이스 출력
├── data/agent_memory.db               # gitignored
└── docs/traces/                       # 실행 결과
```

## 12. 의존성 (초안)

```toml
dependencies = [
  "langgraph>=0.2",
  "langchain-openai>=0.2",
  "langgraph-checkpoint-sqlite>=2.0",
  "openai>=1.50",
  "arize-phoenix-otel",
  "openinference-instrumentation-langchain",
  "httpx>=0.27",
  "pydantic>=2.7",
  "python-dotenv>=1.0",
  "rapidfuzz>=3.9",
  "rich>=13.7",
]
```

## 13. 채점자 친화 부록

### 필수 환경변수 (2개만)
- `OPENAI_API_KEY` — LLM 호출
- `KAKAO_REST_API_KEY` — geocode + 메인 검색

### 선택 (있으면 풀이 풍부해짐)
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` — 블로그 리뷰 시그널
- `GOOGLE_PLACES_API_KEY` — 평점·가격대 필터
- `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` — 트레이스 전송 (없으면 자동 비활성)

### 실행 4단계
```bash
uv sync                                          # 1. 의존성
cp .env.example .env && $EDITOR .env             # 2. 키 입력
uv run python scripts/seed_memory.py             # 3. 메모리 시드 (Memory 패턴 시연용)
uv run python main.py "전주 객사 근처에서..."     # 4. 실행
```

### 시연 시나리오 (README에 예시)
- 메인: 스펙 명시 프롬프트
- region_not_found: `"외계행성 뮤뮤성 맛집"`
- 모호 입력: `"맛집"` → clarification
- 메모리 충돌: `"전주 객사 회 맛집"` → 비선호 카테고리로 0건 → reflector relaxation
- API 부재: 일부 키 누락 상태로 실행 → graceful degradation

---

## 자기검토 체크리스트 (구현 시작 전)

- [ ] 환경변수 4종 발급 링크가 README에 있는가
- [ ] `.env`가 `.gitignore`에 있는가 (Phoenix 키 포함)
- [ ] `init_telemetry()`가 LangChain import 전에 호출되는가
- [ ] react_agent의 max_turns(8) + reflector의 max retry(2)가 코드 상수로 분리되어 있는가
- [ ] 가중치(`DEFAULT_WEIGHTS`)가 한 곳에 정의되고 planner가 override 가능한가
- [ ] visit_history의 윈도우 기본값(1일)이 planner 파라미터로 노출되는가
- [ ] 시드 스크립트가 idempotent한가 (재실행해도 중복 안 쌓임)
