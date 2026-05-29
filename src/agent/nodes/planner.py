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
    resp = llm.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    raw = resp.content
    try:
        plan_dict = json.loads(raw)
    except json.JSONDecodeError:
        # LLM 파싱 실패 → 최소 plan + clarification으로 graph가 finalizer로 직행
        import logging
        logging.getLogger(__name__).warning("planner: LLM 응답 JSON 파싱 실패. raw=%r", raw[:200])
        plan_dict = {
            "region_query": state.get("query", ""),
            "needs_geocoding": False,
            "clarification_needed": ["LLM 응답을 해석할 수 없습니다. 질문을 다시 입력해주세요."],
        }

    # clarification 경로에서 LLM이 다른 필드를 null로 줘도 Pydantic은 거부함.
    # 코드가 None을 기본값으로 채워서 Plan validation 통과시킴.
    _PLAN_DEFAULTS = {
        "food_keywords": [],
        "kakao": {},
        "naver": {},
        "google": {},
        "post_filters": {},
        "weights": {},
    }
    for k, default in _PLAN_DEFAULTS.items():
        if plan_dict.get(k) is None:
            plan_dict[k] = default
    plan_dict.setdefault("needs_geocoding", False)
    plan_dict.setdefault("region_query", state.get("query", ""))

    # ── 코드 가드레일: 비선호 카테고리 강제 머지 ──
    session_allowed = set(state.get("session_allowed") or [])
    user_dislikes = set(
        (state.get("user_profile") or {}).get("disliked_categories", []) or []
    ) - session_allowed
    pf = plan_dict.setdefault("post_filters", {})
    existing = set(pf.get("exclude_categories") or [])
    pf["exclude_categories"] = sorted(existing | user_dislikes)

    plan_obj = Plan.model_validate(plan_dict)

    # ── I3: 빈 query 기본값 채우기 ──
    # LLM이 query 필드를 생략하면 검색 도구가 빈 결과를 반환함.
    region = plan_obj.region_query or ""
    food = plan_obj.food_keywords[0] if plan_obj.food_keywords else "맛집"
    if not plan_obj.kakao.query:
        plan_dict["kakao"] = {**plan_dict.get("kakao", {}), "query": food}
    if not plan_obj.naver.query:
        plan_dict["naver"] = {**plan_dict.get("naver", {}),
                              "query": f"{region} {food}".strip()}
    if not plan_obj.google.query:
        plan_dict["google"] = {**plan_dict.get("google", {}),
                               "query": f"{food} {region}".strip()}

    # ── 코드 가드: region이 있으면 needs_geocoding=true 강제 ──
    # planner LLM이 "전주 객사처럼 익숙한 곳은 좌표 추론 가능"으로 잘못 판단하면
    # react_agent가 lat=null로 도구 호출 → Pydantic validation error → 자가 회복 사이클
    # → 토큰/시간 낭비. region이 있으면 무조건 geocode 먼저 부르도록 강제.
    if region.strip():
        plan_dict["needs_geocoding"] = True

    # ── 코드 가드: user query에 "N곳/N개/N군데" 명시 → post_filters.k 강제 ──
    # LLM이 깜빡해도 코드가 잡음. 식당 개수는 사용자 의도가 가장 명확.
    import re
    m = re.search(
        r"(\d+)\s*(?:곳|개|군데|가지|가게|식당|개의)",
        state.get("query", ""),
    )
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            plan_dict["post_filters"] = {
                **plan_dict.get("post_filters", {}), "k": n,
            }

    plan = Plan.model_validate(plan_dict)

    # 코드 가드: user query에 (세션 허용 제외) disliked 단어가 있으면 clarification 추가
    user_query = state.get("query", "").lower()
    all_disliked = (state.get("user_profile") or {}).get("disliked_categories", []) or []
    disliked_effective = [d for d in all_disliked if d not in session_allowed]
    conflicts = [d for d in disliked_effective if d.lower() in user_query]
    if conflicts and not plan.clarification_needed:
        conflict_msg = (
            f"프로필 비선호({', '.join(disliked_effective)})와 요청({', '.join(conflicts)})이 충돌합니다. "
            "이번에만 허용할지 다른 음식으로 바꿀지 알려주세요. "
            "(/allow <카테고리> 로 이번 세션만 허용 가능)"
        )
        plan_dict["clarification_needed"] = [conflict_msg]
        plan = Plan.model_validate(plan_dict)

    return {
        "plan": plan.model_dump(),
        "trace_log": state.get("trace_log", []) + [{
            "node": "planner",
            "plan": plan.model_dump(),
        }],
    }
