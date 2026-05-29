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
    resp = llm.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content=REFLECTOR_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    try:
        eval_ = json.loads(resp.content)
    except json.JSONDecodeError:
        # LLM 파싱 실패 → 즉시 통과로 처리 (graph가 finalizer로)
        import logging
        logging.getLogger(__name__).warning(
            "reflector: LLM 응답 JSON 파싱 실패. raw=%r", resp.content[:200]
        )
        eval_ = {
            "passed": True,
            "reason": "(reflector 응답 파싱 실패 — 통과로 처리)",
            "suggested_relaxation": None,
        }

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
    relaxation = eval_.get("suggested_relaxation")
    plan_dict = plan.model_dump()
    if not eval_.get("passed") and new_count < MAX_REFLECTION and relaxation:
        new_plan = _apply_relaxation(plan_dict, relaxation)
        # 실제로 plan이 바뀌었는지 검증 — LLM이 dummy relaxation을 줘도 무력화
        if new_plan != plan_dict:
            out["plan"] = new_plan
            out["plan_relaxed"] = True
        else:
            out["plan_relaxed"] = False  # LLM 거짓 relaxation 차단
    else:
        out["plan_relaxed"] = False
    return out


def should_retry(state: dict[str, Any]) -> str:
    """conditional edge에서 사용. retry|finalize 분기.

    재시도 의미가 있으려면 reflector가 plan을 완화했어야 한다.
    plan이 그대로면 재실행해도 같은 결과 → 즉시 finalize.
    """
    if state.get("reflection_passed"):
        return "finalize"
    if int(state.get("reflection_count", 0)) >= MAX_REFLECTION:
        return "finalize"
    if not state.get("plan_relaxed"):
        return "finalize"
    return "retry"
