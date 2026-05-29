from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import FINALIZER_SYSTEM


def finalizer_node(state: dict[str, Any], *, llm) -> dict[str, Any]:
    aggregated = state.get("aggregated", [])
    clarification = state.get("plan", {}).get("clarification_needed") or []
    reflection_passed = state.get("reflection_passed", True)
    reflection_reason = state.get("reflection_reason", "")

    relaxation_hint = ""
    if not reflection_passed:
        relaxation_hint = f"\nNOTE: 조건이 완화되었습니다 (사유: {reflection_reason}). 추천 앞에 안내 박스 한 줄 추가하라."

    user_msg = (
        f"사용자 원본 요청: {state.get('query', '')}\n"
        f"추천 후보 (top {len(aggregated)}):\n"
        f"{json.dumps(aggregated, ensure_ascii=False, indent=2)}\n"
        f"\n[meta]\n"
        f"clarification_needed: {json.dumps(clarification, ensure_ascii=False)}\n"
        f"reflection_passed: {reflection_passed}\n"
        f"{relaxation_hint}"
    )
    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    final_recs = [] if clarification else aggregated
    return {
        "final_recommendation": final_recs,
        "final_text": resp.content,
        "trace_log": state.get("trace_log", []) + [{
            "node": "finalizer",
            "k": len(final_recs),
            "mode": "clarification" if clarification else ("relaxed" if not reflection_passed else "normal"),
        }],
    }
