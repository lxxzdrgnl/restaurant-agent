"""React loop: LLM이 도구를 자율 호출. LangGraph의 create_react_agent를 활용.

검색 도구만 노출 (geocode/kakao/naver/google). 결과는 messages에 누적되고,
이후 코드가 ToolMessage를 파싱해서 candidates로 평탄화한다."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import create_react_agent

from src.agent.types import Plan

REACT_SYSTEM = """\
당신은 한국 맛집 검색 에이전트다. 사용자의 plan(JSON)을 받아 도구를 호출해서
후보 식당 풀을 모은다.

핵심 규칙 — 효율을 위해 cycle 수를 최소화하라:
1. plan.needs_geocoding == true 면 첫 turn에서 geocode를 정확히 1번 호출.
   geocode 결과를 받은 다음 turn에서는 절대 재호출하지 말 것.
2. 좌표를 받은 직후 turn에서 search_kakao_local + search_google_places + search_naver_local을
   **반드시 한 번에 병렬 호출**하라 (3개 tool_call을 같은 응답에 묶어서 발행).
   순차 호출이나 분할 호출 금지.
3. search_naver_local은 좌표 없이 query만 받는다.
4. 도구 호출은 plan의 파라미터를 **그대로** 사용한다. 특히 다음 query 필드는 **planner가 이미 결정**한 값이므로 절대 수정 금지:
   - search_kakao_local(query=plan.kakao.query, ...)
   - search_naver_local(query=plan.naver.query, ...)
   - search_google_places(query=plan.google.query, ...)
5. **이미 같은 인자로 호출한 도구는 절대 재호출 금지.** 결과는 이미 messages에 있다.
6. 검색 도구 3개의 결과를 받으면 즉시 도구 호출 없이 'DONE'이라고만 답하라.
   결과가 적다고 추가 검색 시도하지 말 것 — 그 판단은 reflector가 한다.

이상적 흐름 (2 turn):
  turn 1: geocode(region)
  turn 2: search_kakao_local + search_naver_local + search_google_places (병렬)
  turn 3: 'DONE'
"""


def make_react_agent(llm, tools: list, recursion_limit: int = 8) -> Callable:
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
