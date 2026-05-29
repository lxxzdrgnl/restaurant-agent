from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.agent.types import Plan, Restaurant


class AgentState(TypedDict, total=False):
    # input
    query: str
    session_allowed: list[str]

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
