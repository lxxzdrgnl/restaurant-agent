from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes.aggregator import aggregator_node
from src.agent.nodes.finalizer import finalizer_node
from src.agent.nodes.load_memory import load_memory_node
from src.agent.nodes.planner import planner_node
from src.agent.nodes.preference_extractor import preference_extractor_node
from src.agent.nodes.react_agent import make_react_agent
from src.agent.nodes.reflector import reflector_node, should_retry
from src.agent.nodes.save_memory import save_memory_node
from src.agent.state import AgentState
from src.memory.store import MemoryStore
from src.tools.geocode import geocode
from src.tools.google_places import search_google_places
from src.tools.kakao_local import search_kakao_local
from src.tools.naver_local import search_naver_local


def _needs_clarification(state: dict[str, Any]) -> str:
    """planner가 clarification을 요구하면 검색 건너뛰고 finalizer로 직행."""
    plan = state.get("plan") or {}
    return "clarify" if plan.get("clarification_needed") else "proceed"


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
    g.add_node("preference_extractor",
               lambda s: preference_extractor_node(s, llm=llm, store=store))
    g.add_node("planner", lambda s: planner_node(s, llm=llm))
    g.add_node("react_agent", react_node)
    g.add_node("aggregator", aggregator_node)
    g.add_node("reflector", lambda s: reflector_node(s, llm=llm))
    g.add_node("finalizer", lambda s: finalizer_node(s, llm=llm))
    g.add_node("save_memory", lambda s: save_memory_node(s, store=store))

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "preference_extractor")
    g.add_edge("preference_extractor", "planner")
    g.add_conditional_edges("planner", _needs_clarification, {
        "clarify": "finalizer",
        "proceed": "react_agent",
    })
    g.add_edge("react_agent", "aggregator")
    g.add_edge("aggregator", "reflector")
    g.add_conditional_edges("reflector", should_retry, {
        "retry": "react_agent",
        "finalize": "finalizer",
    })
    g.add_edge("finalizer", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
