"""Preference extractor — user query에서 새 메모리 정보를 추출해 store에 반영."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.prompts import EXTRACTOR_SYSTEM
from src.memory.store import MemoryStore


def preference_extractor_node(
    state: dict[str, Any],
    *,
    llm,
    store: MemoryStore,
) -> dict[str, Any]:
    """user query를 LLM이 분석해 새 선호/방문을 store에 자동 반영.

    이 노드 다음의 planner가 갱신된 memory로 plan을 작성한다."""
    query = state.get("query", "")
    profile = state.get("user_profile") or {}
    visits = state.get("recent_visits") or []

    if not query:
        return _no_op(state)

    user_msg = (
        f"이번 turn의 user query: {query}\n"
        f"현재 user_profile: {json.dumps(profile, ensure_ascii=False)}\n"
        f"최근 7일 visit 이름: "
        f"{json.dumps([v.get('name') for v in visits], ensure_ascii=False)}"
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=EXTRACTOR_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        extracted = json.loads(resp.content)
    except Exception:  # noqa: BLE001
        # 추출 실패 시 silent — 그래프 진행 막지 않음
        return _no_op(state, reason="parse_error")

    add_disliked = list(extracted.get("add_disliked") or [])
    remove_disliked = list(extracted.get("remove_disliked") or [])
    log_visits = list(extracted.get("log_visits") or [])

    # 1) disliked_categories 갱신
    current = profile.get("disliked_categories") or []
    updated = sorted(
        (set(current) | set(add_disliked)) - set(remove_disliked)
    )
    if updated != sorted(current):
        store.set_user_profile("disliked_categories", updated)

    # 2) visit_history append
    appended = 0
    now = datetime.now()
    for v in log_visits:
        name = (v.get("name") or "").strip()
        if not name:
            continue
        category = v.get("category")
        try:
            days_ago = max(0, int(v.get("days_ago", 1)))
        except (TypeError, ValueError):
            days_ago = 1
        store.append_visit(
            name=name,
            category=category,
            visited_at=now - timedelta(days=days_ago),
            source="user_logged",
        )
        appended += 1

    # 3) state refresh (planner가 갱신된 메모리로 plan 작성)
    new_profile = store.all_user_profile()
    new_visits = store.get_recent_visits(within_days=7)

    return {
        "user_profile": new_profile,
        "recent_visits": new_visits,
        "trace_log": state.get("trace_log", []) + [{
            "node": "preference_extractor",
            "add_disliked": add_disliked,
            "remove_disliked": remove_disliked,
            "logged_visits": appended,
        }],
    }


def _no_op(state: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """추출할 게 없거나 실패 시: 메모리는 그대로 통과시킨다."""
    log_entry = {"node": "preference_extractor",
                 "add_disliked": [], "remove_disliked": [], "logged_visits": 0}
    if reason:
        log_entry["status"] = reason
    return {
        "user_profile": state.get("user_profile") or {},
        "recent_visits": state.get("recent_visits") or [],
        "trace_log": state.get("trace_log", []) + [log_entry],
    }
