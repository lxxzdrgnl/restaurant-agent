from __future__ import annotations

from typing import Any

from src.memory.store import MemoryStore


def save_memory_node(state: dict[str, Any], *,
                     store: MemoryStore) -> dict[str, Any]:
    """추천 결과는 실제 방문이 아니므로 visit_history에 자동 등록하지 않는다.
    visit는 사용자가 명시적으로 '갔어'라고 한 발화만 preference_extractor가 등록한다.
    이 노드는 그래프 구조 보존을 위해 남겨두고 trace 로깅만 수행."""
    return {
        "trace_log": state.get("trace_log", []) + [{
            "node": "save_memory",
            "saved": 0,
            "note": "auto-save disabled — visits only via preference_extractor user-logged",
        }],
    }
