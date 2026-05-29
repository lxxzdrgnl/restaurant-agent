from __future__ import annotations

from typing import Any

from src.memory.store import MemoryStore


def save_memory_node(state: dict[str, Any], *,
                     store: MemoryStore) -> dict[str, Any]:
    """final_recommendation을 visit_history에 append."""
    recs = state.get("final_recommendation") or []
    for r in recs:
        store.append_visit(
            name=r.get("name", ""),
            category=r.get("category"),
            source="recommended",
        )
    return {
        "trace_log": state.get("trace_log", []) + [{
            "node": "save_memory",
            "saved": len(recs),
        }],
    }
