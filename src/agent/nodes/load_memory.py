from __future__ import annotations

from typing import Any

from src.memory.store import MemoryStore


def load_memory_node(state: dict[str, Any], *,
                     store: MemoryStore,
                     recency_days: int = 7) -> dict[str, Any]:
    """user_profile + recent_visits (지난 N일)을 state에 주입."""
    profile = store.all_user_profile()
    recent = store.get_recent_visits(within_days=recency_days)
    return {
        "user_profile": profile,
        "recent_visits": recent,
        "trace_log": state.get("trace_log", []) + [{
            "node": "load_memory",
            "profile_keys": list(profile.keys()),
            "recent_count": len(recent),
        }],
    }
