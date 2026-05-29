"""Seed the agent's memory with a demo user profile and visit history,
so the Memory pattern is observable from the very first run.

Idempotent — re-running replaces profile keys and skips duplicate visits."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from src.memory.store import MemoryStore

DB_PATH = Path("data/agent_memory.db")

PROFILE = {
    "disliked_categories": ["해물", "회"],
    "default_budget": "moderate",
    "notes": "친구와의 저녁은 적당한 가격대의 한식을 선호",
}

DEMO_VISITS = [
    {"name": "백송갈비 객사점", "category": "한식", "days_ago": 1},
    {"name": "전주 콩나물국밥 본점", "category": "한식", "days_ago": 3},
]


def main() -> None:
    store = MemoryStore(DB_PATH)

    for k, v in PROFILE.items():
        store.set_user_profile(k, v)

    existing_names = {v["name"] for v in store.get_recent_visits(within_days=30)}
    now = datetime.now()
    added = 0
    for v in DEMO_VISITS:
        if v["name"] in existing_names:
            continue
        store.append_visit(
            v["name"], v["category"],
            visited_at=now - timedelta(days=v["days_ago"]),
            source="seed",
        )
        added += 1

    print(f"[seed] profile keys: {list(store.all_user_profile().keys())}")
    print(f"[seed] visits added: {added}, total recent (7d): "
          f"{len(store.get_recent_visits(within_days=7))}")
    print(f"[seed] db: {DB_PATH}")


if __name__ == "__main__":
    main()
