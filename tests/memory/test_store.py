import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


def test_user_profile_set_get(store):
    store.set_user_profile("disliked_categories", ["해물", "회"])
    assert store.get_user_profile("disliked_categories") == ["해물", "회"]


def test_user_profile_missing_returns_default(store):
    assert store.get_user_profile("nope", default="x") == "x"


def test_visit_history_append_and_recent(store):
    now = datetime(2026, 5, 28, 19, 0, 0)
    store.append_visit("백송갈비", "한식", visited_at=now - timedelta(days=1))
    store.append_visit("스시오마카세", "일식", visited_at=now - timedelta(days=3))
    recent = store.get_recent_visits(within_days=1, now=now)
    assert [v["name"] for v in recent] == ["백송갈비"]


def test_visit_history_within_7_days(store):
    now = datetime(2026, 5, 28, 19, 0, 0)
    store.append_visit("백송갈비", "한식", visited_at=now - timedelta(days=1))
    store.append_visit("스시오마카세", "일식", visited_at=now - timedelta(days=3))
    recent = store.get_recent_visits(within_days=7, now=now)
    assert {v["name"] for v in recent} == {"백송갈비", "스시오마카세"}


def test_all_profile_keys(store):
    store.set_user_profile("a", 1)
    store.set_user_profile("b", "two")
    assert store.all_user_profile() == {"a": 1, "b": "two"}


def test_remove_user_profile(store):
    store.set_user_profile("a", 1)
    store.set_user_profile("b", 2)
    store.remove_user_profile("a")
    assert store.get_user_profile("a") is None
    assert store.get_user_profile("b") == 2


def test_clear_all(store):
    store.set_user_profile("a", 1)
    store.append_visit("x", "한식")
    store.clear_all()
    assert store.all_user_profile() == {}
    assert store.get_recent_visits(within_days=30) == []
