import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.nodes.preference_extractor import preference_extractor_node
from src.memory.store import MemoryStore


def _fake_llm(content_dict):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(content_dict))
    return llm


def test_extracts_new_dislike_and_writes_to_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["회"])
    llm = _fake_llm({
        "add_disliked": ["해물"],
        "remove_disliked": [],
        "log_visits": [],
    })
    state = {
        "query": "전주 객사 한식 추천. 해물 싫어",
        "user_profile": {"disliked_categories": ["회"]},
        "recent_visits": [],
    }
    out = preference_extractor_node(state, llm=llm, store=store)
    assert sorted(out["user_profile"]["disliked_categories"]) == ["해물", "회"]
    assert store.get_user_profile("disliked_categories") == ["해물", "회"]


def test_removes_dislike(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["해물", "회"])
    llm = _fake_llm({
        "add_disliked": [],
        "remove_disliked": ["해물"],
        "log_visits": [],
    })
    state = {
        "query": "이제 해물 좀 먹어볼래",
        "user_profile": {"disliked_categories": ["해물", "회"]},
        "recent_visits": [],
    }
    out = preference_extractor_node(state, llm=llm, store=store)
    assert out["user_profile"]["disliked_categories"] == ["회"]


def test_logs_visit(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    llm = _fake_llm({
        "add_disliked": [],
        "remove_disliked": [],
        "log_visits": [
            {"name": "백송갈비", "category": "한식", "days_ago": 1}
        ],
    })
    state = {
        "query": "전주 객사 한식 추천해줘. 백송갈비는 어제 갔어",
        "user_profile": {},
        "recent_visits": [],
    }
    out = preference_extractor_node(state, llm=llm, store=store)
    assert len(out["recent_visits"]) == 1
    assert out["recent_visits"][0]["name"] == "백송갈비"
    assert out["trace_log"][-1]["logged_visits"] == 1


def test_no_op_when_nothing_to_extract(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    llm = _fake_llm({
        "add_disliked": [],
        "remove_disliked": [],
        "log_visits": [],
    })
    state = {
        "query": "한식 추천해줘",
        "user_profile": {},
        "recent_visits": [],
    }
    out = preference_extractor_node(state, llm=llm, store=store)
    assert out["user_profile"] == {}
    assert out["recent_visits"] == []
    assert out["trace_log"][-1]["logged_visits"] == 0


def test_llm_parse_error_is_silent(tmp_path: Path):
    """LLM이 JSON 안 주면 메모리 변경 없이 통과."""
    store = MemoryStore(tmp_path / "m.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="not json at all")
    state = {
        "query": "x",
        "user_profile": {"disliked_categories": ["회"]},
        "recent_visits": [],
    }
    out = preference_extractor_node(state, llm=llm, store=store)
    assert out["user_profile"] == {"disliked_categories": ["회"]}
    assert out["trace_log"][-1].get("status") == "parse_error"
