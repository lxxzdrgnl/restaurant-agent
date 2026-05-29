from pathlib import Path

from src.agent.nodes.load_memory import load_memory_node
from src.agent.nodes.save_memory import save_memory_node
from src.memory.store import MemoryStore


def test_load_memory_pulls_profile_and_recent_visits(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["해물"])
    store.append_visit("어제집", "한식")

    out = load_memory_node({"query": "x"}, store=store, recency_days=7)
    assert out["user_profile"]["disliked_categories"] == ["해물"]
    assert len(out["recent_visits"]) == 1


def test_save_memory_appends_recommendations(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    state = {
        "final_recommendation": [
            {"name": "A", "category": "한식"},
            {"name": "B", "category": "한식"},
        ],
    }
    out = save_memory_node(state, store=store)
    assert out["trace_log"][-1]["saved"] == 2
    visits = store.get_recent_visits(within_days=1)
    assert {v["name"] for v in visits} == {"A", "B"}
