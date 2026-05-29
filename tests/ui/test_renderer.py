from src.ui.renderer import render_trace_md


def test_render_trace_md_includes_each_node():
    trace = [
        {"node": "load_memory", "profile_keys": ["disliked_categories"], "recent_count": 1},
        {"node": "planner", "plan": {"region_query": "전주 객사"}},
        {"node": "react_agent", "messages_count": 8, "candidates_count": 18},
        {"node": "aggregator", "raw_count": 18, "merged_count": 12,
         "excluded_by_category": 2, "excluded_by_recency": 1, "kept": 3},
        {"node": "reflector", "passed": True, "reason": "ok",
         "reflection_count": 1},
        {"node": "finalizer", "k": 3},
        {"node": "save_memory", "saved": 3},
    ]
    md = render_trace_md(
        query="전주 객사 근처...",
        final_text="1. A\n2. B\n3. C",
        trace_log=trace,
    )
    for node in ("load_memory", "planner", "react_agent", "aggregator",
                 "reflector", "finalizer", "save_memory"):
        assert node in md
    assert "전주 객사" in md
    assert "## Final" in md
