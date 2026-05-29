from datetime import datetime, timedelta

from src.agent.nodes.aggregator import aggregator_node
from src.agent.types import Plan, Restaurant


def _r(**kw):
    base = dict(id=kw.pop("id", "x"), name=kw.pop("name", "X"),
                source=kw.pop("source", "kakao"))
    return Restaurant(**base, **kw).model_dump()


def test_dedup_merges_same_restaurant_from_multiple_sources():
    state = {
        "plan": Plan(region_query="전주 객사").model_dump(),
        "candidates": [
            _r(id="k_1", name="백송갈비 객사점", source="kakao",
               category="한식", lat=35.8186, lng=127.1489),
            _r(id="g_1", name="백송갈비", source="google",
               rating=4.4, review_count=200, price_level=2,
               lat=35.8186, lng=127.1489),
            _r(id="n_1", name="백송갈비 본점", source="naver",
               review_count=150, lat=35.8186, lng=127.1489),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    agg = out["aggregated"]
    assert len(agg) == 1
    r = agg[0]
    assert r["source_count"] == 3
    assert r["rating"] == 4.4
    assert r["category"] == "한식"
    # review_count = max(naver=150, google=200) = 200
    assert r["review_count"] == 200


def test_excludes_disliked_categories():
    state = {
        "plan": Plan(
            region_query="전주",
            post_filters={"exclude_categories": ["해물"], "k": 3,
                          "exclude_visited_within_days": 1},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="해물탕집", source="kakao", category="해물", rating=4.5),
            _r(id="k_2", name="한식당", source="kakao", category="한식", rating=4.2),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "해물탕집" not in names
    assert "한식당" in names


def test_excludes_recent_within_window():
    now = datetime.now()
    state = {
        "plan": Plan(
            region_query="전주",
            post_filters={"exclude_visited_within_days": 1, "k": 3,
                          "exclude_categories": []},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="어제먹은집", source="kakao", category="한식", rating=4.5),
            _r(id="k_2", name="새로운집", source="kakao", category="한식", rating=4.2),
        ],
        "recent_visits": [
            {"name": "어제먹은집", "category": "한식",
             "visited_at": (now - timedelta(hours=20)).isoformat(),
             "source": "seed"},
        ],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "어제먹은집" not in names
    assert "새로운집" in names


def test_aggregator_uses_plan_exclude_categories_not_user_profile():
    """C1 fix: aggregator trusts plan.post_filters.exclude_categories, ignores user_profile."""
    state = {
        "plan": Plan(
            region_query="x",
            post_filters={"exclude_categories": [],  # planner allowed all
                          "exclude_visited_within_days": 1, "k": 3},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="회집", source="kakao", category="회", rating=4.5),
            _r(id="k_2", name="한식집", source="kakao", category="한식", rating=4.0),
        ],
        "recent_visits": [],
        "user_profile": {"disliked_categories": ["회"]},  # still in profile
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "회집" in names  # NOT filtered (planner overrode)
    assert "한식집" in names


def test_aggregator_skips_recommended_source_in_recency():
    """I2 fix: 'recommended' source not treated as actually visited."""
    now = datetime.now()
    state = {
        "plan": Plan(
            region_query="x",
            post_filters={"k": 3, "exclude_categories": [], "exclude_visited_within_days": 7},
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="추천만받은집", source="kakao", category="한식", rating=4.5),
        ],
        "recent_visits": [
            {"name": "추천만받은집", "category": "한식",
             "visited_at": (now - timedelta(hours=1)).isoformat(),
             "source": "recommended"},  # 추천만 받았지 실제 안 갔음
        ],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "추천만받은집" in names  # NOT filtered


def test_top_k_ordered_by_score():
    state = {
        "plan": Plan(region_query="전주",
                     post_filters={"k": 2, "exclude_categories": [],
                                   "exclude_visited_within_days": 1}).model_dump(),
        "candidates": [
            _r(id="k_1", name="A", source="kakao", category="한식",
               rating=3.0, review_count=10, distance_m=100, price_level=2),
            _r(id="k_2", name="B", source="kakao", category="한식",
               rating=4.8, review_count=500, distance_m=200, price_level=2),
            _r(id="k_3", name="C", source="kakao", category="한식",
               rating=4.0, review_count=100, distance_m=150, price_level=2),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    assert len(out["aggregated"]) == 2
    assert out["aggregated"][0]["name"] == "B"  # 최고점


def test_max_price_level_hard_blocks_expensive(monkeypatch):
    """post_filters.max_price_level=2면 price_level=3,4 후보는 강제 제외."""
    state = {
        "plan": Plan(
            region_query="x",
            post_filters={
                "exclude_categories": [], "exclude_visited_within_days": 1,
                "max_price_level": 2, "k": 5,
            },
        ).model_dump(),
        "candidates": [
            _r(id="g_1", name="저렴한집", source="google",
               category="한식", rating=4.0, review_count=50, price_level=1),
            _r(id="g_2", name="적당한집", source="google",
               category="한식", rating=4.0, review_count=50, price_level=2),
            _r(id="g_3", name="비싼집", source="google",
               category="한식", rating=4.9, review_count=999, price_level=3),
            _r(id="g_4", name="고급집", source="google",
               category="한식", rating=5.0, review_count=999, price_level=4),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    names = [r["name"] for r in out["aggregated"]]
    assert "비싼집" not in names      # rating 좋아도 강제 차단
    assert "고급집" not in names
    assert "저렴한집" in names
    assert "적당한집" in names
    assert out["trace_log"][-1]["excluded_by_price"] == 2


def test_max_price_level_none_disables_price_filter():
    """max_price_level=None이면 price_level 무관하게 통과 (기존 동작)."""
    state = {
        "plan": Plan(
            region_query="x",
            post_filters={
                "exclude_categories": [], "exclude_visited_within_days": 1,
                "max_price_level": None, "k": 5,
            },
        ).model_dump(),
        "candidates": [
            _r(id="g_1", name="비싼집", source="google",
               category="한식", rating=4.5, review_count=999, price_level=4),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    assert "비싼집" in [r["name"] for r in out["aggregated"]]
    assert out["trace_log"][-1]["excluded_by_price"] == 0


def test_max_price_level_passes_candidates_with_no_price_info():
    """price_level=None인 후보(kakao/naver)는 정보 부재라 차단하지 않음."""
    state = {
        "plan": Plan(
            region_query="x",
            post_filters={
                "exclude_categories": [], "exclude_visited_within_days": 1,
                "max_price_level": 2, "k": 5,
            },
        ).model_dump(),
        "candidates": [
            _r(id="k_1", name="가격불명", source="kakao",
               category="한식", rating=4.0, price_level=None),
        ],
        "recent_visits": [],
        "user_profile": {},
    }
    out = aggregator_node(state)
    assert "가격불명" in [r["name"] for r in out["aggregated"]]
    assert out["trace_log"][-1]["excluded_by_price"] == 0
