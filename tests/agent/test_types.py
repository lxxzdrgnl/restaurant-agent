from src.agent.types import Restaurant, Plan, AggregationWeights, DEFAULT_WEIGHTS


def test_restaurant_minimum_fields():
    r = Restaurant(id="k_1", name="가게", source="kakao")
    assert r.id == "k_1"
    assert r.rating is None
    assert r.source_count == 1


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.model_dump().values()) - 1.0) < 1e-9


def test_plan_parses_json():
    plan = Plan.model_validate({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance"},
        "naver": {"sort": "comment"},
        "google": {
            "included_type": "restaurant",
            "price_levels": ["PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"],
            "min_rating": 4.0,
        },
        "post_filters": {
            "exclude_categories": ["해물", "회"],
            "exclude_visited_within_days": 1,
            "k": 3,
        },
        "weights": {"rating": 0.45, "review": 0.30, "distance": 0.05,
                    "match": 0.10, "price": 0.10},
    })
    assert plan.weights.rating == 0.45
    assert plan.post_filters.k == 3
