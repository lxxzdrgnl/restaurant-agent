import json
from unittest.mock import MagicMock

from src.agent.nodes.reflector import reflector_node


def _make_llm(content: str) -> MagicMock:
    """bind(response_format=...).invoke(...) 체인을 흉내내는 가짜 LLM."""
    fake_llm = MagicMock()
    fake_llm.bind.return_value = fake_llm
    fake_llm.invoke.return_value = MagicMock(content=content)
    return fake_llm


def test_fail_with_noop_relaxation_does_not_retry():
    """LLM이 dummy relaxation을 줘도 실제 plan 변화 없으면 plan_relaxed=False."""
    fake_llm = _make_llm(json.dumps({
        "passed": False, "reason": "충돌",
        "suggested_relaxation": {
            "kakao_radius_delta_m": 0,           # delta=0
            "google_min_rating_delta": None,
            "google_drop_price_filter": False,
            "post_filters_exclude_categories_remove": [],
        },
    }))
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1},
                 "kakao": {"query": "", "radius_m": 800, "category_group_code": "FD6",
                           "sort": "distance", "size": 15},
                 "naver": {"query": "", "sort": "comment"},
                 "google": {"query": "", "included_type": "restaurant",
                            "price_levels": None, "min_rating": 4.0,
                            "open_now": None, "language_code": "ko"},
                 "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                             "match": 0.15, "price": 0.10},
                 "needs_geocoding": True, "food_keywords": [],
                 "clarification_needed": []},
        "aggregated": [],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["plan_relaxed"] is False  # 거짓 relaxation 차단


def test_passes_when_enough_candidates():
    fake_llm = _make_llm(json.dumps({
        "passed": True, "reason": "조건 충족", "suggested_relaxation": None,
    }))
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1}},
        "aggregated": [{"name": f"r{i}"} for i in range(3)],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["reflection_passed"] is True
    assert out["reflection_count"] == 1


def test_fail_returns_relaxation_and_bumps_count():
    fake_llm = _make_llm(json.dumps({
        "passed": False, "reason": "후보 부족",
        "suggested_relaxation": {
            "kakao_radius_delta_m": 400,
            "google_min_rating_delta": -0.3,
            "google_drop_price_filter": False,
            "post_filters_exclude_categories_remove": [],
        },
    }))
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1},
                 "kakao": {"radius_m": 800, "category_group_code": "FD6",
                           "sort": "distance", "size": 15},
                 "naver": {"sort": "comment"},
                 "google": {"included_type": "restaurant",
                            "price_levels": None, "min_rating": 4.0,
                            "open_now": None, "language_code": "ko"},
                 "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                             "match": 0.15, "price": 0.10},
                 "needs_geocoding": True, "food_keywords": [],
                 "clarification_needed": []},
        "aggregated": [],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["reflection_passed"] is False
    assert out["reflection_count"] == 1
    # plan was relaxed
    assert out["plan"]["kakao"]["radius_m"] == 1200
    assert abs(out["plan"]["google"]["min_rating"] - 3.7) < 1e-9


def test_reflector_handles_malformed_json_passes_through():
    """C2 fix: reflector가 JSON 파싱 실패하면 passed=True로 안전하게 통과."""
    fake_llm = _make_llm("not json")
    state = {
        "plan": {"region_query": "x",
                 "post_filters": {"k": 3, "exclude_categories": [],
                                  "exclude_visited_within_days": 1}},
        "aggregated": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "reflection_count": 0,
    }
    out = reflector_node(state, llm=fake_llm)
    assert out["reflection_passed"] is True  # 안전하게 통과
