import json
from unittest.mock import MagicMock

from src.agent.nodes.planner import planner_node
from src.agent.types import Plan


def _make_llm(content: str) -> MagicMock:
    """bind(response_format=...).invoke(...) 체인을 흉내내는 가짜 LLM."""
    fake_llm = MagicMock()
    fake_llm.bind.return_value = fake_llm
    fake_llm.invoke.return_value = MagicMock(content=content)
    return fake_llm


def test_planner_auto_clarification_on_dislike_conflict():
    """user가 명시한 음식이 disliked와 충돌하면 코드가 clarification 자동 추가."""
    fake_llm = _make_llm(json.dumps({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "food_keywords": ["회"],
        "kakao": {"query": "회", "category_group_code": "FD6", "radius_m": 800, "sort": "distance", "size": 15},
        "naver": {"query": "전주 객사 회 맛집", "sort": "comment"},
        "google": {"query": "회 맛집", "included_type": "restaurant",
                   "price_levels": None, "min_rating": None,
                   "open_now": None, "language_code": "ko"},
        "post_filters": {"exclude_categories": [], "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                    "match": 0.15, "price": 0.10},
        "clarification_needed": [],  # LLM이 빼먹음
    }))
    state = {
        "query": "전주 객사 회 맛집",
        "user_profile": {"disliked_categories": ["해물", "회"]},
        "recent_visits": [],
    }
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.clarification_needed  # 자동 추가됨
    assert any("회" in c for c in plan.clarification_needed)


def test_planner_returns_validated_plan():
    fake_llm = _make_llm(json.dumps({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "food_keywords": ["저녁"],
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance", "size": 15},
        "naver": {"sort": "comment"},
        "google": {
            "included_type": "restaurant",
            "price_levels": ["PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"],
            "min_rating": 4.0, "open_now": None, "language_code": "ko",
        },
        "post_filters": {"exclude_categories": ["해물"],
                         "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.45, "review": 0.30, "distance": 0.05,
                    "match": 0.10, "price": 0.10},
        "clarification_needed": [],
    }))

    state = {
        "query": "전주 객사 근처...",
        "user_profile": {"disliked_categories": ["해물"]},
        "recent_visits": [],
    }
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.region_query == "전주 객사"
    assert "해물" in plan.post_filters.exclude_categories


def test_planner_session_allowed_skips_conflict():
    """session_allowed에 있으면 disliked와 user query가 같아도 clarification 안 띄움."""
    fake_llm = _make_llm(json.dumps({
        "region_query": "전주 객사", "needs_geocoding": True, "food_keywords": ["회"],
        "kakao": {"query": "회", "category_group_code": "FD6", "radius_m": 800,
                  "sort": "distance", "size": 15},
        "naver": {"query": "전주 객사 회 맛집", "sort": "comment"},
        "google": {"query": "회 맛집", "included_type": "restaurant",
                   "price_levels": None, "min_rating": None,
                   "open_now": None, "language_code": "ko"},
        "post_filters": {"exclude_categories": [], "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                    "match": 0.15, "price": 0.10},
        "clarification_needed": [],
    }))
    state = {
        "query": "전주 객사 회 맛집",
        "user_profile": {"disliked_categories": ["해물", "회"]},
        "recent_visits": [],
        "session_allowed": ["회"],
    }
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.clarification_needed == []  # 세션 허용으로 충돌 없음
    assert "회" not in plan.post_filters.exclude_categories  # 회 카테고리 제외 안 됨


def test_planner_force_merges_user_dislikes_even_if_llm_forgets():
    """LLM이 깜빡해도 코드가 user_profile.disliked_categories를 머지."""
    fake_llm = _make_llm(json.dumps({
        "region_query": "전주",
        "needs_geocoding": True,
        "food_keywords": [],
        "kakao": {"category_group_code": "FD6", "radius_m": 800, "sort": "distance", "size": 15},
        "naver": {"sort": "comment"},
        "google": {"included_type": "restaurant", "price_levels": None,
                   "min_rating": None, "open_now": None, "language_code": "ko"},
        "post_filters": {"exclude_categories": [],  # LLM이 빼먹음
                         "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                    "match": 0.15, "price": 0.10},
        "clarification_needed": [],
    }))

    state = {"query": "x",
             "user_profile": {"disliked_categories": ["해물", "회"]},
             "recent_visits": []}
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert set(plan.post_filters.exclude_categories) >= {"해물", "회"}


def test_planner_fills_query_defaults_when_llm_omits():
    """I3 fix: LLM이 query 필드를 비워도 코드가 sensible 기본값을 채운다."""
    fake_llm = _make_llm(json.dumps({
        "region_query": "전주 객사",
        "needs_geocoding": True,
        "food_keywords": ["한식"],
        "kakao": {"query": "", "category_group_code": "FD6", "radius_m": 800,
                  "sort": "distance", "size": 15},
        "naver": {"query": "", "sort": "comment"},
        "google": {"query": "", "included_type": "restaurant",
                   "price_levels": None, "min_rating": None,
                   "open_now": None, "language_code": "ko"},
        "post_filters": {"exclude_categories": [], "exclude_visited_within_days": 1, "k": 3},
        "weights": {"rating": 0.35, "review": 0.25, "distance": 0.15,
                    "match": 0.15, "price": 0.10},
        "clarification_needed": [],
    }))
    state = {"query": "전주 객사 한식 추천",
             "user_profile": {}, "recent_visits": []}
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.kakao.query  # not empty
    assert plan.naver.query
    assert plan.google.query
    assert "전주 객사" in plan.naver.query  # region included


def test_planner_handles_malformed_json_gracefully():
    """C2 fix: LLM이 JSON이 아닌 응답을 주면 clarification fallback으로 처리."""
    fake_llm = _make_llm("not json at all")
    state = {"query": "test", "user_profile": {}, "recent_visits": []}
    out = planner_node(state, llm=fake_llm)
    plan = Plan.model_validate(out["plan"])
    assert plan.clarification_needed  # fallback에 clarification 들어가야 함
