from unittest.mock import MagicMock

from src.agent.nodes.finalizer import finalizer_node


def test_finalizer_clarification_mode_skips_recommendations():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="⚠️ 추가 정보가 필요합니다\n\n- 회 vs 비선호 충돌")
    state = {
        "query": "전주 객사 회 맛집",
        "aggregated": [],
        "plan": {"clarification_needed": ["회 비선호와 충돌"]},
        "reflection_passed": True,
    }
    out = finalizer_node(state, llm=fake_llm)
    assert out["final_recommendation"] == []  # 추천 없음
    assert "추가 정보" in out["final_text"]


def test_finalizer_relaxed_mode_when_reflection_failed():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="> ⚠️ 완화된 추천\n\n1. **A** ...\n2. **B** ...\n3. **C** ...")
    state = {
        "query": "...",
        "aggregated": [{"name": "A", "category": "한식", "rating": 4.5,
                        "distance_m": 200, "price_level": 2, "review_count": 100,
                        "source_count": 2, "id": "x", "source": "kakao", "score": 0.7}],
        "plan": {"clarification_needed": []},
        "reflection_passed": False,
        "reflection_reason": "비선호 카테고리 충돌",
    }
    out = finalizer_node(state, llm=fake_llm)
    assert out["final_recommendation"]  # 추천 있음
    assert "완화" in out["final_text"] or "⚠" in out["final_text"]


def test_finalizer_builds_text_and_final_recommendation():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=(
        "1. **A** (한식, ★4.5, 도보 3분)\n   친구와 가기 좋은 가성비 한식\n"
        "2. **B** (...)\n3. **C** (...)\n\n적용 가정: 가격대 모더레이트"
    ))
    state = {
        "query": "전주 객사 근처...",
        "aggregated": [
            {"name": "A", "category": "한식", "rating": 4.5,
             "distance_m": 200, "price_level": 2, "review_count": 100,
             "source_count": 2, "id": "x", "source": "kakao", "score": 0.7},
            {"name": "B", "category": "한식", "rating": 4.3,
             "distance_m": 300, "price_level": 2, "review_count": 80,
             "source_count": 1, "id": "y", "source": "kakao", "score": 0.6},
            {"name": "C", "category": "한식", "rating": 4.2,
             "distance_m": 400, "price_level": 2, "review_count": 60,
             "source_count": 1, "id": "z", "source": "kakao", "score": 0.5},
        ],
    }
    out = finalizer_node(state, llm=fake_llm)
    assert len(out["final_recommendation"]) == 3
    assert "1." in out["final_text"]
