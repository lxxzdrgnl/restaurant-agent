"""그래프 빌더 스모크 테스트: 노드/엣지 토폴로지 + 한 번 invoke 가능한지."""

from unittest.mock import MagicMock

from src.agent.graph import build_graph
from src.memory.store import MemoryStore


def test_graph_clarification_path_skips_react_agent(tmp_path, monkeypatch):
    """planner가 clarification_needed를 반환하면 react_agent 없이 finalizer로 직행."""
    monkeypatch.setenv("PHOENIX_DISABLED", "1")

    fake_llm = MagicMock()
    fake_llm.bind.return_value = fake_llm  # bind(response_format=...) → same mock

    # react_agent가 호출되면 실패하도록 마킹
    react_called = []

    def stub_react_node(state):
        react_called.append(True)
        return {"candidates": []}

    store = MemoryStore(tmp_path / "g_clarify.db")
    graph = build_graph(
        llm=fake_llm,
        store=store,
        react_node_override=stub_react_node,
    )

    fake_llm.invoke.side_effect = [
        # preference_extractor (new)
        MagicMock(content='{"add_disliked":[],"remove_disliked":[],"log_visits":[]}'),
        # planner: returns plan with clarification_needed non-empty
        MagicMock(content=(
            '{"region_query":"전주 객사","needs_geocoding":true,"food_keywords":["회"],'
            '"kakao":{"query":"회","category_group_code":"FD6","radius_m":800,"sort":"distance","size":15},'
            '"naver":{"query":"전주 객사 회 맛집","sort":"comment"},'
            '"google":{"query":"회 맛집","included_type":"restaurant","price_levels":null,"min_rating":null,'
            '"open_now":null,"language_code":"ko"},'
            '"post_filters":{"exclude_categories":[],"exclude_visited_within_days":1,"k":3},'
            '"weights":{"rating":0.35,"review":0.25,"distance":0.15,"match":0.15,"price":0.10},'
            '"clarification_needed":["test clarification"]}'
        )),
        # finalizer
        MagicMock(content="⚠️ 추가 정보가 필요합니다\n\n- test clarification"),
    ]

    result = graph.invoke({"query": "전주 객사 회 맛집"})
    assert "final_text" in result
    assert not react_called  # react_agent never invoked
    assert result.get("final_text") is not None


def test_graph_compiles_and_invokes(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")

    fake_llm = MagicMock()
    fake_llm.bind.return_value = fake_llm  # bind(response_format=...) → same mock
    # planner returns minimal valid plan
    fake_llm.invoke.return_value = MagicMock(content=(
        '{"region_query":"x","needs_geocoding":false,"food_keywords":[],'
        '"kakao":{"category_group_code":"FD6","radius_m":800,"sort":"distance","size":15},'
        '"naver":{"sort":"comment"},'
        '"google":{"included_type":"restaurant","price_levels":null,"min_rating":null,'
        '"open_now":null,"language_code":"ko"},'
        '"post_filters":{"exclude_categories":[],"exclude_visited_within_days":1,"k":3},'
        '"weights":{"rating":0.35,"review":0.25,"distance":0.15,"match":0.15,"price":0.10},'
        '"clarification_needed":[]}'
    ))

    # react_agent를 통째로 스텁: 빈 candidates만 채워주는 함수로 대체
    def stub_react_node(state):
        return {"candidates": []}

    store = MemoryStore(tmp_path / "g.db")
    graph = build_graph(
        llm=fake_llm,
        store=store,
        react_node_override=stub_react_node,
    )
    # passed=true 응답 시퀀스 (reflector에서 즉시 통과)
    fake_llm.invoke.side_effect = [
        # preference_extractor (new)
        MagicMock(content='{"add_disliked":[],"remove_disliked":[],"log_visits":[]}'),
        # planner
        fake_llm.invoke.return_value,
        # reflector
        MagicMock(content='{"passed":true,"reason":"ok","suggested_relaxation":null}'),
        # finalizer
        MagicMock(content="1. ...\n2. ...\n3. ..."),
    ]

    result = graph.invoke({"query": "test"})
    assert "final_text" in result
    assert isinstance(result.get("trace_log"), list)
