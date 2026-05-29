from unittest.mock import patch

from src.agent.nodes.menu_enricher import menu_enricher_node


@patch("src.agent.nodes.menu_enricher.fetch_menu_from_naver", return_value=[])
@patch("src.agent.nodes.menu_enricher.fetch_menu_from_kakao")
def test_enriches_aggregated_with_menu_items(mock_kakao, mock_naver):
    """카카오 API가 메뉴를 반환하면 aggregated의 각 후보에 menu_items로 주입."""
    mock_kakao.side_effect = [
        [{"name": "짜장면", "price": "7,000원"},
         {"name": "짬뽕", "price": "9,000원"}],
        [{"name": "비빔밥", "price": "12,000원"}],
    ]
    state = {
        "aggregated": [
            {"name": "복성루", "address": "전북 전주시", "id": "k_111"},
            {"name": "하숙영", "address": "전북 전주시", "id": "k_222"},
        ],
    }
    out = menu_enricher_node(state)
    by_name = {c["name"]: c for c in out["aggregated"]}
    assert by_name["복성루"]["menu_items"][0]["name"] == "짜장면"
    assert by_name["하숙영"]["menu_items"][0]["price"] == "12,000원"
    assert out["trace_log"][-1]["enriched"] == 2


@patch("src.agent.nodes.menu_enricher.fetch_menu_from_naver", return_value=[])
@patch("src.agent.nodes.menu_enricher.fetch_menu_from_kakao")
def test_failure_in_one_candidate_does_not_stop_graph(mock_kakao, mock_naver):
    """한 식당 결과가 빈 결과여도 다른 후보는 정상 enrich. 노드는 통과."""
    mock_kakao.side_effect = [[], [{"name": "비빔밥", "price": "10,000원"}]]
    state = {
        "aggregated": [
            {"name": "실패집", "address": "", "id": "k_1"},
            {"name": "성공집", "address": "", "id": "k_2"},
        ],
    }
    out = menu_enricher_node(state)
    assert out["trace_log"][-1]["enriched"] == 1
    assert out["trace_log"][-1]["empty"] == 1


@patch("src.agent.nodes.menu_enricher.fetch_menu_from_naver", return_value=[])
@patch("src.agent.nodes.menu_enricher.fetch_menu_from_kakao")
def test_no_candidates_skipped(mock_kakao, mock_naver):
    """aggregated가 비어있으면 크롤러 호출 안 함."""
    out = menu_enricher_node({"aggregated": []})
    mock_kakao.assert_not_called()
    mock_naver.assert_not_called()
    assert out["trace_log"][-1].get("skipped") == "no candidates"
