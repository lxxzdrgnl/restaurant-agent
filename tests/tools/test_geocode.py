import httpx
import pytest
import respx

from src.tools.geocode import geocode


KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@respx.mock
def test_geocode_success(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "place_name": "전주 객사",
                        "address_name": "전북 전주시 완산구 중앙동3가",
                        "x": "127.1489",
                        "y": "35.8186",
                    }
                ]
            },
        )
    )
    result = geocode.invoke({"region": "전주 객사"})
    assert result["lat"] == pytest.approx(35.8186)
    assert result["lng"] == pytest.approx(127.1489)
    assert "전주" in result["matched_address"]


@respx.mock
def test_geocode_not_found(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(
        return_value=httpx.Response(200, json={"documents": []})
    )
    result = geocode.invoke({"region": "외계행성 뮤뮤성"})
    assert result["error"] == "region_not_found"
    assert "suggestions" in result


@respx.mock
def test_geocode_http_error(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(KAKAO_URL).mock(return_value=httpx.Response(500))
    result = geocode.invoke({"region": "전주 객사"})
    assert result["error"] == "kakao_http_error"


def test_geocode_missing_key(monkeypatch):
    monkeypatch.delenv("KAKAO_REST_API_KEY", raising=False)
    result = geocode.invoke({"region": "전주 객사"})
    assert result["error"] == "missing_api_key"
