import httpx
import pytest
import respx

from src.tools.kakao_local import search_kakao_local

URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@respx.mock
def test_returns_restaurants(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "id": "1",
                        "place_name": "백송갈비 객사점",
                        "category_name": "음식점 > 한식 > 육류,고기",
                        "address_name": "전북 전주시 완산구 ...",
                        "road_address_name": "전북 전주시 ...",
                        "x": "127.1489", "y": "35.8186",
                        "distance": "120",
                        "phone": "063-...",
                        "place_url": "http://place.map.kakao.com/1",
                    }
                ]
            },
        )
    )
    out = search_kakao_local.invoke(
        {"query": "저녁", "lat": 35.8186, "lng": 127.1489, "radius_m": 800}
    )
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비 객사점"
    assert r["source"] == "kakao"
    assert r["distance_m"] == 120
    assert r["category"] == "한식"


@respx.mock
def test_empty_results(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(return_value=httpx.Response(200, json={"documents": []}))
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["count"] == 0
    assert out["results"] == []


def test_missing_key(monkeypatch):
    monkeypatch.delenv("KAKAO_REST_API_KEY", raising=False)
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["error"] == "missing_api_key"


@respx.mock
def test_http_error_returns_empty_not_raises(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "k")
    respx.get(URL).mock(return_value=httpx.Response(503))
    out = search_kakao_local.invoke(
        {"query": "x", "lat": 0, "lng": 0, "radius_m": 100}
    )
    assert out["error"] == "kakao_http_error"
    assert out["results"] == []
