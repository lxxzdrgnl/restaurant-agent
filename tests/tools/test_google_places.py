import httpx
import respx

from src.tools.google_places import search_google_places

URL = "https://places.googleapis.com/v1/places:searchText"


@respx.mock
def test_returns_results(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "places": [{
            "id": "ChIJxxx",
            "displayName": {"text": "백송갈비"},
            "formattedAddress": "전북 전주시 ...",
            "location": {"latitude": 35.8186, "longitude": 127.1489},
            "rating": 4.4,
            "userRatingCount": 218,
            "priceLevel": "PRICE_LEVEL_MODERATE",
            "primaryType": "korean_restaurant",
        }],
    }))
    out = search_google_places.invoke({
        "query": "전주 객사 저녁", "lat": 35.81, "lng": 127.14,
        "min_rating": 4.0, "price_levels": ["PRICE_LEVEL_MODERATE"],
    })
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비"
    assert r["rating"] == 4.4
    assert r["review_count"] == 218
    assert r["price_level"] == 2     # MODERATE → 2


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    out = search_google_places.invoke({"query": "x", "lat": 0, "lng": 0})
    assert out["error"] == "missing_api_key"


@respx.mock
def test_http_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(429))
    out = search_google_places.invoke({"query": "x", "lat": 0, "lng": 0})
    assert out["error"] == "google_http_error"
