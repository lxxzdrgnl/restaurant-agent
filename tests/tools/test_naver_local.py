import httpx
import respx

from src.tools.naver_local import search_naver_local

URL = "https://openapi.naver.com/v1/search/local.json"


@respx.mock
def test_returns_results(monkeypatch):
    monkeypatch.setenv("NAVER_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "secret")
    respx.get(URL).mock(return_value=httpx.Response(200, json={
        "items": [{
            "title": "<b>백송</b>갈비",
            "category": "한식>육류,고기",
            "address": "전북 전주시 완산구",
            "roadAddress": "전북 전주시 ...",
            "mapx": "1271489000", "mapy": "358186000",
            "link": "http://...",
        }],
    }))
    out = search_naver_local.invoke({"query": "전주 객사 맛집"})
    assert out["count"] == 1
    r = out["results"][0]
    assert r["name"] == "백송갈비"            # HTML 태그 제거
    assert r["source"] == "naver"
    assert r["category"] == "한식"            # leaf 추출


def test_missing_keys(monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)
    out = search_naver_local.invoke({"query": "x"})
    assert out["error"] == "missing_api_key"
    assert out["results"] == []


@respx.mock
def test_http_error(monkeypatch):
    monkeypatch.setenv("NAVER_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "secret")
    respx.get(URL).mock(return_value=httpx.Response(500))
    out = search_naver_local.invoke({"query": "x"})
    assert out["error"] == "naver_http_error"
