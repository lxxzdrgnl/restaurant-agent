from __future__ import annotations

import os
import re
from typing import Any, Literal

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://openapi.naver.com/v1/search/local.json"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(text: str | None) -> str:
    return _TAG_RE.sub("", text or "")


def _category_leaf(category: str | None) -> str | None:
    if not category:
        return None
    return category.split(">")[0].strip()


@tool
def search_naver_local(
    query: str,
    sort: Literal["random", "comment"] = "comment",
) -> dict[str, Any]:
    """네이버 지역검색 API. display=5로 고정(API 제한).
    블로그 리뷰 풍부도 신호로 사용. sort='comment'면 리뷰 많은 곳 우선."""
    cid = os.getenv("NAVER_CLIENT_ID")
    cs = os.getenv("NAVER_CLIENT_SECRET")
    if not (cid and cs):
        return {"error": "missing_api_key", "results": [], "count": 0}

    try:
        with make_client() as client:
            resp = client.get(URL, params={"query": query, "display": 5, "sort": sort},
                              headers={"X-Naver-Client-Id": cid,
                                       "X-Naver-Client-Secret": cs})
    except Exception as e:  # noqa: BLE001
        return {"error": "naver_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "naver_http_error", "status": resp.status_code,
                "results": [], "count": 0}

    items = resp.json().get("items", [])
    results = []
    for i, it in enumerate(items):
        # Naver mapx/mapy are TM coords scaled by 10^7 → convert
        try:
            lng = float(it["mapx"]) / 10_000_000
            lat = float(it["mapy"]) / 10_000_000
        except (KeyError, TypeError, ValueError):
            lng = lat = None
        results.append({
            "id": f"n_{i}_{_strip(it.get('title',''))}",
            "name": _strip(it.get("title", "")),
            "source": "naver",
            "category": _category_leaf(it.get("category")),
            "address": it.get("roadAddress") or it.get("address"),
            "lat": lat, "lng": lng,
            "url": it.get("link"),
        })
    return {"results": results, "count": len(results)}
