from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def _category_leaf(category_name: str | None) -> str | None:
    """'음식점 > 한식 > 육류,고기' → '한식' (두 번째 레벨이 가장 유용)."""
    if not category_name:
        return None
    parts = [p.strip() for p in category_name.split(">")]
    return parts[1] if len(parts) >= 2 else parts[-1]


@tool
def search_kakao_local(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 800,
    category_group_code: str = "FD6",
    sort: Literal["distance", "accuracy"] = "distance",
    size: int = 15,
) -> dict[str, Any]:
    """카카오 로컬 API로 좌표 반경 안의 음식점/카페를 검색한다.

    Args:
        query: 키워드 ("저녁", "비빔밥", "파스타").
        lat, lng: 검색 중심 좌표.
        radius_m: 반경(미터, 최대 20000).
        category_group_code: FD6=음식점, CE7=카페/디저트.
        sort: distance | accuracy.
        size: 최대 15.
    """
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        return {"error": "missing_api_key", "results": [], "count": 0}

    params = {
        "query": query,
        "x": lng, "y": lat,
        "radius": radius_m,
        "category_group_code": category_group_code,
        "sort": sort,
        "size": min(size, 15),
    }
    try:
        with make_client() as client:
            resp = client.get(URL, params=params,
                              headers={"Authorization": f"KakaoAK {api_key}"})
    except Exception as e:  # noqa: BLE001
        return {"error": "kakao_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "kakao_http_error", "status": resp.status_code,
                "results": [], "count": 0}

    docs = resp.json().get("documents", [])
    results = [{
        "id": f"k_{d.get('id', '')}",
        "name": d.get("place_name", ""),
        "source": "kakao",
        "category": _category_leaf(d.get("category_name")),
        "address": d.get("road_address_name") or d.get("address_name"),
        "lat": float(d["y"]) if d.get("y") else None,
        "lng": float(d["x"]) if d.get("x") else None,
        "distance_m": int(d["distance"]) if d.get("distance") else None,
        "phone": d.get("phone") or None,
        "url": d.get("place_url"),
    } for d in docs]
    return {"results": results, "count": len(results)}
