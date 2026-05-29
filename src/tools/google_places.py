from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.tools import tool

from src.tools._http import make_client

URL = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.rating,places.userRatingCount,places.priceLevel,places.primaryType,"
    "places.websiteUri,places.nationalPhoneNumber"
)
PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


@tool
def search_google_places(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 800,
    included_type: str = "restaurant",
    price_levels: Optional[list[str]] = None,
    min_rating: Optional[float] = None,
    open_now: Optional[bool] = None,
    language_code: str = "ko",
) -> dict[str, Any]:
    """Google Places (New) Text Search. 평점·가격대를 API 단에서 거른다."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return {"error": "missing_api_key", "results": [], "count": 0}

    # locationRestriction(hard) 사용 — locationBias는 weak이라 전국 단위 텍스트 매칭이
    # 들어오면 다른 도시 결과까지 섞임("전북대 중국집" 검색에 대전·대구·광주 식당이 들어오는 케이스).
    # restriction은 지정 반경 밖 결과를 API가 반환하지 않음.
    # Google Places API의 textSearch + locationRestriction은 circle을 지원하지 않고
    # rectangle만 받으므로 center±radius로 사각형 변환.
    import math
    lat_delta = radius_m / 111_000.0
    lng_delta = radius_m / max(111_000.0 * math.cos(math.radians(lat)), 1.0)
    body: dict[str, Any] = {
        "textQuery": query,
        "languageCode": language_code,
        "includedType": included_type,
        "locationRestriction": {
            "rectangle": {
                "low": {"latitude": lat - lat_delta, "longitude": lng - lng_delta},
                "high": {"latitude": lat + lat_delta, "longitude": lng + lng_delta},
            }
        },
    }
    if price_levels:
        body["priceLevels"] = price_levels
    if min_rating is not None:
        body["minRating"] = min_rating
    if open_now is not None:
        body["openNow"] = open_now

    try:
        with make_client() as client:
            resp = client.post(URL, json=body, headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": FIELD_MASK,
                "Content-Type": "application/json",
            })
    except Exception as e:  # noqa: BLE001
        return {"error": "google_network_error", "message": str(e),
                "results": [], "count": 0}

    if resp.status_code >= 400:
        return {"error": "google_http_error", "status": resp.status_code,
                "detail": resp.text[:300],
                "results": [], "count": 0}

    places = resp.json().get("places", [])
    results = []
    for p in places:
        loc = p.get("location") or {}
        results.append({
            "id": f"g_{p.get('id', '')}",
            "name": (p.get("displayName") or {}).get("text", ""),
            "source": "google",
            "category": p.get("primaryType"),
            "address": p.get("formattedAddress"),
            "lat": loc.get("latitude"),
            "lng": loc.get("longitude"),
            "rating": p.get("rating"),
            "review_count": p.get("userRatingCount"),
            "price_level": PRICE_LEVEL_MAP.get(p.get("priceLevel")),
            "phone": p.get("nationalPhoneNumber"),
            "url": p.get("websiteUri"),
        })
    return {"results": results, "count": len(results)}
