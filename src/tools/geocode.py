from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from src.tools._http import make_client

KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@tool
def geocode(region: str) -> dict[str, Any]:
    """지역명·랜드마크·주소를 위경도 좌표로 변환한다.

    Args:
        region: 한국어 지역 표현 ("전주 객사", "홍대 정문", "서울시청" 등).
    Returns:
        성공: {"lat":..., "lng":..., "matched_address":...}
        실패: {"error":..., "message":..., "suggestions":[...]}
    """
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        return {"error": "missing_api_key",
                "message": "KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다."}

    try:
        with make_client() as client:
            resp = client.get(
                KAKAO_KEYWORD_URL,
                params={"query": region, "size": 5},
                headers={"Authorization": f"KakaoAK {api_key}"},
            )
    except Exception as e:  # noqa: BLE001
        return {"error": "kakao_network_error", "message": str(e)}

    if resp.status_code >= 400:
        return {"error": "kakao_http_error", "status": resp.status_code}

    docs = resp.json().get("documents", [])
    if not docs:
        return {
            "error": "region_not_found",
            "message": f"'{region}'에 해당하는 장소를 찾지 못했습니다.",
            "suggestions": [
                "지역명을 더 구체적으로 ('전주 객사' → '전주시 완산구 객사길')",
                "근처 랜드마크나 행정구역명으로 다시 입력해주세요",
            ],
        }

    top = docs[0]
    return {
        "lat": float(top["y"]),
        "lng": float(top["x"]),
        "matched_address": top.get("address_name") or top.get("road_address_name", ""),
        "matched_name": top.get("place_name", ""),
    }
