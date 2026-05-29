from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from src.agent.types import (
    AggregationWeights,
    DEFAULT_WEIGHTS,
    Plan,
    Restaurant,
)


_PAREN_RE = re.compile(r"[\(\[].*?[\)\]]")
_PUNCT_RE = re.compile(r"[·\-,./]+")

# 한국어 음식 키워드 → Google Places의 영문 카테고리 substring
_CATEGORY_ALIASES: dict[str, list[str]] = {
    "중식": ["chinese"], "중국": ["chinese"], "중국집": ["chinese"],
    "한식": ["korean"],
    "일식": ["japanese", "sushi", "ramen"],
    "양식": ["italian", "western", "european", "american"],
    "이탈리안": ["italian"], "파스타": ["italian"], "피자": ["pizza"],
    "베트남": ["vietnamese"], "쌀국수": ["vietnamese"],
    "태국": ["thai"], "인도": ["indian"], "멕시칸": ["mexican"],
    "분식": ["snack"], "치킨": ["chicken"], "햄버거": ["hamburger", "burger"],
    "카페": ["cafe", "coffee"],
    "디저트": ["dessert", "bakery", "cake", "pastry"],
    "회": ["sushi", "seafood"], "해물": ["seafood"], "스시": ["sushi"],
    "라면": ["ramen"], "고기": ["barbecue", "grill"], "갈비": ["barbecue"],
}
# 너무 generic해서 mismatch 판정에서 제외할 카테고리
_GENERIC_CATEGORIES = {"음식점", "restaurant", "food", ""}


def _category_matches_keywords(category: str | None,
                                keywords: list[str]) -> bool:
    """plan.food_keywords가 있으면 후보 category와 (한국어 substring 또는 영문 alias) 매칭.
    keywords가 비어있거나 category가 generic이면 통과."""
    if not keywords:
        return True
    cat = (category or "").lower().strip()
    if cat in _GENERIC_CATEGORIES:
        return True
    for kw in keywords:
        kw_lo = kw.lower().strip()
        if kw_lo and kw_lo in cat:
            return True
        for alias in _CATEGORY_ALIASES.get(kw, []):
            if alias in cat:
                return True
    return False


def _normalize_name(name: str) -> str:
    """공백/괄호 부속어/구두점 제거 후 NFKC 정규화."""
    s = _PAREN_RE.sub("", name)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower().strip()


def _haversine_m(a_lat, a_lng, b_lat, b_lng) -> float | None:
    if None in (a_lat, a_lng, b_lat, b_lng):
        return None
    R = 6_371_000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def _same_restaurant(a: dict, b: dict, name_threshold: int = 85,
                     coord_threshold_m: float = 50.0) -> bool:
    name_sim = fuzz.token_set_ratio(_normalize_name(a["name"]),
                                    _normalize_name(b["name"]))
    if name_sim < name_threshold:
        return False
    dist = _haversine_m(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
    # 좌표 없으면 이름 유사도만으로 판단 (Naver가 일부 좌표 누락)
    return dist is None or dist <= coord_threshold_m


def _merge(group: list[dict]) -> dict:
    """소스별 강점에 따라 메타데이터 머지. group은 같은 식당 후보."""
    by_source = {r["source"]: r for r in group}
    kakao = by_source.get("kakao")
    naver = by_source.get("naver")
    google = by_source.get("google")
    head = kakao or naver or google or group[0]

    merged = dict(head)
    # name/address/category — Kakao 우선
    for k in ("name", "address", "category"):
        merged[k] = (kakao or {}).get(k) or (naver or {}).get(k) or (google or {}).get(k)
    # rating/price_level — Google 우선
    merged["rating"] = (google or {}).get("rating") or (kakao or {}).get("rating")
    merged["price_level"] = (google or {}).get("price_level")
    # review_count = max
    rev = [r.get("review_count") for r in group if r.get("review_count")]
    merged["review_count"] = max(rev) if rev else None
    # distance — kakao가 들고 있는 게 보통 가장 정확
    merged["distance_m"] = (kakao or {}).get("distance_m") or (google or {}).get("distance_m")
    # 좌표
    for k in ("lat", "lng"):
        merged[k] = (kakao or {}).get(k) or (google or {}).get(k) or (naver or {}).get(k)
    merged["source_count"] = len(by_source)
    merged["id"] = "+".join(sorted(by_source.keys())) + ":" + (merged["name"] or "")
    merged["source"] = head["source"]  # primary
    return merged


def _dedup_and_merge(raw: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for r in raw:
        if not r.get("name"):
            continue
        for g in groups:
            if any(_same_restaurant(existing, r) for existing in g):
                g.append(r); break
        else:
            groups.append([r])
    return [_merge(g) for g in groups]


def _norm_clip(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _score(r: dict, w: AggregationWeights) -> float:
    rating = _norm_clip(r.get("rating") or 0.0, 0.0, 5.0)
    review = _norm_clip(math.log1p(r.get("review_count") or 0), 0.0, math.log1p(10_000))
    dist = r.get("distance_m")
    distance = _norm_clip(1 / ((dist or 1_000) + 50), 0.0, 1 / 50)
    match = (r.get("source_count") or 1) / 3
    price = _norm_clip(float(r.get("price_level") or 2), 0.0, 4.0)
    return (
        w.rating * rating
        + w.review * review
        + w.distance * distance
        + w.match * match
        - w.price * price
    )


def _is_recently_visited(
    name: str,
    recent: list[dict],
    window_days: int,
    now: datetime,
) -> bool:
    """실제 방문(user_logged / seed)만 체크. 추천 기록(recommended)은 무시."""
    target = _normalize_name(name)
    for v in recent:
        # 사용자가 실제로 갔다고 보고한 것만 차단. "추천만 받았던 곳"은 재추천 가능.
        if v.get("source") == "recommended":
            continue
        if _normalize_name(v["name"]) != target:
            continue
        try:
            visited_at = datetime.fromisoformat(v.get("visited_at", ""))
        except (TypeError, ValueError):
            continue
        if (now - visited_at).total_seconds() / 86400 <= window_days:
            return True
    return False


def aggregator_node(state: dict[str, Any]) -> dict[str, Any]:
    """결정론적 코드 노드. LLM 호출 없음."""
    plan = Plan.model_validate(state["plan"])
    raw = state.get("candidates", [])
    # Trust planner. planner.py already merges user_profile dislikes AND
    # subtracts session_allowed. aggregator must NOT re-add them.
    user_dislikes = set(plan.post_filters.exclude_categories)
    weights = plan.weights or DEFAULT_WEIGHTS

    merged = _dedup_and_merge(raw)

    # filter
    recent = state.get("recent_visits", [])
    now = datetime.now()
    window = plan.post_filters.exclude_visited_within_days
    max_price = plan.post_filters.max_price_level  # None이면 가격 차단 없음
    filtered = []
    excluded_by_recency = 0
    excluded_by_category = 0
    excluded_by_price = 0
    excluded_by_mismatch = 0
    food_keywords = plan.food_keywords or []
    for r in merged:
        if r.get("category") in user_dislikes:
            excluded_by_category += 1
            continue
        if _is_recently_visited(r["name"], recent, window_days=window, now=now):
            excluded_by_recency += 1
            continue
        # 가격 강제 차단 — price_level 정보가 있고 max_price를 초과하면 제외.
        # 정보 없는 후보(None)는 통과 (false negative 방지: 정보가 없을 뿐 비싸다는 증거 X).
        if max_price is not None:
            pl = r.get("price_level")
            if pl is not None and pl > max_price:
                excluded_by_price += 1
                continue
        # 음식 종류 mismatch 강제 차단 — "중국집" 요청에 vietnamese_restaurant이
        # 평점/리뷰 가중치 때문에 top-1로 올라오는 케이스 방지.
        if not _category_matches_keywords(r.get("category"), food_keywords):
            excluded_by_mismatch += 1
            continue
        filtered.append(r)

    # score + sort
    for r in filtered:
        r["score"] = _score(r, weights)
    filtered.sort(key=lambda r: r["score"], reverse=True)

    top_k = filtered[: plan.post_filters.k]
    return {
        "aggregated": [Restaurant.model_validate(r).model_dump() for r in top_k],
        "trace_log": state.get("trace_log", []) + [{
            "node": "aggregator",
            "raw_count": len(raw),
            "merged_count": len(merged),
            "excluded_by_category": excluded_by_category,
            "excluded_by_recency": excluded_by_recency,
            "excluded_by_price": excluded_by_price,
            "excluded_by_mismatch": excluded_by_mismatch,
            "kept": len(top_k),
        }],
    }
