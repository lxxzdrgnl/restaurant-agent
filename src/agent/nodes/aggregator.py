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

# 음식 키워드 → 매칭 alias (한국어 카테고리 + 영문 카테고리 substring).
# 구체 메뉴 키도 상위 카테고리를 포함시켜, 사용자가 "라멘"만 적어도
# 카카오의 "일식" 카테고리에 등록된 후보가 mismatch로 잘못 차단되지 않게 함.
_CATEGORY_ALIASES: dict[str, list[str]] = {
    # 상위 카테고리
    "중식": ["chinese"], "중국": ["chinese"], "중국집": ["chinese"],
    "한식": ["korean"],
    "일식": ["japanese", "sushi", "ramen"],
    "양식": ["italian", "western", "european", "american"],
    "이탈리안": ["italian"],
    "베트남": ["vietnamese"], "태국": ["thai"], "인도": ["indian"],
    "멕시칸": ["mexican"],
    "분식": ["snack"], "치킨": ["chicken"], "햄버거": ["hamburger", "burger"],
    "카페": ["cafe", "coffee"],
    "디저트": ["dessert", "bakery", "cake", "pastry"],
    "해물": ["seafood"], "회": ["sushi", "seafood"],
    "고기": ["barbecue", "grill"],
    # 구체 메뉴 → 상위 카테고리 + 영문 alias
    "라멘": ["일식", "japanese", "ramen"],
    "라면": ["일식", "japanese", "ramen"],
    "마라탕": ["중식", "chinese"],
    "짬뽕": ["중식", "chinese"],
    "짜장": ["중식", "chinese"], "짜장면": ["중식", "chinese"],
    "탕수육": ["중식", "chinese"],
    "스시": ["일식", "japanese", "sushi"],
    "초밥": ["일식", "japanese", "sushi"],
    "돈가스": ["일식", "japanese"], "돈까스": ["일식", "japanese"],
    "우동": ["일식", "japanese"],
    "소바": ["일식", "japanese"],
    "쌀국수": ["베트남", "vietnamese"],
    "파스타": ["양식", "italian"],
    "피자": ["양식", "italian", "pizza"],
    "스테이크": ["양식", "western", "steak"],
    "비빔밥": ["한식", "korean"],
    "갈비": ["한식", "korean", "barbecue"],
    "삼겹살": ["한식", "korean", "barbecue"],
    "냉면": ["한식", "korean"],
    "국밥": ["한식", "korean"],
    "찌개": ["한식", "korean"],
    "백반": ["한식", "korean"],
    "김밥": ["분식", "한식", "korean"],
    "떡볶이": ["분식", "한식", "korean"],
}
# 너무 generic해서 mismatch 판정에서 제외할 카테고리
_GENERIC_CATEGORIES = {"음식점", "restaurant", "food", ""}


def _category_matches_keywords(category: str | None,
                                keywords: list[str],
                                acceptable: list[str] | None = None) -> bool:
    """후보 category가 acceptable_categories 또는 food_keywords와 매칭되는지.

    - acceptable이 비어있지 않으면 그것만 사용 (planner가 LLM으로 결정한 화이트리스트).
    - acceptable이 비어있으면 keywords + 내장 _CATEGORY_ALIASES로 fallback.
    - generic 카테고리("음식점"/"restaurant")는 항상 통과.
    """
    cat = (category or "").lower().strip()
    if cat in _GENERIC_CATEGORIES:
        return True
    tokens = list(acceptable or [])
    if not tokens:
        if not keywords:
            return True
        tokens = list(keywords)
        for kw in keywords:
            tokens.extend(_CATEGORY_ALIASES.get(kw, []))
    for tok in tokens:
        t = tok.lower().strip()
        if t and t in cat:
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
    mismatched = 0
    food_keywords = plan.food_keywords or []
    acceptable = plan.acceptable_categories or []
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
        # 음식 종류 mismatch — soft penalty. LLM이 acceptable_categories를
        # 빠뜨려도 정당한 후보가 살아남도록 hard cutoff 대신 score를 강하게 깎음.
        # 명백한 mismatch (베트남집이 중식 쿼리에 끼는 등)는 페널티로도 top-k 밖.
        if not _category_matches_keywords(r.get("category"), food_keywords,
                                            acceptable):
            r["_mismatch"] = True
            mismatched += 1
        filtered.append(r)

    # score + sort. mismatch 후보는 페널티 0.4× (정렬에서 자연 도태 보조).
    normal: list[dict] = []
    mismatched_list: list[dict] = []
    for r in filtered:
        is_mismatch = r.pop("_mismatch", False)
        s = _score(r, weights)
        if is_mismatch:
            s *= 0.4
        r["score"] = s
        (mismatched_list if is_mismatch else normal).append(r)
    normal.sort(key=lambda r: r["score"], reverse=True)
    mismatched_list.sort(key=lambda r: r["score"], reverse=True)

    # Hybrid: 정상 후보가 k개 이상이면 mismatch는 전부 제외,
    # 부족하면 부족분만 mismatch에서 채움.
    k = plan.post_filters.k
    top_k = normal[:k]
    if len(top_k) < k:
        top_k += mismatched_list[: k - len(top_k)]
    mismatched_filled = max(0, k - len(normal))
    mismatched_dropped = len(mismatched_list) - mismatched_filled
    return {
        "aggregated": [Restaurant.model_validate(r).model_dump() for r in top_k],
        "trace_log": state.get("trace_log", []) + [{
            "node": "aggregator",
            "raw_count": len(raw),
            "merged_count": len(merged),
            "excluded_by_category": excluded_by_category,
            "excluded_by_recency": excluded_by_recency,
            "excluded_by_price": excluded_by_price,
            "mismatched_total": mismatched,
            "mismatched_filled": mismatched_filled,
            "mismatched_dropped": mismatched_dropped,
            "kept": len(top_k),
        }],
    }
