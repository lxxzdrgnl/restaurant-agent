from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Source = Literal["kakao", "naver", "google"]


class Restaurant(BaseModel):
    id: str
    name: str
    source: Source
    category: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    distance_m: Optional[float] = None
    rating: Optional[float] = None              # 0..5 (Google)
    review_count: Optional[int] = None
    price_level: Optional[int] = None           # 0..4 (Google)
    phone: Optional[str] = None
    url: Optional[str] = None
    source_count: int = 1                       # set by aggregator after dedup
    score: Optional[float] = None               # set by aggregator


class AggregationWeights(BaseModel):
    rating: float = 0.35
    review: float = 0.25
    distance: float = 0.15
    match: float = 0.15
    price: float = 0.10


DEFAULT_WEIGHTS = AggregationWeights()


class KakaoParams(BaseModel):
    query: str = ""                              # ex: "전주 객사 한식". planner가 채움.
    category_group_code: str = "FD6"            # FD6=음식점, CE7=카페
    radius_m: int = 800
    sort: Literal["distance", "accuracy"] = "distance"
    size: int = 15


class NaverParams(BaseModel):
    query: str = ""                              # ex: "전주 객사 맛집". planner가 채움 (region 필수).
    sort: Literal["random", "comment"] = "comment"


class GoogleParams(BaseModel):
    query: str = ""                              # ex: "한식 맛집". planner가 채움.
    included_type: str = "restaurant"
    price_levels: Optional[list[str]] = None    # e.g. ["PRICE_LEVEL_MODERATE"]
    min_rating: Optional[float] = None
    open_now: Optional[bool] = None
    language_code: str = "ko"


class PostFilters(BaseModel):
    exclude_categories: list[str] = Field(default_factory=list)
    exclude_visited_within_days: int = 1
    k: int = 3


class Plan(BaseModel):
    region_query: str
    needs_geocoding: bool = True
    food_keywords: list[str] = Field(default_factory=list)   # ex: ["저녁", "한식"]
    kakao: KakaoParams = Field(default_factory=KakaoParams)
    naver: NaverParams = Field(default_factory=NaverParams)
    google: GoogleParams = Field(default_factory=GoogleParams)
    post_filters: PostFilters = Field(default_factory=PostFilters)
    weights: AggregationWeights = Field(default_factory=AggregationWeights)
    clarification_needed: list[str] = Field(default_factory=list)
