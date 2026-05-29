"""aggregator 후 menu_enricher — top-k 후보에 대해 네이버 메뉴 크롤링.

graph가 멈추지 않도록 모든 실패는 menu_items=[] fallback."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.tools.menu_crawler import fetch_menu_from_naver

logger = logging.getLogger(__name__)


def menu_enricher_node(state: dict[str, Any]) -> dict[str, Any]:
    """aggregated 각 후보의 메뉴 정보를 채움. 실패해도 진행."""
    aggregated = state.get("aggregated") or []
    if not aggregated:
        return {
            "trace_log": state.get("trace_log", []) + [{
                "node": "menu_enricher",
                "enriched": 0,
                "skipped": "no candidates",
            }],
        }

    # 병렬 크롤링 (최대 5 worker — 차단 위험 낮춤)
    enriched_count = 0
    total = len(aggregated)
    results: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_idx = {
            pool.submit(
                fetch_menu_from_naver,
                r.get("name", ""),
                r.get("address", "") or "",
            ): i
            for i, r in enumerate(aggregated)
        }
        for fut in as_completed(future_to_idx, timeout=60):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result(timeout=15)
            except Exception as e:  # noqa: BLE001
                logger.info("menu_enricher: candidate %d failed: %s", i, e)
                results[i] = []

    # candidates에 menu_items 주입
    new_aggregated = []
    for i, r in enumerate(aggregated):
        items = results.get(i, [])
        new_r = dict(r)
        new_r["menu_items"] = items
        if items:
            enriched_count += 1
        new_aggregated.append(new_r)

    return {
        "aggregated": new_aggregated,
        "trace_log": state.get("trace_log", []) + [{
            "node": "menu_enricher",
            "total": total,
            "enriched": enriched_count,
            "empty": total - enriched_count,
        }],
    }
