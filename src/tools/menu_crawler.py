"""네이버 모바일 place 페이지에서 식당 메뉴 + 가격을 추출.

ToS 회색 영역이므로 best-effort. 실패는 빈 리스트로 fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Naver place_id 추출 패턴
_PLACE_ID_RE = re.compile(r"/restaurant/(\d+)")

# 가격 정규화: "7,000원", "₩7000", "7000원" → "7,000원"
_PRICE_RE = re.compile(r"([\d,]+)\s*원")


def _normalize_price(raw: str) -> str:
    m = _PRICE_RE.search(raw or "")
    if not m:
        return (raw or "").strip()
    n = int(m.group(1).replace(",", ""))
    return f"{n:,}원"


def fetch_menu_from_naver(name: str, address_hint: str = "",
                          timeout_ms: int = 8000) -> list[dict[str, str]]:
    """식당 이름 + 주소 일부로 네이버 모바일 검색 → 첫 결과 → 메뉴 페이지 크롤링.
    Returns: [{"name": "짜장면", "price": "7,000원"}, ...] 또는 [].
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.warning("Playwright 미설치. 메뉴 크롤링 비활성.")
        return []

    query = f"{name} {address_hint.split()[0] if address_hint else ''}".strip()

    with sync_playwright() as p:
        # Ubuntu 26.04 등에서 Playwright 자체 Chromium 설치가 불가하면
        # 시스템에 깔린 google-chrome을 우선 사용.
        custom_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
        launch_attempts = []
        if custom_path and os.path.exists(custom_path):
            launch_attempts.append({"executable_path": custom_path,
                                     "args": ["--no-sandbox"]})
        # system Chrome auto-detection
        launch_attempts.append({"channel": "chrome", "args": ["--no-sandbox"]})
        # default Playwright Chromium
        launch_attempts.append({})

        browser = None
        for kwargs in launch_attempts:
            try:
                browser = p.chromium.launch(headless=True, **kwargs)
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("launch attempt %s failed: %s", kwargs, e)
                continue
        if browser is None:
            logger.warning("Playwright Chromium 실행 실패 — 모든 fallback 소진")
            return []

        try:
            context = browser.new_context(
                user_agent=("Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) "
                            "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"),
                viewport={"width": 375, "height": 812},
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            # 1) 검색 페이지
            search_url = f"https://m.place.naver.com/restaurant/list?query={query}"
            page.goto(search_url, wait_until="domcontentloaded")

            # 2) 첫 결과의 place_id 추출
            try:
                first = page.locator("a[href*='/restaurant/']").first
                href = first.get_attribute("href", timeout=timeout_ms) or ""
            except PWTimeout:
                return []
            m = _PLACE_ID_RE.search(href)
            if not m:
                return []
            place_id = m.group(1)

            # 3) menu 페이지 이동
            menu_url = f"https://m.place.naver.com/restaurant/{place_id}/menu"
            page.goto(menu_url, wait_until="domcontentloaded")

            # 4) menu 아이템 추출 — DOM selector는 페이지 구조에 따라 시도
            items: list[dict[str, str]] = []
            # 흔한 패턴들 — Naver place의 menu list class는 자주 바뀜
            selectors = [
                "li[class*='menu']",
                "div[class*='MenuList'] li",
                "ul[class*='menu_list'] li",
                "div[class*='menu_item']",
            ]
            menu_locator = None
            for sel in selectors:
                if page.locator(sel).count() > 0:
                    menu_locator = page.locator(sel)
                    break

            if menu_locator is None:
                return []

            for i in range(min(menu_locator.count(), 10)):
                el = menu_locator.nth(i)
                try:
                    text = el.inner_text(timeout=2000).strip()
                except PWTimeout:
                    continue
                if not text:
                    continue
                # text에서 가격 패턴 추출 (가격 라인 + 이름 라인)
                price_match = _PRICE_RE.search(text)
                if not price_match:
                    continue
                price_str = _normalize_price(price_match.group(0))
                # 메뉴 이름: 가격 앞의 첫 라인
                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                name_line = lines[0] if lines else "?"
                # 가격이 이름에 섞여있을 수도 — 제거
                name_clean = _PRICE_RE.sub("", name_line).strip()
                if name_clean:
                    items.append({"name": name_clean[:40], "price": price_str})

            return items[:10]
        except Exception as e:  # noqa: BLE001
            logger.info("Naver menu crawl 실패 (%s): %s", name, e)
            return []
        finally:
            browser.close()
