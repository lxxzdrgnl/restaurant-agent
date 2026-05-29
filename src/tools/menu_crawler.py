"""식당 메뉴 + 가격 추출.

1순위: 카카오맵 내부 panel3 API (httpx, JSON). Playwright 불필요.
2순위: 네이버 모바일 place 페이지 (Playwright). 카카오에 등록 안 된 가게용.
ToS 회색 영역이므로 best-effort. 실패는 빈 리스트로 fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_KAKAO_PANEL_URL = "https://place-api.map.kakao.com/places/panel3/{pid}"
_KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def fetch_menu_from_kakao(name: str, address_hint: str = "",
                          place_id: Optional[str] = None,
                          timeout_s: float = 5.0) -> list[dict[str, str]]:
    """카카오맵 panel3 API로 메뉴 JSON 직접 호출. Playwright 불필요.

    place_id가 주어지면 그대로 사용. 없으면 Kakao Local keyword search로 찾는다.
    Returns: [{"name":"물국수","price":"5,000원"}, ...] 또는 [].
    """
    pid = place_id
    if not pid:
        api_key = os.getenv("KAKAO_REST_API_KEY")
        if not api_key:
            return []
        try:
            q = f"{name} {address_hint.split()[0] if address_hint else ''}".strip()
            r = httpx.get(
                _KAKAO_KEYWORD_URL,
                params={"query": q, "size": 3,
                         "category_group_code": "FD6"},
                headers={"Authorization": f"KakaoAK {api_key}"},
                timeout=timeout_s,
            )
            if r.status_code != 200:
                return []
            docs = r.json().get("documents", [])
            if not docs:
                return []
            # 이름 fuzzy match — 너무 다른 가게 잡지 않게
            target = re.sub(r"\s+", "", name)
            best = next((d for d in docs
                          if target and target in re.sub(r"\s+", "", d.get("place_name", ""))),
                         docs[0])
            pid = str(best.get("id", ""))
        except Exception as e:  # noqa: BLE001
            logger.debug("kakao keyword search 실패 (%s): %s", name, e)
            return []
    if not pid:
        return []

    try:
        r = httpx.get(
            _KAKAO_PANEL_URL.format(pid=pid),
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": f"https://place.map.kakao.com/{pid}",
                "pf": "web",
            },
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return []
        items = (((r.json().get("menu") or {}).get("menus") or {}).get("items")) or []
    except Exception as e:  # noqa: BLE001
        logger.debug("kakao panel3 실패 (%s): %s", pid, e)
        return []

    out: list[dict[str, str]] = []
    for it in items[:10]:
        nm = (it.get("name") or "").strip()
        price = it.get("price")
        if not nm or price is None:
            continue
        try:
            n = int(price)
        except (TypeError, ValueError):
            continue
        out.append({"name": nm[:40], "price": f"{n:,}원"})
    return out

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
