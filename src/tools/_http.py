"""Shared httpx Client factory. Centralized so respx can patch one place
and tools share connection settings."""

from __future__ import annotations

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


def make_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)
