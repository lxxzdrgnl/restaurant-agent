import os
from unittest.mock import patch

import pytest

from src.observability import telemetry


@pytest.fixture(autouse=True)
def reset_state():
    telemetry._provider = None
    yield
    telemetry._provider = None


def test_init_is_idempotent(monkeypatch):
    monkeypatch.setenv("PHOENIX_API_KEY", "k")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.rheon.kr/api/collect")
    with patch.object(telemetry, "register") as m:
        m.return_value.force_flush = lambda: None
        telemetry.init_telemetry()
        telemetry.init_telemetry()
    assert m.call_count == 1


def test_init_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")
    with patch.object(telemetry, "register") as m:
        telemetry.init_telemetry()
    m.assert_not_called()


def test_init_skips_when_key_missing(monkeypatch):
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    with patch.object(telemetry, "register") as m:
        telemetry.init_telemetry()
    m.assert_not_called()


def test_flush_safe_when_uninitialized():
    # Should not raise even if init was never called
    telemetry.flush_telemetry()
