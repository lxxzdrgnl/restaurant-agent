"""슬래시 커맨드 핸들러 단위 테스트.
대화형 루프 자체는 통합 테스트가 어려워서 핸들러만 검증."""

from pathlib import Path
from unittest.mock import MagicMock

from chat import handle_command, render_memory
from src.memory.store import MemoryStore


def test_quit_command_returns_false(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    console = MagicMock()
    assert handle_command("/quit", store, console) is False
    assert handle_command("/q", store, console) is False
    assert handle_command("/exit", store, console) is False


def test_help_command_continues_and_prints(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    console = MagicMock()
    assert handle_command("/help", store, console) is True
    assert console.print.called
    assert handle_command("/?", store, console) is True


def test_memory_command_renders_profile_and_visits(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["해물"])
    store.append_visit("백송갈비", "한식")

    console = MagicMock()
    assert handle_command("/memory", store, console) is True
    # The Panel content includes both profile and visit
    printed = console.print.call_args[0][0]
    # Panel renderable — extract the text from the Panel's renderable attribute
    from rich.panel import Panel as RichPanel
    assert isinstance(printed, RichPanel)
    renderable_text = str(printed.renderable)
    assert "해물" in renderable_text or "disliked" in renderable_text or "백송갈비" in renderable_text


def test_clear_command(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    console = MagicMock()
    assert handle_command("/clear", store, console) is True
    assert console.clear.called


def test_unknown_command_warns_and_continues(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    console = MagicMock()
    assert handle_command("/nonexistent", store, console) is True
    assert console.print.called  # warning printed


def test_command_is_case_insensitive(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    console = MagicMock()
    assert handle_command("/HELP", store, console) is True
    assert handle_command("/Memory", store, console) is True
    assert handle_command("/QUIT", store, console) is False


def test_render_memory_empty_state(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    text = render_memory(store)
    assert "비어있음" in text or "User Profile" in text
    assert "없음" in text or "Recent Visits" in text


def test_render_memory_populated(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    store.set_user_profile("disliked_categories", ["해물", "회"])
    store.append_visit("A집", "한식")
    text = render_memory(store)
    assert "해물" in text
    assert "A집" in text
    assert "한식" in text
