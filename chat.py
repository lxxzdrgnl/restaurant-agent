"""대화형 맛집 추천 CLI.

같은 LangGraph 객체와 메모리를 세션 내내 유지하므로,
이전 추천이 visit_history에 누적되고 다음 추천에서 자동 제외된다.
이 흐름이 Memory 패턴의 동작을 가장 명확히 보여준다.

사용:
    uv run python chat.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory

load_dotenv()

# Phoenix는 LangChain import 전에 초기화
from src.observability.telemetry import init_telemetry, flush_telemetry  # noqa: E402
init_telemetry()

from langchain_openai import ChatOpenAI  # noqa: E402
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.graph import build_graph
from src.memory.store import MemoryStore


# 슬래시 커맨드 카탈로그 — autocomplete + ghost-suggest source of truth
SLASH_COMMANDS = [
    "/help",
    "/?",
    "/memory",
    "/memory clear",
    "/memory dislike add",
    "/memory dislike remove",
    "/allow",
    "/allow clear",
    "/clear",
    "/quit",
    "/exit",
    "/q",
]


class _SlashAutoSuggest(AutoSuggest):
    """현재 입력으로 시작하는 슬래시 커맨드를 ghost text로 제안."""

    def __init__(self, commands: list[str]):
        self._commands = commands

    def get_suggestion(self, buffer, document):
        text = document.text
        if not text.startswith("/"):
            return None
        # Find first command that starts with current text and is longer
        for cmd in self._commands:
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text):])
        return None


def _build_completer() -> NestedCompleter:
    return NestedCompleter.from_nested_dict({
        "/help": None,
        "/?": None,
        "/memory": {
            "clear": None,
            "dislike": {
                "add": None,
                "remove": None,
            },
        },
        "/allow": {
            "clear": None,
        },
        "/clear": None,
        "/quit": None,
        "/exit": None,
        "/q": None,
    })


LOGO = r"""███╗   ███╗ █████╗ ████████╗███████╗██╗██████╗
████╗ ████║██╔══██╗╚══██╔══╝╚══███╔╝██║██╔══██╗
██╔████╔██║███████║   ██║     ███╔╝ ██║██████╔╝
██║╚██╔╝██║██╔══██║   ██║    ███╔╝  ██║██╔═══╝
██║ ╚═╝ ██║██║  ██║   ██║   ███████╗██║██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝"""

HELP_TEXT = """\
[bold cyan]❯[/] Commands
  /help, /?                      show this help
  /memory                        show stored profile + recent visits
  /memory clear                  wipe all memory (profile + visits)
  /memory dislike add <cat>      add disliked category
  /memory dislike remove <cat>   remove disliked category
  /allow <category>              allow disliked category for THIS session only
  /allow clear                   reset session-allowed
  /clear                         clear screen
  /quit, /exit, /q               exit (or Ctrl-D)

[bold cyan]❯[/] Tip:
  Type [cyan]/[/] to see suggestions. Press [cyan]→[/] or [cyan]Tab[/] to autocomplete.
  Press [cyan]↑[/] / [cyan]↓[/] for command history.

[bold cyan]❯[/] Multi-turn clarification:
  After agent asks about memory conflict, reply:
    yes / 응 / ok    — temporarily allow, re-run original query
    no  / 아니         — cancel, ask differently

[bold cyan]❯[/] Or type a natural-language query:
  전주 객사 근처에서 친구랑 저녁...
"""

# 이번 세션에만 허용된 카테고리 (disliked에서 임시 제외)
_SESSION_ALLOWED: set[str] = set()

# 마지막 clarification 상태 (multi-turn 흐름에서 사용)
_LAST_CLARIFICATION: dict | None = None


def render_welcome(console: Console) -> None:
    console.print()
    console.print(LOGO, style="bold cyan")
    console.print()
    console.print(
        "LangGraph ReAct Agent for restaurant recommendation",
        style="dim",
    )
    console.print(
        f"Model: [cyan]{os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')}[/] · type [cyan]/help[/] for commands",
        style="dim",
    )
    console.print()


def render_greeting(console: Console, store: MemoryStore) -> None:
    """메모리 요약과 함께 인사말 출력."""
    profile = store.all_user_profile()
    visits = store.get_recent_visits(within_days=7)
    bits = []
    disliked = profile.get("disliked_categories")
    if disliked:
        bits.append(f"비선호: {', '.join(disliked)}")
    if profile.get("default_budget"):
        bits.append(f"기본 예산: {profile['default_budget']}")
    if visits:
        bits.append(f"최근 7일 방문 {len(visits)}건")
    summary = " · ".join(bits) if bits else "(메모리 비어있음 — scripts/seed_memory.py로 시드 가능)"

    console.print(
        f"[dim]현재 메모리:[/] {summary}"
    )
    console.print(
        "[bold cyan]🍽[/]  어떤 맛집을 찾으시나요? "
        "[dim](지역 + 음식 종류를 알려주시면 좋아요. 예: \"홍대 근처 디저트 카페\")[/]"
    )
    console.print()


def render_memory(store: MemoryStore) -> str:
    """현재 메모리 상태를 사람이 읽기 좋은 텍스트로 포맷."""
    profile = store.all_user_profile()
    visits = store.get_recent_visits(within_days=7)

    lines = ["[bold]User Profile[/]"]
    if profile:
        for k, v in profile.items():
            lines.append(f"  • {k}: {v}")
    else:
        lines.append("  [dim](비어있음 — scripts/seed_memory.py로 시드)[/]")

    lines.append("")
    lines.append("[bold]Recent Visits (7일 이내)[/]")
    if visits:
        for v in visits:
            lines.append(
                f"  • {v['name']} ({v.get('category') or '?'}) "
                f"[dim]@ {v['visited_at']}[/]"
            )
    else:
        lines.append("  [dim](없음)[/]")
    return "\n".join(lines)


def handle_command(
    cmd: str,
    store: MemoryStore,
    console: Console,
) -> bool:
    """슬래시 커맨드 처리. 종료해야 하면 False 반환."""
    global _SESSION_ALLOWED
    raw = cmd.strip()
    low = raw.lower()

    if low in ("/quit", "/exit", "/q"):
        return False
    if low in ("/help", "/?"):
        console.print(Panel(HELP_TEXT, title="[bold cyan]Help",
                            border_style="cyan"))
        return True
    if low == "/clear":
        console.clear()
        return True

    # /memory subcommands
    if low.startswith("/memory") or low == "/m":
        rest = raw[len("/memory"):].strip() if low.startswith("/memory") else ""
        if not rest or rest.lower() in ("", "show", "list"):
            console.print(Panel(render_memory(store),
                                title="[bold magenta]Memory",
                                border_style="magenta"))
            return True
        if rest.lower() == "clear":
            store.clear_all()
            console.print("[yellow]메모리(profile + visits)를 모두 비웠습니다.[/]")
            return True
        m = re.match(r"dislike\s+(add|remove)\s+(.+)", rest, re.IGNORECASE)
        if m:
            action, cat = m.group(1).lower(), m.group(2).strip()
            existing = store.get_user_profile("disliked_categories", default=[]) or []
            if action == "add":
                if cat not in existing:
                    existing.append(cat)
                    store.set_user_profile("disliked_categories", existing)
                    console.print(f"[green]비선호 추가:[/] {cat}")
                else:
                    console.print(f"[dim]이미 비선호:[/] {cat}")
            else:  # remove
                if cat in existing:
                    existing.remove(cat)
                    store.set_user_profile("disliked_categories", existing)
                    console.print(f"[green]비선호 제거:[/] {cat}")
                else:
                    console.print(f"[dim]비선호에 없음:[/] {cat}")
            return True
        console.print(f"[yellow]알 수 없는 /memory 하위 명령: {rest}. /help 입력[/]")
        return True

    # /allow subcommands
    if low.startswith("/allow"):
        rest = raw[len("/allow"):].strip()
        if not rest:
            if _SESSION_ALLOWED:
                console.print(f"[cyan]세션 허용 카테고리:[/] {sorted(_SESSION_ALLOWED)}")
            else:
                console.print("[dim]세션 허용 카테고리 없음[/]")
            return True
        if rest.lower() == "clear":
            _SESSION_ALLOWED.clear()
            console.print("[yellow]세션 허용 카테고리 초기화[/]")
            return True
        _SESSION_ALLOWED.add(rest)
        console.print(f"[green]이번 세션 허용:[/] {rest}")
        return True

    console.print(f"[yellow]알 수 없는 명령: {raw} — /help 입력[/]")
    return True


def _summarize_node(name: str, output: dict) -> str:
    """노드별 콤팩트 요약 — Dexter 스타일."""
    trace = output.get("trace_log", [])
    last = trace[-1] if trace else {}
    if name == "load_memory":
        return (
            f"profile={len(last.get('profile_keys', []))} keys, "
            f"recent={last.get('recent_count', 0)} visits"
        )
    if name == "planner":
        plan = last.get("plan", {})
        region = plan.get("region_query", "?")
        k = plan.get("post_filters", {}).get("k", "?")
        return f"plan(region=\"{region}\", k={k})"
    if name == "react_agent":
        return (
            f"{last.get('messages_count', 0)} messages, "
            f"{last.get('candidates_count', 0)} candidates"
        )
    if name == "aggregator":
        excluded = (
            last.get("excluded_by_category", 0)
            + last.get("excluded_by_recency", 0)
        )
        return (
            f"{last.get('raw_count', 0)} → {last.get('kept', 0)} kept, "
            f"{excluded} excluded"
        )
    if name == "reflector":
        ok = "[green]✓[/]" if last.get("passed") else "[yellow]↻[/]"
        return f"{ok} {last.get('reason', '')}"
    if name == "finalizer":
        return f"{last.get('k', 0)} recommendations"
    if name == "save_memory":
        return f"saved {last.get('saved', 0)} visits"
    return ""


def run_one_turn(graph, query: str, console: Console) -> None:
    """한 turn 실행: graph.stream → 노드 bullet → 최종 추천 (Markdown) → trace.md."""
    global _LAST_CLARIFICATION
    final_state: dict = {}
    last_traced_idx = 0
    initial_state = {"query": query, "session_allowed": list(_SESSION_ALLOWED)}

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for name, output in chunk.items():
                summary = _summarize_node(name, output)
                console.print(
                    f"[green]●[/] [bold]{name:<14}[/] [dim]{summary}[/]"
                )
                # Accumulate final state from streaming updates
                for k, v in output.items():
                    if k == "trace_log":
                        if isinstance(v, list):
                            final_state.setdefault("trace_log", []).extend(
                                v[last_traced_idx:]
                            )
                            last_traced_idx = len(v)
                        # else: ignore non-list trace_log values
                    else:
                        final_state[k] = v
    except KeyboardInterrupt:
        console.print("\n[yellow]요청이 취소되었습니다.[/]")
        return
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]오류:[/] {e}")
        return

    # Track clarification state for multi-turn flow
    plan = final_state.get("plan") or {}
    clarification = plan.get("clarification_needed") or []
    if clarification:
        _LAST_CLARIFICATION = {
            "original_query": query,
            "messages": clarification,
        }
    else:
        _LAST_CLARIFICATION = None

    final_text = final_state.get("final_text", "(no answer)")

    console.print()
    console.print(Panel(
        Markdown(final_text),
        title="[bold magenta]Recommendation[/]",
        border_style="magenta",
        padding=(0, 1),
    ))

    from src.ui.renderer import write_trace_md
    trace_path = write_trace_md(
        query=query,
        final_text=final_text,
        trace_log=final_state.get("trace_log", []),
    )
    console.print(f"[dim][trace] {trace_path}[/]")


def main() -> int:
    console = Console()

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY 가 .env에 없습니다.[/]")
        return 2
    if not os.getenv("KAKAO_REST_API_KEY"):
        console.print("[yellow]KAKAO_REST_API_KEY 누락 — geocode/kakao 검색 비활성[/]")

    render_welcome(console)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    store = MemoryStore(Path("data/agent_memory.db"))
    graph = build_graph(llm=llm, store=store)

    render_greeting(console, store)

    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=_SlashAutoSuggest(SLASH_COMMANDS),
        completer=_build_completer(),
        complete_while_typing=False,  # Tab키로만 트리거. ghost는 입력 즉시.
    )

    global _LAST_CLARIFICATION

    try:
        while True:
            try:
                # ANSI bold cyan ❯ + space. Rich markup is not supported by prompt_toolkit input.
                query = session.prompt(ANSI("\x1b[1;36m❯\x1b[0m ")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print("[dim]bye 👋[/]")
                break

            if not query:
                continue

            if query.startswith("/"):
                if not handle_command(query, store, console):
                    break
                continue

            # Multi-turn clarification: "no" path — cancel and accept new query
            if _LAST_CLARIFICATION and query.lower() in {"n", "no", "아니", "ㄴㄴ"}:
                console.print("[dim]→ 이전 충돌 해소. 새 요청 입력해주세요.[/]")
                _LAST_CLARIFICATION = None  # type: ignore[assignment]
                continue

            # Multi-turn clarification: "yes" path — auto-allow and re-run
            if _LAST_CLARIFICATION and query.lower() in {
                "y", "yes", "응", "허용", "ㅇㅇ", "그래", "ok"
            }:
                profile = store.all_user_profile()
                disliked = profile.get("disliked_categories", []) or []
                orig = _LAST_CLARIFICATION["original_query"].lower()
                matched = [d for d in disliked if d.lower() in orig]
                for cat in matched:
                    _SESSION_ALLOWED.add(cat)
                if matched:
                    console.print(f"[green]이번 세션 허용:[/] {', '.join(matched)}")
                # re-run with original query
                query = _LAST_CLARIFICATION["original_query"]
                _LAST_CLARIFICATION = None  # type: ignore[assignment]
                console.print(f"[dim]→ 원래 요청으로 재추천: {query}[/]")

            console.print()  # blank line before tool output
            run_one_turn(graph, query, console)
            console.print()  # blank line before next prompt
    finally:
        flush_telemetry()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
