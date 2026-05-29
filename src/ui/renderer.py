from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def announce_query(query: str) -> None:
    console.print(Rule(style="cyan"))
    console.print(Panel(query, title="[bold cyan]User Query", border_style="cyan"))


def log_node(name: str, summary: str) -> None:
    console.print(f"[green]▸[/] [bold]{name}[/]  {summary}")


def announce_recommendation(text: str) -> None:
    console.print(Rule(style="magenta"))
    console.print(Panel(text, title="[bold magenta]Recommendation",
                        border_style="magenta"))


def announce_trace_url(url: str | None) -> None:
    if url:
        console.print(f"[dim][phoenix] trace: {url}[/]")


def render_trace_md(*, query: str, final_text: str,
                    trace_log: list[dict[str, Any]]) -> str:
    """제출용 trace.md 본문 생성."""
    lines = [
        "# Execution Trace",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## User Query",
        "",
        f"> {query}",
        "",
        "## Node Trace",
        "",
        "| # | Node | Summary |",
        "|--|--|--|",
    ]
    for i, entry in enumerate(trace_log, 1):
        node = entry.get("node", "?")
        summary = {k: v for k, v in entry.items() if k != "node"}
        lines.append(f"| {i} | `{node}` | `{json.dumps(summary, ensure_ascii=False)}` |")

    lines += ["", "## Final", "", final_text, ""]
    return "\n".join(lines)


def write_trace_md(*, query: str, final_text: str,
                   trace_log: list[dict[str, Any]],
                   out_dir: Path = Path("docs/traces")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"trace_{ts}.md"
    path.write_text(
        render_trace_md(query=query, final_text=final_text, trace_log=trace_log),
        encoding="utf-8",
    )
    return path
