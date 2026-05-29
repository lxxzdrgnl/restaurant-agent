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


def _format_messages(messages: list[Any]) -> list[str]:
    """react_agent의 messages 배열을 ReAct trace 형식으로 마크다운 변환.
    Agent 판단(Thought), 도구 호출(Action), 결과(Observation)를 명시적으로 표기."""
    lines: list[str] = []
    turn = 0
    for m in messages or []:
        # LangChain 메시지 객체와 dict 둘 다 지원
        msg_type = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
        if msg_type == "ai":
            data = getattr(m, "data", None) or m
            content = (data.get("content") if isinstance(data, dict) else getattr(m, "content", "")) or ""
            tool_calls = (data.get("tool_calls") if isinstance(data, dict)
                          else getattr(m, "tool_calls", None)) or []
            if tool_calls:
                turn += 1
                lines.append(f"### Turn {turn} — 🤖 Agent decision")
                if content.strip():
                    lines.append(f"> {content.strip()}")
                for tc in tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    lines.append(f"- **Action** `{name}`")
                    lines.append(f"  ```json")
                    lines.append(f"  {json.dumps(args, ensure_ascii=False)}")
                    lines.append(f"  ```")
            elif content.strip():
                # final assistant text (no tool calls)
                lines.append(f"### Turn {turn + 1} — 🤖 Agent: `{content.strip()[:60]}`")
        elif msg_type == "tool":
            data = getattr(m, "data", None) or m
            name = (data.get("name") if isinstance(data, dict) else getattr(m, "name", "?"))
            raw = (data.get("content") if isinstance(data, dict) else getattr(m, "content", ""))
            try:
                obj = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(obj, dict) and "results" in obj:
                    n = obj.get("count", len(obj.get("results", [])))
                    samples = [r.get("name", "?") for r in obj["results"][:3]]
                    lines.append(f"- **Observation** `{name}` → {n} results "
                                 f"(sample: {', '.join(samples)}{', ...' if n > 3 else ''})")
                elif isinstance(obj, dict) and "lat" in obj:
                    lines.append(f"- **Observation** `{name}` → "
                                 f"lat={obj['lat']:.5f}, lng={obj['lng']:.5f}, "
                                 f"matched=\"{obj.get('matched_name', '')}\"")
                elif isinstance(obj, dict) and "error" in obj:
                    lines.append(f"- **Observation** `{name}` → ❌ error: {obj['error']}")
                else:
                    lines.append(f"- **Observation** `{name}` → {str(obj)[:120]}")
            except (json.JSONDecodeError, TypeError):
                lines.append(f"- **Observation** `{name}` → {str(raw)[:120]}")
    return lines


def render_trace_md(*, query: str, final_text: str,
                    trace_log: list[dict[str, Any]],
                    messages: list[Any] | None = None,
                    plan: dict | None = None,
                    candidates: list[dict] | None = None,
                    aggregated: list[dict] | None = None) -> str:
    """제출용 trace.md 본문 생성. ReAct trace 전체를 사람이 읽을 수 있게 직렬화."""
    lines = [
        "# Execution Trace",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. User Query",
        "",
        f"> {query}",
        "",
        "## 2. Node 진행 요약",
        "",
        "| # | Node | Summary |",
        "|--|--|--|",
    ]
    for i, entry in enumerate(trace_log, 1):
        node = entry.get("node", "?")
        summary = {k: v for k, v in entry.items() if k != "node"}
        lines.append(f"| {i} | `{node}` | `{json.dumps(summary, ensure_ascii=False)}` |")

    # Plan (planner LLM이 만든 JSON)
    if plan:
        lines += [
            "",
            "## 3. Planner 출력 (Plan-and-Solve)",
            "",
            "```json",
            json.dumps(plan, ensure_ascii=False, indent=2),
            "```",
        ]

    # ReAct loop trace
    if messages:
        lines += [
            "",
            "## 4. ReAct Loop — Thought / Action / Observation",
            "",
            "agent가 도구를 어떻게 선택하고 결과를 어떻게 받았는지의 ReAct 사이클입니다.",
            "",
        ]
        lines.extend(_format_messages(messages))

    # Candidates (도구 결과 합친 풀)
    if candidates is not None:
        lines += [
            "",
            f"## 5. 후보 풀 (raw candidates: {len(candidates)}건)",
            "",
            "도구 호출 결과를 합친 raw 후보 (dedup 전):",
            "",
            "| # | Name | Source | Category | Rating | Reviews | Price | Address |",
            "|--|--|--|--|--|--|--|--|",
        ]
        for i, c in enumerate(candidates[:30], 1):  # 너무 많으면 30개까지만
            lines.append(
                f"| {i} | {c.get('name', '?')} | {c.get('source', '?')} | "
                f"{c.get('category', '-')} | {c.get('rating', '-')} | "
                f"{c.get('review_count', '-')} | {c.get('price_level', '-')} | "
                f"{(c.get('address') or '-')[:50]} |"
            )
        if len(candidates) > 30:
            lines.append(f"| ... | (+{len(candidates) - 30}건 생략) | | | | | | |")

    # Aggregated (점수+필터 후 top-k)
    if aggregated is not None:
        lines += [
            "",
            f"## 6. Aggregated (dedup + score + filter 후 top {len(aggregated)})",
            "",
            "| # | Name | Score | Sources | Rating | Reviews | Price | Menu items |",
            "|--|--|--|--|--|--|--|--|",
        ]
        for i, a in enumerate(aggregated, 1):
            items = a.get("menu_items") or []
            if items:
                menu_str = ", ".join(f"{m['name']}({m['price']})"
                                       for m in items[:3])
                if len(items) > 3:
                    menu_str += f" +{len(items)-3}"
            else:
                menu_str = "-"
            lines.append(
                f"| {i} | {a.get('name', '?')} | "
                f"{a.get('score', 0):.4f} | {a.get('source_count', 1)} | "
                f"{a.get('rating', '-')} | {a.get('review_count', '-')} | "
                f"{a.get('price_level', '-')} | {menu_str} |"
            )

    # Final 추천
    lines += ["", "## 7. Final Recommendation", "", final_text, ""]
    return "\n".join(lines)


def write_trace_md(*, query: str, final_text: str,
                   trace_log: list[dict[str, Any]],
                   messages: list[Any] | None = None,
                   plan: dict | None = None,
                   candidates: list[dict] | None = None,
                   aggregated: list[dict] | None = None,
                   out_dir: Path = Path("docs/traces")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"trace_{ts}.md"
    path.write_text(
        render_trace_md(
            query=query, final_text=final_text, trace_log=trace_log,
            messages=messages, plan=plan,
            candidates=candidates, aggregated=aggregated,
        ),
        encoding="utf-8",
    )
    return path
