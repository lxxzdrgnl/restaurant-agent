"""맛집 추천 에이전트 CLI.

사용:
    uv run python main.py "전주 객사 근처에서 친구랑 저녁..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Phoenix 트레이스는 LangChain import 전에 초기화해야 자동 instrumentation이 잡힘.
from src.observability.telemetry import init_telemetry, flush_telemetry  # noqa: E402
init_telemetry()

from langchain_openai import ChatOpenAI  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402
from src.ui.renderer import (  # noqa: E402
    announce_query,
    announce_recommendation,
    announce_trace_url,
    log_node,
    write_trace_md,
)


DEFAULT_QUERY = (
    "전주 객사 근처에서 친구랑 저녁 먹기 좋은 맛집을 찾아줘. "
    "너무 비싸지 않고, 리뷰가 좋은 곳 위주로 3곳 추천해줘."
)


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY 가 .env에 없습니다.", file=sys.stderr)
        return 2
    if not os.getenv("KAKAO_REST_API_KEY"):
        print("[warn] KAKAO_REST_API_KEY 누락 — geocode/kakao 검색이 비활성됩니다.",
              file=sys.stderr)

    announce_query(query)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    store = MemoryStore(Path("data/agent_memory.db"))
    graph = build_graph(llm=llm, store=store)

    try:
        result = graph.invoke({"query": query})
    except Exception as e:  # noqa: BLE001
        print(f"[error] graph invoke failed: {e}", file=sys.stderr)
        flush_telemetry()
        return 1

    # 콘솔에 노드별 한 줄 요약
    for entry in result.get("trace_log", []):
        node = entry.get("node", "?")
        rest = {k: v for k, v in entry.items() if k != "node"}
        log_node(node, str(rest))

    final_text = result.get("final_text", "(no answer)")
    announce_recommendation(final_text)

    trace_path = write_trace_md(
        query=query, final_text=final_text,
        trace_log=result.get("trace_log", []),
    )
    print(f"[trace] saved: {trace_path}")

    project = os.getenv("PHOENIX_PROJECT_NAME", "restaurant")
    base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").replace("/api/collect", "")
    if base and os.getenv("PHOENIX_API_KEY"):
        announce_trace_url(f"{base}/projects/{project}")

    flush_telemetry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
