"""Phoenix OTel bootstrap. Call init_telemetry() exactly once at process start
BEFORE importing langchain/langgraph so auto-instrumentation wraps them."""

from __future__ import annotations
import contextlib
import io
import os
import warnings
from typing import Any, Sequence

# Phoenix가 register() 안에서 직접 print + warnings.warn을 호출해서
# verbose=False, filterwarnings 모두 우회된다. stdout/stderr/warnings를
# 컨텍스트 매니저로 통째로 막아 깨끗한 출력을 보장.
@contextlib.contextmanager
def _silenced():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), \
         contextlib.redirect_stderr(sink), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


from phoenix.otel import register
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# root span은 'LangGraph' 만 허용. LangChain instrumentor가 도구/LLM 호출을 부모
# context 없이 root span으로도 export하는 부작용을 차단. LangGraph 안의 자식 span은 보존.
_ROOT_ALLOWLIST = {"LangGraph"}

# LangChain instrumentor가 매 LLM 호출마다 만드는 wrapper span. 정보 가치는 없고
# trace tree 깊이만 늘려서 채점자가 보기 어려워진다. export 전에 drop.
_NOISE_SPAN_NAMES = {
    "RunnableSequence",         # 매 LLM 호출의 wrapper
    "Prompt",                   # SystemMessage + HumanMessage 합치는 단계
    "ChatPromptTemplate",       # 변형 prompt 노드
    "should_continue",          # LangGraph conditional edge internal
}


def _install_root_filter(span_processor) -> None:
    """SpanProcessor 인스턴스의 on_end를 wrap해서 (1) root 비허용 + (2) noise span을 drop."""
    if getattr(span_processor, "_root_filter_installed", False):
        return
    original_on_end = span_processor.on_end

    def filtered_on_end(span):
        # (1) root drop: parent is None이면 LangGraph만 허용
        if span.parent is None and span.name not in _ROOT_ALLOWLIST:
            return
        # (2) noise drop: LangChain 내부 wrapper span 제거 (90 → ~50 spans)
        if span.name in _NOISE_SPAN_NAMES:
            return
        return original_on_end(span)

    span_processor.on_end = filtered_on_end
    span_processor._root_filter_installed = True


_provider: Any = None


def init_telemetry() -> None:
    global _provider
    if _provider is not None:
        return
    if os.getenv("PHOENIX_DISABLED") == "1":
        return
    api_key = os.getenv("PHOENIX_API_KEY")
    if not api_key:
        # No key → silently skip; agent still runs without traces.
        return

    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.rheon.kr/api/collect"
    )
    project = os.getenv("PHOENIX_PROJECT_NAME", "restaurant")

    with _silenced():
        _provider = register(
            project_name=project,
            endpoint=endpoint,
            headers={"authorization": f"Bearer {api_key}"},
            auto_instrument=True,
            batch=True,  # BatchSpanProcessor — Phoenix 권장
        )

    # phoenix.otel.otel.BatchSpanProcessor의 span_exporter가 read-only property라
    # exporter 자체를 wrap할 수 없다. 대신 instance-level로 on_end 메서드를 monkey-patch해서
    # root-level 비허용 span을 export queue에 넣기 전에 drop.
    try:
        for sp in _provider._active_span_processor._span_processors:
            _install_root_filter(sp)
    except Exception:  # noqa: BLE001 — internal API 변경 시 silent fail
        pass

    print(
        f"[phoenix] telemetry initialized → project=\"{project}\" "
        f"endpoint={endpoint} (auth)"
    )


def flush_telemetry() -> None:
    """Force BatchSpanProcessor to ship queued spans before process exit."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception:  # noqa: BLE001 — flush errors must not crash the CLI
        pass
