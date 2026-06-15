from __future__ import annotations

from typing import Any

from fugue_observability import emit_uni_api_ember_request_observability


def emit_request_observability(current_info: dict[str, Any], runtime_metrics: dict[str, Any]) -> None:
    emit_uni_api_ember_request_observability(
        current_info=current_info,
        runtime_metrics=runtime_metrics,
    )

