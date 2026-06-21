from __future__ import annotations

from typing import Any


def merge_timing_spans(current_info: dict[str, Any], spans: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(current_info, dict):
        return {}

    merged: dict[str, Any] = {}
    existing = current_info.get("timing_spans")
    if isinstance(existing, dict):
        for key, value in existing.items():
            name = str(key or "").strip()
            if name:
                merged[name] = value

    if isinstance(spans, dict):
        for key, value in spans.items():
            name = str(key or "").strip()
            if name:
                merged[name] = value

    current_info["timing_spans"] = merged
    return merged
