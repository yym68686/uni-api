from __future__ import annotations

from typing import Any

from core.utils import safe_get


def apply_provider_preference_headers(
    headers: dict[str, Any],
    provider: dict[str, Any],
    *,
    http_request: Any | None = None,
) -> None:
    headers.update(safe_get(provider, "preferences", "headers", default={}) or {})
    apply_passthrough_request_headers(headers, provider, http_request=http_request)


def apply_passthrough_request_headers(
    headers: dict[str, Any],
    provider: dict[str, Any],
    *,
    http_request: Any | None = None,
) -> None:
    passthrough_names = _passthrough_header_names(provider)
    if not passthrough_names:
        return

    request_headers = getattr(http_request, "headers", None) if http_request is not None else None
    for header_name in passthrough_names:
        _remove_header_case_insensitive(headers, header_name)
        value = _get_header_case_insensitive(request_headers, header_name)
        if value is not None and str(value) != "":
            headers[header_name] = str(value)


def _passthrough_header_names(provider: dict[str, Any]) -> list[str]:
    configured = safe_get(provider, "preferences", "passthrough_request_headers", default=[]) or []
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for raw_name in configured:
        name = str(raw_name or "").strip()
        if not name:
            continue
        normalized = name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        names.append(name)
    return names


def _remove_header_case_insensitive(headers: dict[str, Any], header_name: str) -> None:
    normalized = header_name.lower()
    for existing_name in list(headers.keys()):
        if str(existing_name).lower() == normalized:
            headers.pop(existing_name, None)


def _get_header_case_insensitive(headers: Any, header_name: str) -> Any | None:
    if headers is None:
        return None

    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(header_name)
        if value is not None:
            return value

    if isinstance(headers, dict):
        normalized = header_name.lower()
        for existing_name, value in headers.items():
            if str(existing_name).lower() == normalized:
                return value
    return None
