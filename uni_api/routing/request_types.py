from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


COMPACTION_REQUEST_TYPE = "compaction"
COMPACTION_TRIGGER_INPUT_TYPE = "compaction_trigger"


def normalize_request_type(value: Any) -> str:
    return str(value or "").strip().lower()


def request_type_values(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set, frozenset)):
        values = (values,)
    normalized = []
    for item in values:
        request_type = normalize_request_type(item)
        if request_type and request_type not in normalized:
            normalized.append(request_type)
    return tuple(normalized)


def provider_request_type_values(provider: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = []
    values.extend(request_type_values(provider.get(key)))
    preferences = provider.get("preferences")
    if isinstance(preferences, Mapping):
        values.extend(request_type_values(preferences.get(key)))
    return tuple(dict.fromkeys(values))


def provider_accepts_request_type(
    provider: Mapping[str, Any],
    request_type: Optional[str],
) -> bool:
    normalized_request_type = normalize_request_type(request_type)
    only_request_types = provider_request_type_values(provider, "only_request_types")
    if only_request_types and normalized_request_type not in only_request_types:
        return False

    exclude_request_types = provider_request_type_values(provider, "exclude_request_types")
    return not (
        normalized_request_type
        and normalized_request_type in exclude_request_types
    )


def detect_request_type(endpoint: Optional[str], request_body: Any) -> Optional[str]:
    normalized_endpoint = str(endpoint or "").strip().rstrip("/")
    if normalized_endpoint.endswith("/responses/compact"):
        return COMPACTION_REQUEST_TYPE

    if isinstance(request_body, Mapping):
        request_input = request_body.get("input")
    else:
        request_input = getattr(request_body, "input", None)
    if not isinstance(request_input, list):
        return None

    for item in request_input:
        if (
            isinstance(item, Mapping)
            and item.get("type") == COMPACTION_TRIGGER_INPUT_TYPE
        ):
            return COMPACTION_REQUEST_TYPE
    return None
