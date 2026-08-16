from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


SUPPORTED_REQUEST_RULE_FIELDS = frozenset(
    {
        "endpoint",
        "request_model",
        "upstream_model",
        "reasoning_effort",
        "request_type",
    }
)


def _normalize_string(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_endpoint(value: Any) -> str:
    endpoint = str(value or "").strip().rstrip("/")
    if endpoint and not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return endpoint.lower()


def request_reasoning_effort(request_body: Any) -> Optional[str]:
    if isinstance(request_body, Mapping):
        effort = request_body.get("reasoning_effort")
        reasoning = request_body.get("reasoning")
    else:
        effort = getattr(request_body, "reasoning_effort", None)
        reasoning = getattr(request_body, "reasoning", None)

    if not effort:
        if isinstance(reasoning, Mapping):
            effort = reasoning.get("effort")
        elif reasoning is not None:
            effort = getattr(reasoning, "effort", None)

    normalized = _normalize_string(effort)
    return normalized or None


def _rule_values(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def provider_exclude_request_rules(
    provider: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    rules = list(_rule_values(provider.get("exclude_request_rules")))
    preferences = provider.get("preferences")
    if isinstance(preferences, Mapping):
        rules.extend(_rule_values(preferences.get("exclude_request_rules")))
    return tuple(rules)


def _value_matches(expected: Any, actual: Optional[str], *, endpoint: bool = False) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(_value_matches(item, actual, endpoint=endpoint) for item in expected)
    if actual is None or not isinstance(expected, str):
        return False

    normalize = _normalize_endpoint if endpoint else _normalize_string
    normalized_expected = normalize(expected)
    normalized_actual = normalize(actual)
    if not normalized_expected or not normalized_actual:
        return False
    if normalized_expected == "*":
        return True
    if normalized_expected.endswith("*"):
        return normalized_actual.startswith(normalized_expected[:-1])
    return normalized_expected == normalized_actual


def exclude_request_rule_matches(
    rule: Mapping[str, Any],
    *,
    endpoint: Optional[str],
    request_model: str,
    upstream_model: str,
    reasoning_effort: Optional[str],
    request_type: Optional[str],
) -> bool:
    match = rule.get("match")
    if not isinstance(match, Mapping) or not match:
        return False
    if any(key not in SUPPORTED_REQUEST_RULE_FIELDS for key in match):
        return False

    context = {
        "endpoint": endpoint,
        "request_model": request_model,
        "upstream_model": upstream_model,
        "reasoning_effort": reasoning_effort,
        "request_type": request_type,
    }
    return all(
        _value_matches(
            expected,
            context[key],
            endpoint=key == "endpoint",
        )
        for key, expected in match.items()
    )


def provider_accepts_request_rules(
    provider: Mapping[str, Any],
    *,
    endpoint: Optional[str],
    request_model: str,
    reasoning_effort: Optional[str],
    request_type: Optional[str],
) -> bool:
    model_dict = provider.get("_model_dict_cache")
    upstream_model = (
        str(model_dict.get(request_model) or request_model)
        if isinstance(model_dict, Mapping)
        else request_model
    )
    return not any(
        exclude_request_rule_matches(
            rule,
            endpoint=endpoint,
            request_model=request_model,
            upstream_model=upstream_model,
            reasoning_effort=reasoning_effort,
            request_type=request_type,
        )
        for rule in provider_exclude_request_rules(provider)
    )
