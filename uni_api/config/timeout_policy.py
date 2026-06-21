from __future__ import annotations

from typing import Any, Optional

from core.utils import safe_get


def init_timeout_policy(all_config: dict[str, Any]) -> dict[str, Any]:
    preferences = safe_get(all_config, "preferences", default={}) or {}
    result: dict[str, Any] = {
        "global": _normalize_timeout_policy(preferences.get("timeout_policy")),
        "providers": {},
    }
    for provider in safe_get(all_config, "providers", default=[]) or []:
        provider_name = str(provider.get("provider") or "").strip()
        if not provider_name:
            continue
        policy = _normalize_timeout_policy(safe_get(provider, "preferences", "timeout_policy", default=None))
        if policy["default"] or policy["rules"]:
            result["providers"][provider_name] = policy
    return result


def resolve_timeout_policy(
    timeout_policy: dict[str, Any],
    *,
    provider_name: str,
    endpoint: str,
    method: str,
    stream: bool,
    engine: str,
    original_model: str,
    request_model: str,
    role: Optional[str] = None,
) -> dict[str, Any]:
    context = {
        "provider": str(provider_name or ""),
        "endpoint": str(endpoint or ""),
        "method": str(method or "POST").upper(),
        "stream": bool(stream),
        "engine": str(engine or ""),
        "model": str(request_model or ""),
        "request_model": str(request_model or ""),
        "upstream_model": str(original_model or ""),
        "role": str(role or ""),
    }
    merged: dict[str, int] = {}
    sources: list[str] = []

    global_policy = _policy_section(timeout_policy, "global")
    if global_policy["default"]:
        merged.update(global_policy["default"])
        sources.append("global.default")
    provider_policy = _policy_section(timeout_policy, "providers", provider_name)
    if provider_policy["default"]:
        merged.update(provider_policy["default"])
        sources.append("provider.default")
    for source, policy in (
        ("global", global_policy),
        ("provider", provider_policy),
    ):
        best = _best_timeout_policy_rule(policy.get("rules") or [], context)
        if best is None:
            continue
        merged.update(best["timeout"])
        sources.append(f"{source}.rules[{best['index']}]")

    return {"timeout": merged, "sources": sources, "context": context}


def apply_timeout_policy(
    *,
    base_timeout: int,
    timeout_policy: dict[str, Any],
    provider_name: str,
    endpoint: str,
    stream: bool,
    engine: str,
    original_model: str,
    request_model: str,
    method: str = "POST",
    role: Optional[str] = None,
) -> dict[str, Any]:
    resolved = resolve_timeout_policy(
        timeout_policy,
        provider_name=provider_name,
        endpoint=endpoint,
        method=method,
        stream=stream,
        engine=engine,
        original_model=original_model,
        request_model=request_model,
        role=role,
    )
    timeout_values = dict(resolved.get("timeout") or {})
    selected = int(base_timeout)
    if timeout_values.get("first_byte") is not None:
        selected = int(timeout_values["first_byte"])
    elif timeout_values.get("total") is not None:
        selected = int(timeout_values["total"])
    elif timeout_values.get("idle") is not None:
        selected = int(timeout_values["idle"])
    adjusted_from = int(base_timeout) if selected != int(base_timeout) else None
    first_byte_timeout = (
        int(timeout_values["first_byte"])
        if timeout_values.get("first_byte") is not None
        else int(base_timeout)
    )
    return {
        "timeout_value": selected,
        "first_byte_timeout": first_byte_timeout,
        "idle_timeout": timeout_values.get("idle"),
        "total_timeout": timeout_values.get("total"),
        "timeout_policy": timeout_values,
        "timeout_policy_sources": list(resolved.get("sources") or []),
        "timeout_adjusted_from": adjusted_from,
    }


def _normalize_timeout_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {"default": {}, "rules": []}
    default = _normalize_timeout_values(policy.get("default"))
    rules = []
    for item in policy.get("rules") or []:
        if not isinstance(item, dict):
            continue
        match = item.get("match") or {}
        timeout = _normalize_timeout_values(item.get("timeout"))
        if not isinstance(match, dict) or not timeout:
            continue
        rules.append({"match": dict(match), "timeout": timeout})
    return {"default": default, "rules": rules}


def _normalize_timeout_values(raw: Any) -> dict[str, int]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return {"first_byte": max(0, int(raw))}
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for key in ("connect", "write", "pool", "first_byte", "idle", "total"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _policy_section(policy: dict[str, Any], *path: str) -> dict[str, Any]:
    current: Any = policy or {}
    for part in path:
        if not isinstance(current, dict):
            return {"default": {}, "rules": []}
        current = current.get(part, {})
    if isinstance(current, dict):
        return {
            "default": dict(current.get("default") or {}),
            "rules": list(current.get("rules") or []),
        }
    return {"default": {}, "rules": []}


def _best_timeout_policy_rule(rules: list[dict[str, Any]], context: dict[str, Any]) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    best_score = -1
    for index, rule in enumerate(rules):
        match = rule.get("match") or {}
        if not isinstance(match, dict) or not _timeout_policy_rule_matches(match, context):
            continue
        score = len([key for key in match if match.get(key) is not None])
        if score > best_score:
            best = {"index": index, "timeout": dict(rule.get("timeout") or {})}
            best_score = score
    return best


def _timeout_policy_rule_matches(match: dict[str, Any], context: dict[str, Any]) -> bool:
    aliases = {
        "model": ("model", "request_model", "upstream_model"),
        "request_model": ("request_model",),
        "upstream_model": ("upstream_model",),
        "provider": ("provider",),
        "endpoint": ("endpoint",),
        "method": ("method",),
        "stream": ("stream",),
        "engine": ("engine",),
        "role": ("role",),
    }
    for key, expected in match.items():
        if expected is None:
            continue
        context_keys = aliases.get(str(key), (str(key),))
        if not any(_timeout_policy_value_matches(expected, context.get(context_key)) for context_key in context_keys):
            return False
    return True


def _timeout_policy_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(_timeout_policy_value_matches(item, actual) for item in expected)
    if isinstance(expected, bool):
        return bool(actual) is expected
    expected_str = str(expected).strip()
    actual_str = str(actual or "").strip()
    if not expected_str:
        return False
    if expected_str == "*":
        return True
    if expected_str.endswith("*"):
        return actual_str.lower().startswith(expected_str[:-1].lower())
    return expected_str.lower() == actual_str.lower()
