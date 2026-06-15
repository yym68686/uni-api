from __future__ import annotations

import copy

from core.utils import get_model_dict, safe_get


def _merge_post_body_override_dict(target: dict, override: dict, *, removal_key: str = "__remove__") -> None:
    for key, value in override.items():
        if key == removal_key:
            continue
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_post_body_override_dict(current, value, removal_key=removal_key)
        else:
            target[key] = copy.deepcopy(value)


def apply_post_body_parameter_overrides(
    payload: dict,
    provider: dict,
    request_model: str,
    *,
    skip_keys: set[str] | None = None,
) -> dict:
    removal_key = "__remove__"

    def normalize_removals(value) -> list[str]:
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, (list, tuple, set)):
            removals: list[str] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        removals.append(cleaned)
            return removals
        return []

    overrides = safe_get(provider, "preferences", "post_body_parameter_overrides", default={}) or {}
    if not isinstance(overrides, dict):
        return payload

    model_dict = provider.get("_model_dict_cache")
    if not isinstance(model_dict, dict):
        model_dict = get_model_dict(provider)

    skipped = skip_keys or set()

    def apply_override_values(override: dict, *, skip_model_override_keys: bool = False) -> None:
        for key, value in override.items():
            if (
                key in skipped
                or key == removal_key
                or (skip_model_override_keys and key in model_dict)
            ):
                continue
            current = payload.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                _merge_post_body_override_dict(current, value, removal_key=removal_key)
            else:
                payload[key] = copy.deepcopy(value)

    def apply_removals(override: dict) -> None:
        for key in normalize_removals(override.get(removal_key)):
            if key in skipped:
                continue
            payload.pop(key, None)

    apply_override_values(overrides, skip_model_override_keys=True)
    apply_removals(overrides)

    model_overrides = overrides.get(request_model)
    if isinstance(model_overrides, dict):
        apply_override_values(model_overrides)
        apply_removals(model_overrides)

    return payload
