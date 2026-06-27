from __future__ import annotations

import copy

from core.utils import get_model_dict, safe_get


def _path_parts(path: object) -> list[str]:
    if not isinstance(path, str):
        return []
    return [part for part in (part.strip() for part in path.split(".")) if part]


def _get_path(value: object, path: str) -> tuple[object, bool]:
    current = value
    for part in _path_parts(path):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _delete_path(value: dict, path: str) -> None:
    parts = _path_parts(path)
    if not parts:
        return
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        current = child
    current.pop(parts[-1], None)


def _set_path(value: dict, path: str, replacement: object) -> None:
    parts = _path_parts(path)
    if not parts:
        return
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        current = child
    current[parts[-1]] = replacement


def _value_matches_expected(actual: object, expected: object) -> bool:
    return actual == expected


def _matches_condition(value: object, condition: object) -> bool:
    if isinstance(condition, str):
        return value == condition
    if isinstance(condition, dict):
        for key, expected in condition.items():
            actual, exists = _get_path(value, str(key))
            if not exists or not _value_matches_expected(actual, expected):
                return False
        return True
    return False


def _matches_any_condition(value: object, conditions: object) -> bool:
    if isinstance(conditions, (list, tuple, set)):
        return any(_matches_condition(value, condition) for condition in conditions)
    return _matches_condition(value, conditions)


def _apply_structured_removal(payload: dict, rule: dict) -> None:
    path = rule.get("path")
    if not isinstance(path, str) or not path.strip():
        return

    target, exists = _get_path(payload, path)
    if not exists:
        return

    has_where = "where" in rule
    has_where_any = "where_any" in rule

    if not has_where and not has_where_any:
        _delete_path(payload, path)
        return

    def should_remove(value: object) -> bool:
        return (
            (has_where and _matches_condition(value, rule.get("where")))
            or (has_where_any and _matches_any_condition(value, rule.get("where_any")))
        )

    if isinstance(target, list):
        filtered = [item for item in target if not should_remove(item)]
        if len(filtered) == len(target):
            return
        if filtered or not rule.get("drop_empty", False):
            _set_path(payload, path, filtered)
        else:
            _delete_path(payload, path)
        return

    if should_remove(target):
        _delete_path(payload, path)


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

    def normalize_removals(value) -> list[str | dict]:
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, (list, tuple, set)):
            removals: list[str | dict] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        removals.append(cleaned)
                elif isinstance(item, dict):
                    removals.append(item)
            return removals
        if isinstance(value, dict):
            return [value]
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
        for removal in normalize_removals(override.get(removal_key)):
            if isinstance(removal, str):
                key = removal
            elif isinstance(removal, dict):
                key = removal.get("path")
            else:
                continue
            if key in skipped:
                continue
            if isinstance(removal, dict):
                _apply_structured_removal(payload, removal)
            else:
                payload.pop(key, None)

    apply_override_values(overrides, skip_model_override_keys=True)
    apply_removals(overrides)

    model_overrides = overrides.get(request_model)
    if isinstance(model_overrides, dict):
        apply_override_values(model_overrides)
        apply_removals(model_overrides)

    return payload
