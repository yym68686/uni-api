from __future__ import annotations

import re
from hashlib import sha256
from dataclasses import dataclass
from typing import Any, Iterable


_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9]+_([A-Za-z0-9]+)$")
_MAX_ITEM_ID_LENGTH = 64
_HASHED_ITEM_ID_HEX_LENGTH = 40
_ITEM_ID_PREFIX_BY_TYPE = {
    "message": "msg",
    "reasoning": "rs",
    "function_call": "fc",
    "function_call_output": "fco",
    "tool_search_call": "tsc",
    "custom_tool_call": "ctc",
    "custom_tool_call_output": "ctco",
}
_ITEM_ID_PREFIX_BY_EVENT = {
    "response.output_text.delta": "msg",
    "response.output_text.done": "msg",
    "response.reasoning_summary_text.delta": "rs",
    "response.reasoning_summary_text.done": "rs",
    "response.reasoning_text.delta": "rs",
    "response.reasoning_text.done": "rs",
    "response.function_call_arguments.delta": "fc",
    "response.function_call_arguments.done": "fc",
    "response.custom_tool_call_input.delta": "ctc",
    "response.custom_tool_call_input.done": "ctc",
}
_MAX_RECORDED_PATHS = 16
_DEFAULT_NORMALIZATION_PROVIDERS = frozenset({"fugue-codex", "937auth", "937auth01"})


class ResponsesCustomToolCallIdCollisionError(ValueError):
    """Raised when prefix normalization would create a duplicate item ID."""


@dataclass(frozen=True, slots=True)
class ResponsesCustomToolCallIdNormalizationResult:
    normalized_ids: int = 0
    rewritten_references: int = 0
    paths: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.normalized_ids or self.rewritten_references)


def responses_custom_tool_call_id_normalization_enabled(
    provider: dict[str, Any],
    models: Iterable[str],
) -> bool:
    requested_models = {str(model).strip() for model in models if str(model).strip()}
    preferences = provider.get("preferences") or {}
    if not isinstance(preferences, dict):
        return False
    if "normalize_responses_custom_tool_call_ids" not in preferences:
        return (
            str(provider.get("engine") or "").strip().lower() == "codex"
            or str(provider.get("provider") or "")
            in _DEFAULT_NORMALIZATION_PROVIDERS
        )

    configured = preferences["normalize_responses_custom_tool_call_ids"]
    if configured is True:
        return True
    if not isinstance(configured, (list, tuple, set, frozenset)):
        return False
    enabled_models = {str(model).strip() for model in configured if str(model).strip()}
    return "*" in enabled_models or bool(requested_models.intersection(enabled_models))


class ResponsesCustomToolCallIdNormalizer:
    """Normalize non-canonical Responses item IDs without touching content."""

    def __init__(self) -> None:
        self._id_map: dict[str, str] = {}
        self._seen_item_ids: set[str] = set()

    def normalize(self, payload: Any) -> ResponsesCustomToolCallIdNormalizationResult:
        if not isinstance(payload, dict):
            return ResponsesCustomToolCallIdNormalizationResult()

        item_locations = self._item_locations(payload)
        existing_ids = {
            item_id
            for _, item in item_locations
            if isinstance((item_id := item.get("id")), str)
        }
        candidates: list[tuple[str, dict[str, Any], str, str]] = []
        candidate_targets: dict[str, str] = {}
        candidate_sources: dict[str, str] = {}

        for path, item in item_locations:
            item_id = item.get("id")
            normalized_id = self._normalized_id(item_id, item.get("type"))
            if normalized_id is None:
                continue
            if self._would_collide(item_id, normalized_id, existing_ids):
                raise ResponsesCustomToolCallIdCollisionError(
                    f"Responses item ID normalization at {path}.id would collide with an existing item ID"
                )
            target_owner = candidate_targets.get(normalized_id)
            if target_owner is not None and target_owner != item_id:
                raise ResponsesCustomToolCallIdCollisionError(
                    f"Responses item ID normalization at {path}.id would collide with another normalized item ID"
                )
            source_target = candidate_sources.get(item_id)
            if source_target is not None and source_target != normalized_id:
                raise ResponsesCustomToolCallIdCollisionError(
                    f"Responses item ID normalization at {path}.id produced an inconsistent mapping"
                )
            candidate_sources[item_id] = normalized_id
            candidate_targets[normalized_id] = item_id
            candidates.append((f"{path}.id", item, item_id, normalized_id))

        event_type = str(payload.get("type") or "")
        item_id_reference = payload.get("item_id")
        if isinstance(item_id_reference, str):
            normalized_reference = self._id_map.get(item_id_reference)
            if normalized_reference is None:
                normalized_reference = self._normalized_id_for_prefix(
                    item_id_reference,
                    _ITEM_ID_PREFIX_BY_EVENT.get(event_type),
                )
            if normalized_reference is not None:
                if self._would_collide(
                    item_id_reference,
                    normalized_reference,
                    existing_ids,
                ):
                    raise ResponsesCustomToolCallIdCollisionError(
                        "Responses item_id normalization would collide with an existing item ID"
                    )
                self._register(item_id_reference, normalized_reference)

        normalized_paths: list[str] = []
        normalized_ids = 0
        for path, item, item_id, normalized_id in candidates:
            self._register(item_id, normalized_id)
            item["id"] = normalized_id
            normalized_ids += 1
            if len(normalized_paths) < _MAX_RECORDED_PATHS:
                normalized_paths.append(path)

        rewritten_references = 0
        if isinstance(item_id_reference, str):
            normalized_reference = self._id_map.get(item_id_reference)
            if normalized_reference is not None:
                payload["item_id"] = normalized_reference
                rewritten_references = 1
                if len(normalized_paths) < _MAX_RECORDED_PATHS:
                    normalized_paths.append("item_id")

        self._seen_item_ids.update(
            item_id
            for _, item in item_locations
            if isinstance((item_id := item.get("id")), str)
        )

        return ResponsesCustomToolCallIdNormalizationResult(
            normalized_ids=normalized_ids,
            rewritten_references=rewritten_references,
            paths=tuple(normalized_paths),
        )

    def requires_item_id_full_normalization(
        self,
        event_type: str,
        item_id: str,
    ) -> bool:
        """Return whether this reference must stay on the stateful full path."""

        return item_id in self._id_map or (
            isinstance(item_id, str)
            and self._normalized_id_for_prefix(
                item_id,
                _ITEM_ID_PREFIX_BY_EVENT.get(event_type),
            )
            is not None
        )

    def _would_collide(
        self,
        item_id: str,
        normalized_id: str,
        current_item_ids: set[str],
    ) -> bool:
        if normalized_id == item_id:
            return False
        if normalized_id in current_item_ids:
            return True
        return (
            normalized_id in self._seen_item_ids
            and self._id_map.get(item_id) != normalized_id
        )

    def _register(self, item_id: str, normalized_id: str) -> None:
        existing = self._id_map.get(item_id)
        if existing is not None and existing != normalized_id:
            raise ResponsesCustomToolCallIdCollisionError(
                "Responses item ID normalization produced an inconsistent mapping"
            )
        self._id_map[item_id] = normalized_id

    @staticmethod
    def _normalized_id(item_id: Any, item_type: Any = None) -> str | None:
        prefix = _ITEM_ID_PREFIX_BY_TYPE.get(str(item_type))
        return ResponsesCustomToolCallIdNormalizer._normalized_id_for_prefix(
            item_id,
            prefix,
        )

    @staticmethod
    def _normalized_id_for_prefix(item_id: Any, prefix: str | None) -> str | None:
        if not isinstance(item_id, str) or not prefix:
            return None
        canonical_prefix = f"{prefix}_"
        canonical_suffix = item_id.removeprefix(canonical_prefix)
        if (
            item_id.startswith(canonical_prefix)
            and canonical_suffix
            and canonical_suffix.isascii()
            and canonical_suffix.isalnum()
            and len(item_id) <= _MAX_ITEM_ID_LENGTH
        ):
            return None

        match = _ITEM_ID_RE.fullmatch(item_id)
        if match is not None:
            candidate = f"{prefix}_{match.group(1)}"
            if len(candidate) <= _MAX_ITEM_ID_LENGTH:
                return candidate if candidate != item_id else None

        digest = sha256(item_id.encode("utf-8")).hexdigest()[:_HASHED_ITEM_ID_HEX_LENGTH]
        candidate = f"{prefix}_{digest}"
        return candidate if candidate != item_id else None

    @staticmethod
    def _item_locations(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        locations: list[tuple[str, dict[str, Any]]] = []

        direct_item = payload.get("item")
        if isinstance(direct_item, dict):
            locations.append(("item", direct_item))

        for collection_name in ("input", "output"):
            collection = payload.get(collection_name)
            if isinstance(collection, list):
                locations.extend(
                    (f"{collection_name}[{index}]", item)
                    for index, item in enumerate(collection)
                    if isinstance(item, dict)
                )

        response = payload.get("response")
        if isinstance(response, dict):
            output = response.get("output")
            if isinstance(output, list):
                locations.extend(
                    (f"response.output[{index}]", item)
                    for index, item in enumerate(output)
                    if isinstance(item, dict)
                )

        return locations
