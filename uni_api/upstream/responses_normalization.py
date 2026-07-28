from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_CUSTOM_TOOL_CALL_ITEM_ID_RE = re.compile(r"^item_([A-Za-z0-9]+)$")
_CUSTOM_TOOL_CALL_INPUT_EVENT_TYPES = frozenset(
    {
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    }
)
_MAX_RECORDED_PATHS = 16
_DEFAULT_NORMALIZATION_MODELS_BY_PROVIDER = {
    "fugue-codex": frozenset({"gpt-5.6-sol"}),
    "937auth": frozenset({"gpt-5.6-sol"}),
    "937auth01": frozenset({"gpt-5.6-sol"}),
}


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
        defaults = _DEFAULT_NORMALIZATION_MODELS_BY_PROVIDER.get(
            str(provider.get("provider") or ""),
            frozenset(),
        )
        return bool(requested_models.intersection(defaults))

    configured = preferences["normalize_responses_custom_tool_call_ids"]
    if configured is True:
        return True
    if not isinstance(configured, (list, tuple, set, frozenset)):
        return False
    enabled_models = {str(model).strip() for model in configured if str(model).strip()}
    return "*" in enabled_models or bool(requested_models.intersection(enabled_models))


class ResponsesCustomToolCallIdNormalizer:
    """Normalize non-canonical custom tool call item IDs without touching content."""

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

        for path, item in item_locations:
            if item.get("type") != "custom_tool_call":
                continue
            item_id = item.get("id")
            normalized_id = self._normalized_id(item_id)
            if normalized_id is None:
                continue
            if self._would_collide(item_id, normalized_id, existing_ids):
                raise ResponsesCustomToolCallIdCollisionError(
                    f"custom_tool_call ID normalization at {path}.id would collide with an existing item ID"
                )
            candidates.append((f"{path}.id", item, item_id, normalized_id))

        event_type = str(payload.get("type") or "")
        item_id_reference = payload.get("item_id")
        if (
            event_type in _CUSTOM_TOOL_CALL_INPUT_EVENT_TYPES
            and isinstance(item_id_reference, str)
        ):
            normalized_reference = self._normalized_id(item_id_reference)
            if normalized_reference is not None:
                if self._would_collide(
                    item_id_reference,
                    normalized_reference,
                    existing_ids,
                ):
                    raise ResponsesCustomToolCallIdCollisionError(
                        "custom tool call event item_id normalization would collide with an existing item ID"
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

        return (
            event_type in _CUSTOM_TOOL_CALL_INPUT_EVENT_TYPES
            or item_id in self._id_map
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
                "custom_tool_call ID normalization produced an inconsistent mapping"
            )
        self._id_map[item_id] = normalized_id

    @staticmethod
    def _normalized_id(item_id: Any) -> str | None:
        if not isinstance(item_id, str):
            return None
        match = _CUSTOM_TOOL_CALL_ITEM_ID_RE.fullmatch(item_id)
        if match is None:
            return None
        return f"ctc_{match.group(1)}"

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
