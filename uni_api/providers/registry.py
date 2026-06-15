from __future__ import annotations

from typing import Iterable

from uni_api.providers.base import ProviderAdapter


class ProviderRegistry:
    def __init__(self, adapters: Iterable[ProviderAdapter] = ()) -> None:
        self._by_name: dict[str, ProviderAdapter] = {}
        self._by_engine: dict[str, ProviderAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ProviderAdapter) -> None:
        name = str(adapter.name or "").strip()
        if not name:
            raise ValueError("Provider adapter name is required")
        if name in self._by_name:
            raise ValueError(f"Provider adapter already registered: {name}")

        self._by_name[name] = adapter
        for engine in adapter.supported_engines:
            engine_name = str(engine or "").strip()
            if not engine_name:
                continue
            if engine_name in self._by_engine:
                raise ValueError(f"Provider engine already registered: {engine_name}")
            self._by_engine[engine_name] = adapter

    def get(self, name: str) -> ProviderAdapter:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider adapter: {name}") from exc

    def for_engine(self, engine: str) -> ProviderAdapter:
        try:
            return self._by_engine[engine]
        except KeyError as exc:
            raise KeyError(f"Unknown provider engine: {engine}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())

    def engines(self) -> tuple[str, ...]:
        return tuple(self._by_engine.keys())
