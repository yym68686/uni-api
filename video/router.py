from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

from .adapters.callxyq import CallxyqVideoAdapter
from .adapters.deyunai import DeyunaiVideoAdapter
from .adapters.lingjing import LingjingVideoAdapter
from .config import find_video_provider_config
from .schema import VideoProviderConfig
from .adapters.base import VideoProviderAdapter


ADAPTERS = {
    "callxyq": CallxyqVideoAdapter,
    "deyunai": DeyunaiVideoAdapter,
    "content_generation": DeyunaiVideoAdapter,
    "content-generation": DeyunaiVideoAdapter,
    "http_json": DeyunaiVideoAdapter,
    "lingjing": LingjingVideoAdapter,
}


def _infer_adapter_name(provider: dict[str, Any], provider_name: str, config: Optional[VideoProviderConfig]) -> str:
    if config and config.adapter:
        return config.adapter

    provider_name_lower = str(provider_name or provider.get("provider") or "").strip().lower()
    if provider_name_lower in ADAPTERS:
        return provider_name_lower

    engine = str(provider.get("engine") or "").strip().lower()
    if engine in ADAPTERS:
        return engine

    parsed = urlparse(str(provider.get("base_url") or ""))
    host = parsed.netloc.lower()
    if host.endswith("lingjingai.cn"):
        return "lingjing"
    if host.endswith("callxyq.xyz"):
        return "callxyq"
    return "deyunai"


def get_video_adapter(
    config: dict[str, Any],
    provider: dict[str, Any],
    provider_name: str,
) -> VideoProviderAdapter:
    provider_config = find_video_provider_config(config, provider_name)
    adapter_name = _infer_adapter_name(provider, provider_name, provider_config)
    adapter_cls = ADAPTERS.get(adapter_name)
    if adapter_cls is None:
        raise ValueError(f"Unsupported video provider adapter: {adapter_name}")
    return adapter_cls(provider_config)


__all__ = ["get_video_adapter"]
