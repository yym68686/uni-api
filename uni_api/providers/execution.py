from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException

from core.utils import get_engine, get_model_dict, safe_get
from uni_api.providers.base import ProviderRequestContext
from uni_api.providers.header_passthrough import apply_provider_preference_headers
from uni_api.providers.payloads import force_codex_client_headers
from uni_api.providers.registry import ProviderRegistry


@dataclass(frozen=True)
class PreparedProviderRequest:
    original_model: str
    engine: str
    channel_id: str
    proxy: Optional[str]
    provider_api_key_raw: Optional[str]
    api_key: Optional[str]
    codex_account_id: Optional[str]
    url: str
    headers: dict[str, Any]
    payload: Any
    last_message_role: Any


async def prepare_provider_request(
    *,
    request: Any,
    provider: dict[str, Any],
    endpoint: Optional[str],
    provider_api_key_raw: Optional[str],
    runtime_api_list: list[str],
    config: dict[str, Any],
    provider_registry: ProviderRegistry,
    select_provider_api_key_raw: Callable[[dict[str, Any], str, list[str]], Awaitable[Optional[str]]],
    resolve_codex_upstream_auth: Callable[[str, Optional[str], Optional[str]], Awaitable[tuple[Optional[str], Optional[str]]]],
    http_request: Any | None = None,
) -> PreparedProviderRequest:
    model_dict = provider.get("_model_dict_cache") or get_model_dict(provider)
    original_model = model_dict[request.model]
    if provider_api_key_raw is None:
        provider_api_key_raw = await select_provider_api_key_raw(
            provider,
            original_model,
            runtime_api_list,
        )

    engine, stream_mode = get_engine(provider, endpoint, original_model)
    if stream_mode is not None:
        request.stream = stream_mode

    proxy = safe_get(config, "preferences", "proxy", default=None)
    proxy = safe_get(provider, "preferences", "proxy", default=proxy)

    api_key = provider_api_key_raw
    codex_account_id = None
    if engine == "codex" and provider_api_key_raw:
        try:
            api_key, codex_account_id = await resolve_codex_upstream_auth(
                provider["provider"],
                provider_api_key_raw,
                proxy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        has_audio_modality = any(str(modality).lower() == "audio" for modality in (getattr(request, "modalities", None) or []))
    except Exception:
        has_audio_modality = False
    if engine in ["gemini", "vertex-gemini"] and (has_audio_modality or "preview-tts" in original_model.lower()):
        request.stream = False

    provider_adapter = provider_registry.for_engine(engine)
    upstream_request = await provider_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine=engine,
            original_model=original_model,
            api_key=api_key,
            endpoint=endpoint,
        )
    )

    headers = dict(upstream_request.headers)
    if engine == "codex" and codex_account_id:
        headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))
    apply_provider_preference_headers(headers, provider, http_request=http_request)
    if engine == "codex":
        force_codex_client_headers(headers)

    return PreparedProviderRequest(
        original_model=original_model,
        engine=engine,
        channel_id=f"{provider['provider']}",
        proxy=proxy,
        provider_api_key_raw=provider_api_key_raw,
        api_key=api_key,
        codex_account_id=codex_account_id,
        url=upstream_request.url,
        headers=headers,
        payload=upstream_request.payload,
        last_message_role=safe_get(request, "messages", -1, "role", default=None),
    )
