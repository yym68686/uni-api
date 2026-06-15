from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from core.utils import BaseAPI
from uni_api.providers import payloads
from uni_api.providers.base import (
    NormalizedResponse,
    ProviderRequestContext,
    ProviderResponseContext,
    ProviderStreamContext,
    UpstreamRequest,
)

PayloadBuilder = Callable[[Any, str, dict[str, Any], str | None], Awaitable[tuple[str, dict[str, Any], Any]]]


class PayloadProviderAdapter:
    name: str
    supported_engines: set[str]

    def __init__(
        self,
        *,
        name: str,
        supported_engines: set[str],
        payload_builder: PayloadBuilder,
        normalize_chat_base_url: bool = False,
    ) -> None:
        self.name = name
        self.supported_engines = supported_engines
        self._payload_builder = payload_builder
        self._normalize_chat_base_url = normalize_chat_base_url

    def _provider_for_request(self, provider: dict[str, Any]) -> dict[str, Any]:
        request_provider = dict(provider)
        if self._normalize_chat_base_url and request_provider.get("base_url"):
            request_provider["base_url"] = BaseAPI(request_provider["base_url"]).chat_url
        return request_provider

    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        url, headers, body = await self._payload_builder(
            context.request,
            context.engine,
            self._provider_for_request(context.provider),
            context.api_key,
        )
        return UpstreamRequest(
            url=url,
            headers=dict(headers or {}),
            payload=body,
            stream=bool(getattr(context.request, "stream", False)),
            metadata={"engine": context.engine, "model": context.original_model},
        )

    async def parse_response(self, context: ProviderResponseContext) -> NormalizedResponse:
        return NormalizedResponse(
            body=context.response,
            status_code=getattr(context.response, "status_code", 200),
            metadata={"engine": context.engine, "model": context.model, "url": context.url},
        )

    async def parse_stream(self, context: ProviderStreamContext) -> AsyncIterator[bytes | str]:
        async for chunk in context.chunks:
            yield chunk


class ImageProviderAdapter(PayloadProviderAdapter):
    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        url, headers, body = await payloads.get_dalle_payload(
            context.request,
            context.engine,
            self._provider_for_request(context.provider),
            context.api_key,
            endpoint=context.endpoint,
        )
        return UpstreamRequest(
            url=url,
            headers=dict(headers or {}),
            payload=body,
            stream=bool(getattr(context.request, "stream", False)),
            metadata={"engine": context.engine, "model": context.original_model},
        )


class SearchProviderAdapter(PayloadProviderAdapter):
    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        url, headers, body = await payloads.get_search_payload(
            context.request,
            self._provider_for_request(context.provider),
            context.api_key,
        )
        return UpstreamRequest(
            url=url,
            headers=dict(headers or {}),
            payload=body,
            stream=False,
            metadata={"engine": context.engine, "model": context.original_model},
        )


class ContentGenerationProviderAdapter(PayloadProviderAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="content-generation",
            supported_engines={"content-generation"},
            payload_builder=_unsupported_content_generation_payload,
        )

    def get_video_adapter(self, config: dict[str, Any], provider: dict[str, Any], provider_name: str) -> Any:
        from video import get_video_adapter

        return get_video_adapter(config, provider, provider_name)


async def _unsupported_content_generation_payload(
    request: Any,
    engine: str,
    provider: dict[str, Any],
    api_key: str | None,
) -> tuple[str, dict[str, Any], Any]:
    _ = request, engine, provider, api_key
    raise NotImplementedError("content-generation uses the video adapter bridge")


openai_gpt_adapter = PayloadProviderAdapter(
    name="openai",
    supported_engines={"gpt"},
    payload_builder=payloads.get_gpt_payload,
    normalize_chat_base_url=True,
)
codex_adapter = PayloadProviderAdapter(name="codex", supported_engines={"codex"}, payload_builder=payloads.get_codex_payload)
gemini_adapter = PayloadProviderAdapter(name="gemini", supported_engines={"gemini"}, payload_builder=payloads.get_gemini_payload)
vertex_gemini_adapter = PayloadProviderAdapter(
    name="vertex-gemini",
    supported_engines={"vertex-gemini"},
    payload_builder=payloads.get_vertex_gemini_payload,
)
anthropic_adapter = PayloadProviderAdapter(name="anthropic", supported_engines={"claude"}, payload_builder=payloads.get_claude_payload)
vertex_anthropic_adapter = PayloadProviderAdapter(
    name="vertex-anthropic",
    supported_engines={"vertex-claude"},
    payload_builder=payloads.get_vertex_claude_payload,
)
aws_bedrock_adapter = PayloadProviderAdapter(name="aws-bedrock", supported_engines={"aws"}, payload_builder=payloads.get_aws_payload)
azure_adapter = PayloadProviderAdapter(name="azure", supported_engines={"azure"}, payload_builder=payloads.get_azure_payload)
azure_databricks_adapter = PayloadProviderAdapter(
    name="azure-databricks",
    supported_engines={"azure-databricks"},
    payload_builder=payloads.get_azure_databricks_payload,
)
openrouter_adapter = PayloadProviderAdapter(
    name="openrouter",
    supported_engines={"openrouter"},
    payload_builder=payloads.get_openrouter_payload,
)
cloudflare_adapter = PayloadProviderAdapter(
    name="cloudflare",
    supported_engines={"cloudflare"},
    payload_builder=payloads.get_cloudflare_payload,
)
cohere_adapter = PayloadProviderAdapter(name="cohere", supported_engines={"cohere"}, payload_builder=payloads.get_cohere_payload)
image_adapter = ImageProviderAdapter(name="image", supported_engines={"dalle"}, payload_builder=payloads.get_dalle_payload)
moderation_adapter = PayloadProviderAdapter(
    name="moderation",
    supported_engines={"moderation"},
    payload_builder=payloads.get_moderation_payload,
)
embedding_adapter = PayloadProviderAdapter(
    name="embedding",
    supported_engines={"embedding"},
    payload_builder=payloads.get_embedding_payload,
)
search_adapter = SearchProviderAdapter(name="search", supported_engines={"search"}, payload_builder=payloads.get_search_payload)
content_generation_adapter = ContentGenerationProviderAdapter()


async def _audio_payload(request: Any, engine: str, provider: dict[str, Any], api_key: str | None):
    if engine == "whisper":
        return await payloads.get_whisper_payload(request, engine, provider, api_key)
    return await payloads.get_tts_payload(request, engine, provider, api_key)


audio_adapter = PayloadProviderAdapter(name="audio", supported_engines={"tts", "whisper"}, payload_builder=_audio_payload)


def default_provider_adapters() -> tuple[PayloadProviderAdapter, ...]:
    return (
        openai_gpt_adapter,
        codex_adapter,
        gemini_adapter,
        vertex_gemini_adapter,
        anthropic_adapter,
        vertex_anthropic_adapter,
        aws_bedrock_adapter,
        azure_adapter,
        azure_databricks_adapter,
        openrouter_adapter,
        cloudflare_adapter,
        cohere_adapter,
        image_adapter,
        audio_adapter,
        moderation_adapter,
        embedding_adapter,
        search_adapter,
        content_generation_adapter,
    )
