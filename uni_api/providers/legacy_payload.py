"""Compatibility exports for the pre-refactor provider adapter module."""

from uni_api.providers.adapters import (
    PayloadProviderAdapter as LegacyPayloadAdapter,
    anthropic_adapter,
    audio_adapter,
    aws_bedrock_adapter,
    azure_adapter,
    azure_databricks_adapter,
    cloudflare_adapter,
    codex_adapter,
    cohere_adapter,
    content_generation_adapter,
    default_provider_adapters,
    embedding_adapter,
    gemini_adapter,
    image_adapter,
    moderation_adapter,
    openai_gpt_adapter,
    openrouter_adapter,
    search_adapter,
    vertex_anthropic_adapter,
    vertex_gemini_adapter,
)


def default_legacy_provider_adapters():
    return default_provider_adapters()
