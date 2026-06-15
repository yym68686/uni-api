import pytest

from uni_api.providers.base import (
    NormalizedResponse,
    ProviderRequestContext,
    ProviderResponseContext,
    ProviderStreamContext,
    UpstreamRequest,
)
from uni_api.providers.registry import ProviderRegistry


class DummyAdapter:
    name = "dummy"
    supported_engines = {"dummy-engine"}

    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        return UpstreamRequest(url="https://example.com", headers={}, payload={"model": context.original_model})

    async def parse_response(self, context: ProviderResponseContext) -> NormalizedResponse:
        return NormalizedResponse(body={"ok": True, "model": context.model})

    async def parse_stream(self, context: ProviderStreamContext):
        async for chunk in context.chunks:
            yield chunk


def test_provider_registry_registers_by_name_and_engine():
    adapter = DummyAdapter()
    registry = ProviderRegistry([adapter])

    assert registry.get("dummy") is adapter
    assert registry.for_engine("dummy-engine") is adapter
    assert registry.names() == ("dummy",)
    assert registry.engines() == ("dummy-engine",)


def test_provider_registry_rejects_duplicate_names_and_engines():
    registry = ProviderRegistry([DummyAdapter()])

    with pytest.raises(ValueError):
        registry.register(DummyAdapter())
