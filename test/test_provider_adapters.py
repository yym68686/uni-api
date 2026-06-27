import json
import inspect
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.models import (
    AudioTranscriptionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    ModerationRequest,
    RequestModel,
    TextToSpeechRequest,
)
from uni_api.providers.base import ProviderRequestContext, ProviderResponseContext, ProviderStreamContext
from uni_api.providers.adapters import (
    anthropic_adapter,
    audio_adapter,
    aws_bedrock_adapter,
    azure_adapter,
    azure_databricks_adapter,
    cloudflare_adapter,
    cohere_adapter,
    codex_adapter,
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
from uni_api.providers.registry import ProviderRegistry


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


async def test_openai_gpt_adapter_builds_request_from_golden_fixture():
    fixture = json.loads((FIXTURE_ROOT / "adapters" / "openai_chat_request.json").read_text())
    request = RequestModel(**fixture["request"])
    provider = {
        "provider": "openai-a",
        "base_url": "https://api.example.com/v1",
        "api": "upstream-key",
        "model": ["gpt-4.1"],
        "tools": True,
    }

    upstream_request = await openai_gpt_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine="gpt",
            original_model="gpt-4.1",
            api_key="upstream-key",
            endpoint="/v1/chat/completions",
        )
    )

    assert upstream_request.url == "https://api.example.com/v1/chat/completions"
    assert upstream_request.headers["Authorization"] == "Bearer upstream-key"
    assert upstream_request.payload["model"] == "gpt-4.1"
    assert upstream_request.payload["messages"][0]["content"] == "hello"


def test_default_provider_registry_allows_engine_lookup_without_runner_changes():
    registry = ProviderRegistry(default_provider_adapters())

    assert registry.for_engine("gpt") is openai_gpt_adapter
    assert registry.for_engine("codex") is codex_adapter
    assert registry.for_engine("openrouter") is openrouter_adapter
    assert registry.for_engine("cloudflare") is cloudflare_adapter
    assert registry.for_engine("cohere") is cohere_adapter
    assert registry.for_engine("dalle") is image_adapter
    assert registry.for_engine("tts") is audio_adapter
    assert registry.for_engine("whisper") is audio_adapter
    assert registry.for_engine("moderation") is moderation_adapter
    assert registry.for_engine("embedding") is embedding_adapter
    assert registry.for_engine("search") is search_adapter
    assert registry.for_engine("content-generation") is content_generation_adapter


async def test_codex_adapter_builds_responses_request_from_chat_shape():
    fixture = json.loads((FIXTURE_ROOT / "adapters" / "openai_chat_request.json").read_text())
    request = RequestModel(**fixture["request"])
    request.model = "gpt-5.4"
    provider = {
        "provider": "codex-a",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "upstream-key",
        "engine": "codex",
        "model": ["gpt-5.4"],
        "tools": True,
    }

    upstream_request = await codex_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine="codex",
            original_model="gpt-5.4",
            api_key="upstream-key",
            endpoint="/v1/chat/completions",
        )
    )

    assert upstream_request.url == "https://chatgpt.com/backend-api/codex/responses"
    assert upstream_request.headers["Authorization"] == "Bearer upstream-key"
    assert upstream_request.payload["model"] == "gpt-5.4"
    assert upstream_request.payload["input"][0]["type"] == "message"
    assert upstream_request.payload["input"][0]["content"][0]["type"] == "input_text"


async def test_codex_adapter_infers_empty_tool_call_name_from_unique_tool_schema():
    request = RequestModel(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "run pwd"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_shell",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": json.dumps({"command": "pwd", "justification": "check cwd"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_shell", "content": "/tmp"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "shell_command",
                    "parameters": {
                        "type": "object",
                        "required": ["command"],
                        "properties": {
                            "command": {"type": "string"},
                            "justification": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_plan",
                    "parameters": {
                        "type": "object",
                        "required": ["plan"],
                        "properties": {"plan": {"type": "array"}},
                    },
                },
            },
        ],
    )
    provider = {
        "provider": "codex-a",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "upstream-key",
        "engine": "codex",
        "model": ["gpt-5.4"],
        "tools": True,
    }

    upstream_request = await codex_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine="codex",
            original_model="gpt-5.4",
            api_key="upstream-key",
            endpoint="/v1/chat/completions",
        )
    )

    function_call = next(item for item in upstream_request.payload["input"] if item["type"] == "function_call")
    function_output = next(item for item in upstream_request.payload["input"] if item["type"] == "function_call_output")
    assert function_call["name"] == "shell_command"
    assert function_call["call_id"] == "call_shell"
    assert function_output["call_id"] == "call_shell"


async def test_codex_adapter_rejects_empty_tool_call_name_when_schema_match_is_ambiguous():
    request = RequestModel(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "run a tool"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_ambiguous",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": json.dumps({"value": "x"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_ambiguous", "content": "ok"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "parameters": {
                        "type": "object",
                        "required": ["value"],
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "parameters": {
                        "type": "object",
                        "required": ["value"],
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
        ],
    )
    provider = {
        "provider": "codex-a",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "upstream-key",
        "engine": "codex",
        "model": ["gpt-5.4"],
        "tools": True,
    }

    with pytest.raises(HTTPException) as exc_info:
        await codex_adapter.build_request(
            ProviderRequestContext(
                request=request,
                provider=provider,
                engine="codex",
                original_model="gpt-5.4",
                api_key="upstream-key",
                endpoint="/v1/chat/completions",
            )
        )

    assert exc_info.value.status_code == 400
    assert "messages[1].tool_calls[0].function.name is empty" in exc_info.value.detail
    assert "arguments match multiple declared tool schemas" in exc_info.value.detail


async def test_codex_adapter_moves_system_messages_into_instructions():
    request = RequestModel(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": "First instruction."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Second instruction."},
        ],
        stream=False,
    )
    provider = {
        "provider": "codex-a",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "upstream-key",
        "engine": "codex",
        "model": ["gpt-5.4"],
        "tools": True,
    }

    upstream_request = await codex_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine="codex",
            original_model="gpt-5.4",
            api_key="upstream-key",
            endpoint="/v1/chat/completions",
        )
    )

    assert upstream_request.payload["instructions"] == "First instruction.\n\nSecond instruction."
    assert [item["role"] for item in upstream_request.payload["input"] if item["type"] == "message"] == ["user"]
    assert all(item["role"] != "developer" for item in upstream_request.payload["input"] if item["type"] == "message")


async def test_search_adapter_builds_jina_request_from_chat_query():
    request = RequestModel(
        model="search",
        messages=[{"role": "user", "content": "uni api"}],
    )
    provider = {
        "provider": "jina",
        "base_url": "https://api.jina.ai",
        "api": "upstream-key",
        "model": ["search"],
    }

    upstream_request = await search_adapter.build_request(
        ProviderRequestContext(
            request=request,
            provider=provider,
            engine="search",
            original_model="search",
            api_key="upstream-key",
            endpoint="/v1/search",
        )
    )

    assert upstream_request.url == "https://s.jina.ai/"
    assert upstream_request.headers["Authorization"] == "Bearer upstream-key"
    assert upstream_request.payload == {"q": "uni api"}


async def test_provider_adapter_request_golden_matrix():
    def chat(model: str) -> RequestModel:
        return RequestModel(model=model, messages=[{"role": "user", "content": "hello"}], stream=False)

    cases = [
        (
            "gemini",
            gemini_adapter,
            chat("gemini-1.5-pro"),
            "gemini",
            {"provider": "gemini-a", "base_url": "https://generativelanguage.googleapis.com/v1beta/models", "model": ["gemini-1.5-pro"]},
            "upstream-key",
            lambda request: (
                request.url.endswith("/models/gemini-1.5-pro:generateContent?key=upstream-key")
                and request.payload["contents"][0]["parts"][0]["text"] == "hello"
            ),
        ),
        (
            "vertex-gemini",
            vertex_gemini_adapter,
            chat("gemini-1.5-pro"),
            "vertex-gemini",
            {"provider": "vertex-gemini-a", "base_url": "https://aiplatform.googleapis.com", "model": ["gemini-1.5-pro"]},
            "ab.c",
            lambda request: (
                request.url.endswith("/v1/publishers/google/models/gemini-1.5-pro:generateContent?key=ab.c")
                and request.payload["contents"][0]["parts"][0]["text"] == "hello"
            ),
        ),
        (
            "anthropic",
            anthropic_adapter,
            chat("claude-3-haiku"),
            "claude",
            {"provider": "anthropic-a", "base_url": "https://api.anthropic.com/v1/messages", "model": ["claude-3-haiku"]},
            "upstream-key",
            lambda request: request.headers["x-api-key"] == "upstream-key" and request.payload["messages"][0]["content"] == "hello",
        ),
        (
            "vertex-anthropic",
            vertex_anthropic_adapter,
            chat("claude-3-haiku"),
            "vertex-claude",
            {
                "provider": "vertex-anthropic-a",
                "base_url": "https://us-central1-aiplatform.googleapis.com",
                "project_id": "project-a",
                "model": ["claude-3-haiku"],
            },
            "upstream-key",
            lambda request: (
                "/projects/project-a/locations/us-east5/" in request.url
                and "/publishers/anthropic/models/claude-3-haiku:streamRawPredict" in request.url
                and request.payload["messages"][0]["content"] == "hello"
            ),
        ),
        (
            "aws-bedrock",
            aws_bedrock_adapter,
            chat("anthropic.claude-3-haiku"),
            "aws",
            {
                "provider": "aws-a",
                "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "model": ["anthropic.claude-3-haiku"],
                "aws_access_key": "AKIA_TEST",
                "aws_secret_key": "secret",
            },
            "unused",
            lambda request: (
                request.url.endswith("/model/anthropic.claude-3-haiku/invoke-with-response-stream")
                and request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")
                and request.payload["messages"][0]["content"] == "hello"
            ),
        ),
        (
            "azure",
            azure_adapter,
            chat("deployment-a"),
            "azure",
            {"provider": "azure-a", "base_url": "https://azure.example.com", "model": ["deployment-a"]},
            "upstream-key",
            lambda request: (
                request.url == "https://azure.example.com/openai/deployments/deployment-a/chat/completions?api-version=2025-01-01-preview"
                and request.headers["api-key"] == "upstream-key"
                and request.payload["model"] == "deployment-a"
            ),
        ),
        (
            "azure-databricks",
            azure_databricks_adapter,
            chat("serving-a"),
            "azure-databricks",
            {"provider": "databricks-a", "base_url": "https://dbc.example.com", "model": ["serving-a"]},
            "upstream-key",
            lambda request: (
                request.url == "https://dbc.example.com/serving-endpoints/serving-a/invocations"
                and request.headers["Authorization"].startswith("Basic ")
                and request.payload["model"] == "serving-a"
            ),
        ),
        (
            "openrouter",
            openrouter_adapter,
            chat("openrouter/model"),
            "openrouter",
            {"provider": "openrouter-a", "base_url": "https://openrouter.ai/api/v1/chat/completions", "model": ["openrouter/model"]},
            "upstream-key",
            lambda request: request.headers["HTTP-Referer"] == "https://github.com/yym68686/uni-api" and request.payload["model"] == "openrouter/model",
        ),
        (
            "cloudflare",
            cloudflare_adapter,
            chat("@cf/meta/llama"),
            "cloudflare",
            {"provider": "cloudflare-a", "base_url": "https://api.cloudflare.com", "cf_account_id": "account-a", "model": ["@cf/meta/llama"]},
            "upstream-key",
            lambda request: request.url == "https://api.cloudflare.com/client/v4/accounts/account-a/ai/run/@cf/meta/llama"
            and request.payload["prompt"] == "hello",
        ),
        (
            "cohere",
            cohere_adapter,
            chat("command-r"),
            "cohere",
            {"provider": "cohere-a", "base_url": "https://api.cohere.ai/v1/chat", "model": ["command-r"]},
            "upstream-key",
            lambda request: request.payload["model"] == "command-r" and request.payload["message"] == "hello",
        ),
        (
            "image",
            image_adapter,
            ImageGenerationRequest(model="gpt-image-1", prompt="draw a square"),
            "dalle",
            {"provider": "image-a", "base_url": "https://api.openai.com/v1", "model": ["gpt-image-1"]},
            "upstream-key",
            lambda request: request.url == "https://api.openai.com/v1/images/generations"
            and request.payload["model"] == "gpt-image-1"
            and request.payload["prompt"] == "draw a square",
        ),
        (
            "audio-speech",
            audio_adapter,
            TextToSpeechRequest(model="tts-1", input="speak", voice="alloy"),
            "tts",
            {"provider": "audio-a", "base_url": "https://api.openai.com/v1", "model": ["tts-1"]},
            "upstream-key",
            lambda request: request.url == "https://api.openai.com/v1/audio/speech"
            and request.payload["model"] == "tts-1"
            and request.payload["voice"] == "alloy",
        ),
        (
            "audio-transcription",
            audio_adapter,
            AudioTranscriptionRequest(model="whisper-1", file=("audio.wav", BytesIO(b"wav"), "audio/wav")),
            "whisper",
            {"provider": "audio-a", "base_url": "https://api.openai.com/v1", "model": ["whisper-1"]},
            "upstream-key",
            lambda request: request.url == "https://api.openai.com/v1/audio/transcriptions"
            and request.payload["model"] == "whisper-1"
            and request.payload["file"][0] == "audio.wav",
        ),
        (
            "moderation",
            moderation_adapter,
            ModerationRequest(model="omni-moderation-latest", input="text to check"),
            "moderation",
            {"provider": "moderation-a", "base_url": "https://api.openai.com/v1", "model": ["omni-moderation-latest"]},
            "upstream-key",
            lambda request: request.url == "https://api.openai.com/v1/moderations"
            and request.payload == {"model": "omni-moderation-latest", "input": "text to check"},
        ),
        (
            "embedding",
            embedding_adapter,
            EmbeddingRequest(model="text-embedding-3-small", input="hello"),
            "embedding",
            {"provider": "embedding-a", "base_url": "https://api.openai.com/v1", "model": ["text-embedding-3-small"]},
            "upstream-key",
            lambda request: request.url == "https://api.openai.com/v1/embeddings"
            and request.payload["model"] == "text-embedding-3-small"
            and request.payload["input"] == "hello",
        ),
    ]

    for label, adapter, request, engine, provider, api_key, validator in cases:
        upstream_request = await adapter.build_request(
            ProviderRequestContext(
                request=request,
                provider=provider,
                engine=engine,
                original_model=provider["model"][0],
                api_key=api_key,
                endpoint="/v1/images/generations" if label == "image" else None,
            )
        )
        assert validator(upstream_request), label


def test_content_generation_adapter_bridges_existing_video_adapter():
    provider = {
        "provider": "video-a",
        "engine": "content-generation",
        "base_url": "https://video.example.com",
        "model": {"video": "seedance"},
    }
    adapter = content_generation_adapter.get_video_adapter({}, provider, "video-a")

    upstream_request = adapter.build_request(
        method="POST",
        task_id=None,
        request_body={"model": "video", "prompt": "make a video"},
        request_model_name="video",
        original_model="seedance",
        provider=provider,
        provider_name="video-a",
        provider_api_key_raw="upstream-key",
    )

    assert upstream_request.method == "POST"
    assert upstream_request.url == "https://video.example.com/api/v3/contents/generations/tasks"
    assert upstream_request.headers["Authorization"] == "Bearer upstream-key"
    assert upstream_request.payload["model"] == "seedance"
    assert upstream_request.payload["content"][0] == {"type": "text", "text": "make a video"}


async def test_provider_adapters_parse_response_and_stream_golden_matrix():
    async def chunks():
        yield b"data: ok\n\n"
        yield "data: [DONE]\n\n"

    for adapter in default_provider_adapters():
        engine = sorted(adapter.supported_engines)[0]
        normalized = await adapter.parse_response(
            ProviderResponseContext(
                response=SimpleNamespace(status_code=203, body={"ok": True}),
                provider={"provider": adapter.name},
                engine=engine,
                model="model-a",
                url="https://upstream.example.com",
            )
        )
        assert normalized.status_code == 203, adapter.name
        assert normalized.metadata == {
            "engine": engine,
            "model": "model-a",
            "url": "https://upstream.example.com",
        }

        stream = adapter.parse_stream(
            ProviderStreamContext(
                chunks=chunks(),
                provider={"provider": adapter.name},
                engine=engine,
                model="model-a",
                url="https://upstream.example.com",
            )
        )
        assert [chunk async for chunk in stream] == [b"data: ok\n\n", "data: [DONE]\n\n"], adapter.name


def test_process_request_uses_provider_registry_instead_of_payload_dispatch():
    import main
    from uni_api.providers.execution import prepare_provider_request

    source = inspect.getsource(main.process_request)
    helper_source = inspect.getsource(prepare_provider_request)
    assert "provider_registry.for_engine(engine)" in helper_source
    assert "prepare_provider_request(" in source
    assert "await get_payload(" not in source


def test_routing_planner_does_not_depend_on_provider_adapter_registry():
    import uni_api.routing.planner as planner

    source = inspect.getsource(planner)
    assert "ProviderRegistry" not in source
    assert "build_request" not in source
