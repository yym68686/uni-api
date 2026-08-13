import asyncio
import base64
import gzip

import httpx
import pytest
from fastapi import HTTPException

from uni_api.admission import (
    RequestAdmissionController,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
import uni_api.providers.image_inputs as image_inputs
from core.utils import gemini_audio_inline_data_to_wav_base64
from uni_api.providers.image_inputs import build_image_message
from uni_api.providers.normalization import (
    GEMINI_THOUGHT_SIGNATURE_MAX_BYTES,
    GeminiThoughtSignatureTooLarge,
    cache_put_gemini_image_thought_signature,
    clear_gemini_thought_signature_cache,
    gemini_thought_signature_cache_snapshot,
    normalize_gemini_parts,
)


_FORMER_FIXED_IMAGE_LIMIT_BYTES = 8 * 1024 * 1024


def _data_url(media_type: str, content: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(content).decode()}"


def test_webp_is_validated_and_forwarded_without_pillow_transcoding():
    webp = b"RIFF" + (4).to_bytes(4, "little") + b"WEBP" + b"VP8 "

    async def run():
        return await build_image_message(_data_url("image/webp", webp), "claude")

    message = asyncio.run(run())
    assert message["source"]["media_type"] == "image/webp"
    assert base64.b64decode(message["source"]["data"]) == webp


def test_data_url_rejects_declared_type_mismatch():
    png = b"\x89PNG\r\n\x1a\n" + b"payload"

    async def mismatch():
        await build_image_message(_data_url("image/jpeg", png), "gpt")

    with pytest.raises(HTTPException) as rejected_type:
        asyncio.run(mismatch())
    assert rejected_type.value.status_code == 400


def test_data_url_accepts_image_above_former_fixed_limit():
    image = b"\x89PNG\r\n\x1a\n" + b"x" * _FORMER_FIXED_IMAGE_LIMIT_BYTES
    image_url = _data_url("image/png", image)

    async def accepted():
        return await build_image_message(image_url, "gpt")

    assert asyncio.run(accepted())["image_url"]["url"] is image_url


def test_data_url_strictly_validates_full_base64():
    valid_small = _data_url("image/png", b"\x89PNG\r\n\x1a\n" + b"payload")
    comma = valid_small.index(",")
    corrupted = valid_small[: comma + 12] + "!!!!" + valid_small[comma + 16 :]

    async def rejected_invalid_tail():
        await build_image_message(corrupted, "gpt")

    with pytest.raises(HTTPException) as rejected_base64:
        asyncio.run(rejected_invalid_tail())
    assert rejected_base64.value.status_code == 400


def test_image_output_copy_is_charged_until_request_lifetime_ends():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=16 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            png = b"\x89PNG\r\n\x1a\n" + b"x" * 1024
            message = await build_image_message(_data_url("image/png", png), "gemini")
            assert message["inlineData"]["mimeType"] == "image/png"
            assert controller.snapshot()["reserved_response_bytes"] > 0
        finally:
            reset_request_admission_lease(token)
            await lease.release()
        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


def test_remote_image_fetch_uses_explicit_30_second_httpx_timeout(monkeypatch):
    observed_timeout = None

    async def handler(request):
        nonlocal observed_timeout
        observed_timeout = request.extensions.get("timeout")
        return httpx.Response(
            200,
            content=b"\x89PNG\r\n\x1a\nsmall",
        )

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(image_inputs.httpx, "AsyncClient", client_factory)

    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            return await build_image_message(
                "https://image.example/small.png",
                "gpt",
            )
        finally:
            reset_request_admission_lease(token)
            await lease.release()

    message = asyncio.run(run())

    assert message["image_url"]["url"].startswith("data:image/png;base64,")
    assert observed_timeout == {
        "connect": 30.0,
        "read": 30.0,
        "write": 30.0,
        "pool": 30.0,
    }


def test_remote_image_fetch_total_deadline_still_maps_to_408(monkeypatch):
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(1)
            yield b"\x89PNG\r\n\x1a\nsmall"

        async def aclose(self):
            return None

    async def handler(_request):
        return httpx.Response(200, stream=SlowStream())

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(image_inputs.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(image_inputs, "IMAGE_FETCH_TIMEOUT_SECONDS", 0.01)

    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            await build_image_message(
                "https://image.example/slow.png",
                "gpt",
            )
        finally:
            reset_request_admission_lease(token)
            await lease.release()

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(run())

    assert rejected.value.status_code == 408


def test_remote_image_above_former_limit_is_governed_by_admission(monkeypatch):
    decoded = (
        b"\x89PNG\r\n\x1a\n" + b"x" * _FORMER_FIXED_IMAGE_LIMIT_BYTES
    )
    compressed = gzip.compress(decoded)

    class CompressedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield compressed

        async def aclose(self):
            return None

    async def handler(_request):
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=CompressedStream(),
        )

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(image_inputs.httpx, "AsyncClient", client_factory)

    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=96 * 1024 * 1024,
            max_response_bytes=96 * 1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            message = await build_image_message(
                "https://image.example/large.png",
                "gpt",
            )
            assert base64.b64decode(
                message["image_url"]["url"].partition(",")[2]
            ) == decoded
            assert controller.snapshot()["reserved_response_bytes"] > 0
        finally:
            reset_request_admission_lease(token)
            await lease.release()
        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


def test_remote_image_stops_when_dynamic_admission_rejects(monkeypatch):
    decoded = b"\x89PNG\r\n\x1a\n" + b"x" * (128 * 1024)
    compressed = gzip.compress(decoded)

    class CompressedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield compressed

        async def aclose(self):
            return None

    async def handler(_request):
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=CompressedStream(),
        )

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(image_inputs.httpx, "AsyncClient", client_factory)

    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=64 * 1024,
            body_budget_bytes=256 * 1024,
            max_response_bytes=256 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        try:
            with pytest.raises(UpstreamResponseBudgetExhausted) as rejected:
                await build_image_message(
                    "https://image.example/resource-pressure.png",
                    "gpt",
                )
            assert rejected.value.status_code == 503
            assert rejected.value.local_admission_rejection is True
        finally:
            reset_request_admission_lease(token)
            await lease.release()
        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


def test_gemini_signature_cache_is_bounded_by_entry_and_total_bytes():
    clear_gemini_thought_signature_cache()
    rejected_before = gemini_thought_signature_cache_snapshot()["rejected"]
    oversized = "s" * (GEMINI_THOUGHT_SIGNATURE_MAX_BYTES + 1)
    first_image = base64.b64encode(b"image-a").decode()
    assert cache_put_gemini_image_thought_signature(first_image, oversized) is False
    snapshot = gemini_thought_signature_cache_snapshot()
    assert snapshot["items"] == 0
    assert snapshot["bytes"] == 0
    assert snapshot["rejected"] == rejected_before + 1

    signature = "s" * GEMINI_THOUGHT_SIGNATURE_MAX_BYTES
    for index in range(100):
        assert cache_put_gemini_image_thought_signature(
            base64.b64encode(f"image-{index}".encode()).decode(),
            signature,
        )
    snapshot = gemini_thought_signature_cache_snapshot()
    assert snapshot["items"] <= snapshot["max_items"]
    assert snapshot["bytes"] <= snapshot["max_bytes"]
    assert snapshot["items"] == snapshot["max_bytes"] // len(signature)
    clear_gemini_thought_signature_cache()


def test_gemini_signature_cache_keys_decoded_bytes_not_base64_spelling():
    clear_gemini_thought_signature_cache()
    png = b"\x89PNG\r\n\x1a\n" + b"canonical-image"
    padded = base64.b64encode(png).decode()
    assert padded.endswith("=")
    assert cache_put_gemini_image_thought_signature(padded, "signature")

    async def run():
        unpadded_url = f"data:image/png;base64,{padded.rstrip('=')}"
        return await build_image_message(unpadded_url, "gemini")

    message = asyncio.run(run())
    assert message["thoughtSignature"] == "signature"
    clear_gemini_thought_signature_cache()


def test_chunked_audio_conversion_matches_legacy_wav_bytes_for_odd_lengths():
    async def run():
        mime_type = "audio/L16;codec=pcm;rate=24000"
        for pcm in (b"x", b"xy", b"xyz", b"odd!!"):
            encoded = base64.b64encode(pcm).decode()
            normalized = await normalize_gemini_parts(
                [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded,
                        }
                    }
                ]
            )
            assert normalized.audio_wav_base64 == (
                gemini_audio_inline_data_to_wav_base64(mime_type, encoded)
            )

    asyncio.run(run())


def test_attacker_sized_audio_mime_rate_is_ignored_without_generic_failure():
    async def run():
        normalized = await normalize_gemini_parts(
            [
                {
                    "inlineData": {
                        "mimeType": "audio/L16;codec=pcm;rate=" + "9" * 5000,
                        "data": base64.b64encode(b"pcm").decode(),
                    }
                }
            ]
        )
        assert normalized.audio_wav_base64 is None

    asyncio.run(run())


def test_four_large_image_validations_do_not_monopolize_event_loop():
    image = b"\x89PNG\r\n\x1a\n" + b"x" * (8 * 1024 * 1024 - 8)
    url = _data_url("image/png", image)

    async def run():
        gaps = []
        stop = asyncio.Event()

        async def heartbeat():
            loop = asyncio.get_running_loop()
            previous = loop.time()
            while not stop.is_set():
                await asyncio.sleep(0.001)
                now = loop.time()
                gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            await asyncio.gather(
                *(build_image_message(url, "claude") for _ in range(4))
            )
        finally:
            stop.set()
            await heartbeat_task
        return max(gaps, default=0.0)

    assert asyncio.run(run()) < 0.06


def test_four_maximum_audio_conversions_do_not_monopolize_event_loop():
    pcm = b"\x00\x01" * (3 * 1024 * 1024)
    encoded = base64.b64encode(pcm).decode()
    parts = [
        {
            "inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=24000",
                "data": encoded,
            }
        }
    ]

    async def run():
        gaps = []
        stop = asyncio.Event()

        async def heartbeat():
            loop = asyncio.get_running_loop()
            previous = loop.time()
            while not stop.is_set():
                await asyncio.sleep(0.001)
                now = loop.time()
                gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            results = await asyncio.gather(
                *(normalize_gemini_parts(parts) for _ in range(4))
            )
            assert all(result.audio_wav_base64 for result in results)
        finally:
            stop.set()
            await heartbeat_task
        return max(gaps, default=0.0)

    assert asyncio.run(run()) < 0.06


def test_function_thought_signature_over_limit_cannot_expand_into_call_id():
    with pytest.raises(GeminiThoughtSignatureTooLarge):
        asyncio.run(
            normalize_gemini_parts(
                [
                    {
                        "functionCall": {"name": "tool", "args": {}},
                        "thoughtSignature": "s"
                        * (GEMINI_THOUGHT_SIGNATURE_MAX_BYTES + 1),
                    }
                ]
            )
        )
