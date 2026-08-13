import asyncio
import random
import threading
import zlib

import pytest

import uni_api.upstream.response_limits as response_limits
from uni_api.upstream.response_limits import (
    UpstreamResponseDecodingError,
    UpstreamResponseEncodingUnsupported,
    read_limited_response_body,
)


class _ChunkedResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = 0
        self.produced = 0

    async def aiter_bytes(self):
        for chunk in self.chunks:
            self.produced += 1
            yield chunk

    async def aclose(self):
        self.closed += 1


class _RawChunkedResponse(_ChunkedResponse):
    def __init__(self, chunks, *, content_encoding=""):
        super().__init__(chunks)
        self.headers = {"content-encoding": content_encoding}
        self.is_stream_consumed = False

    async def aiter_raw(self):
        for chunk in self.chunks:
            self.produced += 1
            yield chunk


def test_limited_error_body_stops_and_closes_at_byte_limit():
    async def scenario():
        response = _ChunkedResponse([b"abcd", b"efgh", b"never-read"])
        result = await read_limited_response_body(response, max_bytes=6)

        assert result.body == b"abcdef"
        assert result.truncated is True
        assert result.observed_bytes_at_least == 8
        assert response.produced == 2
        assert response.closed == 1
        assert result.text().endswith("...[upstream error body truncated]")

    asyncio.run(scenario())


def test_limited_error_body_preserves_complete_body_without_early_close():
    async def scenario():
        response = _ChunkedResponse([b"abc", b"def"])
        result = await read_limited_response_body(response, max_bytes=6)

        assert result.body == b"abcdef"
        assert result.truncated is False
        assert response.produced == 2
        assert response.closed == 0
        assert result.text() == "abcdef"

    asyncio.run(scenario())


def test_gzip_body_is_decoded_in_bounded_reserved_chunks():
    async def scenario():
        payload = b"a" * 100_000
        compressed = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressed.compress(payload) + compressed.flush()
        response = _RawChunkedResponse(
            [encoded[:7], encoded[7:]],
            content_encoding="gzip",
        )
        reservations = []

        async def reserve(size):
            reservations.append(size)

        result = await read_limited_response_body(
            response,
            max_bytes=len(payload),
            reserve_bytes=reserve,
        )

        assert result.body == payload
        assert result.truncated is False
        assert sum(reservations) == len(payload)
        assert max(reservations) <= 64 * 1024
        assert response.closed == 0

    asyncio.run(scenario())


def test_resource_governed_body_has_no_static_byte_ceiling():
    async def scenario():
        payload = b"a" * (300 * 1024)
        compressed = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressed.compress(payload) + compressed.flush()
        response = _RawChunkedResponse(
            [encoded],
            content_encoding="gzip",
        )
        reservations = []

        async def reserve(size):
            reservations.append(size)

        result = await read_limited_response_body(
            response,
            reserve_bytes=reserve,
            resource_governed=True,
        )

        assert result.body == payload
        assert result.truncated is False
        assert sum(reservations) == len(payload)
        assert max(reservations) <= 64 * 1024
        assert response.closed == 0

    asyncio.run(scenario())


def test_resource_governed_body_requires_admission_callback():
    async def scenario():
        response = _ChunkedResponse([b"payload"])
        with pytest.raises(
            ValueError,
            match="requires a reserve_bytes callback",
        ):
            await read_limited_response_body(
                response,
                resource_governed=True,
            )

    asyncio.run(scenario())


def test_incompressible_gzip_at_exact_decoded_limit_is_not_false_truncated():
    async def scenario():
        payload = random.Random(7).randbytes(100_000)
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressor.compress(payload) + compressor.flush()
        assert len(encoded) > len(payload)
        response = _RawChunkedResponse([encoded], content_encoding="gzip")

        result = await read_limited_response_body(
            response,
            max_bytes=len(payload),
        )

        assert result.body == payload
        assert result.truncated is False
        assert response.closed == 0

    asyncio.run(scenario())


def test_gzip_bomb_stops_at_limit_plus_one_and_closes():
    async def scenario():
        limit = 100_000
        payload = b"z" * (limit + 1)
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressor.compress(payload) + compressor.flush()
        response = _RawChunkedResponse([encoded], content_encoding="x-gzip")
        reservations = []

        async def reserve(size):
            reservations.append(size)

        result = await read_limited_response_body(
            response,
            max_bytes=limit,
            reserve_bytes=reserve,
        )

        assert result.body == payload[:limit]
        assert result.truncated is True
        assert result.observed_bytes_at_least == limit + 1
        assert sum(reservations) == limit
        assert max(reservations) <= 64 * 1024
        assert response.closed == 1

    asyncio.run(scenario())


def test_zlib_deflate_body_is_decoded_and_truncated_frames_are_rejected():
    async def scenario():
        payload = (b"deflate-" * 4096) + b"done"
        encoded = zlib.compress(payload)
        complete = _RawChunkedResponse(
            [encoded],
            content_encoding="deflate",
        )
        result = await read_limited_response_body(
            complete,
            max_bytes=len(payload),
        )
        assert result.body == payload
        assert result.truncated is False

        truncated = _RawChunkedResponse(
            [encoded[:-2]],
            content_encoding="deflate",
        )
        with pytest.raises(UpstreamResponseDecodingError):
            await read_limited_response_body(
                truncated,
                max_bytes=len(payload),
            )
        assert truncated.closed == 1

    asyncio.run(scenario())


def test_compressed_trailing_data_and_unsupported_encoding_are_rejected():
    async def scenario():
        payload = b"payload"
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressor.compress(payload) + compressor.flush()
        trailing = _RawChunkedResponse(
            [encoded + b"unexpected"],
            content_encoding="gzip",
        )
        with pytest.raises(UpstreamResponseDecodingError):
            await read_limited_response_body(trailing, max_bytes=1024)
        assert trailing.closed == 1

        unsupported = _RawChunkedResponse(
            [b"encoded"],
            content_encoding="br",
        )
        with pytest.raises(UpstreamResponseEncodingUnsupported):
            await read_limited_response_body(unsupported, max_bytes=1024)
        assert unsupported.closed == 1

    asyncio.run(scenario())


def test_reservation_failure_happens_before_retention_and_closes_response():
    async def scenario():
        payload = b"x" * (128 * 1024)
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressor.compress(payload) + compressor.flush()
        response = _RawChunkedResponse([encoded], content_encoding="gzip")
        reserved = []

        async def reject(size):
            reserved.append(size)
            raise RuntimeError("budget exhausted")

        with pytest.raises(RuntimeError, match="budget exhausted"):
            await read_limited_response_body(
                response,
                max_bytes=len(payload),
                reserve_bytes=reject,
            )
        assert reserved and reserved[0] <= 64 * 1024
        assert response.closed == 1

    asyncio.run(scenario())


def test_decoder_cancellation_waits_for_worker_before_closing_response(monkeypatch):
    async def scenario():
        started = threading.Event()
        release = threading.Event()

        class BlockingDecompressor:
            unconsumed_tail = b""
            unused_data = b""
            eof = True

            def decompress(self, _pending, _max_output):
                started.set()
                release.wait(timeout=2)
                return b"decoded"

            def flush(self, _max_output):
                return b""

        monkeypatch.setattr(
            response_limits.zlib,
            "decompressobj",
            lambda *_args, **_kwargs: BlockingDecompressor(),
        )
        response = _RawChunkedResponse([b"encoded"], content_encoding="gzip")
        task = asyncio.create_task(
            read_limited_response_body(response, max_bytes=1024)
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert response.closed == 0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.closed == 1

    asyncio.run(scenario())
