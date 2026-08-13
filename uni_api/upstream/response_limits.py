from __future__ import annotations

import asyncio
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from uni_api.admission.cpu import run_cpu_phase
from uni_api.admission.resources import startup_cpu_worker_count
from uni_api.observability.threadpool_tasks import register_dedicated_threadpool


DEFAULT_MAX_UPSTREAM_ERROR_BODY_BYTES = 256 * 1024
DEFAULT_MAX_UPSTREAM_SUCCESS_BODY_BYTES = 64 * 1024 * 1024
UPSTREAM_ERROR_BODY_MAX_BYTES_ENV = "UPSTREAM_ERROR_BODY_MAX_BYTES"
UPSTREAM_SUCCESS_BODY_MAX_BYTES_ENV = "UPSTREAM_SUCCESS_BODY_MAX_BYTES"
UPSTREAM_JSON_MEMORY_RESERVATION_MULTIPLIER_ENV = (
    "UPSTREAM_JSON_MEMORY_RESERVATION_MULTIPLIER"
)
_DECODE_CHUNK_BYTES = 64 * 1024
_ENCODED_OVERHEAD_ALLOWANCE_BYTES = 64 * 1024
_DEFAULT_UPSTREAM_RESPONSE_CPU_WORKERS = startup_cpu_worker_count()
try:
    UPSTREAM_RESPONSE_CPU_WORKERS = max(
        1,
        int(
            os.getenv(
                "UPSTREAM_RESPONSE_CPU_WORKERS",
                str(_DEFAULT_UPSTREAM_RESPONSE_CPU_WORKERS),
            )
            or str(_DEFAULT_UPSTREAM_RESPONSE_CPU_WORKERS)
        ),
    )
except (TypeError, ValueError):
    UPSTREAM_RESPONSE_CPU_WORKERS = _DEFAULT_UPSTREAM_RESPONSE_CPU_WORKERS
_UPSTREAM_RESPONSE_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=UPSTREAM_RESPONSE_CPU_WORKERS,
    thread_name_prefix="uni-api-upstream-body",
)
register_dedicated_threadpool(
    "upstream_response_decode",
    _UPSTREAM_RESPONSE_CPU_EXECUTOR,
)


async def _run_response_cpu(callback, *args):
    """Run bounded decoder work without releasing ownership on cancellation."""
    return await run_cpu_phase(
        _UPSTREAM_RESPONSE_CPU_EXECUTOR,
        callback,
        *args,
        phase="upstream_response",
    )


class UpstreamResponseDecodingError(RuntimeError):
    status_code = 502
    reason = "upstream_response_decoding_error"


class UpstreamResponseEncodingUnsupported(RuntimeError):
    status_code = 502
    reason = "upstream_response_encoding_unsupported"

    def __init__(self, encoding: str) -> None:
        super().__init__(f"unsupported upstream content encoding: {encoding}")
        self.encoding = encoding


def upstream_error_body_max_bytes() -> int:
    raw = os.getenv(UPSTREAM_ERROR_BODY_MAX_BYTES_ENV)
    if raw is None:
        return DEFAULT_MAX_UPSTREAM_ERROR_BODY_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_UPSTREAM_ERROR_BODY_BYTES
    return max(1, value)


def upstream_success_body_max_bytes() -> int:
    raw = os.getenv(UPSTREAM_SUCCESS_BODY_MAX_BYTES_ENV)
    if raw is None:
        return DEFAULT_MAX_UPSTREAM_SUCCESS_BODY_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_UPSTREAM_SUCCESS_BODY_BYTES
    return max(1, value)


def upstream_json_memory_reservation_multiplier() -> int:
    raw = os.getenv(UPSTREAM_JSON_MEMORY_RESERVATION_MULTIPLIER_ENV)
    if raw is None:
        return 8
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 8
    return max(1, value)


@dataclass(frozen=True, slots=True)
class LimitedResponseBody:
    body: bytes
    truncated: bool
    observed_bytes_at_least: int

    def text(self) -> str:
        value = self.body.decode("utf-8", errors="replace")
        if self.truncated:
            return f"{value}\n...[upstream error body truncated]"
        return value


async def read_limited_response_body(
    response: Any,
    *,
    max_bytes: int | None = None,
    reserve_bytes: Callable[[int], Awaitable[Any]] | None = None,
    reservation_multiplier: int = 1,
    resource_governed: bool = False,
) -> LimitedResponseBody:
    """Incrementally read a response body without an unbounded ``aread``.

    By default this is intended for diagnostics/error bodies and closes the
    response when the configured limit is crossed.  ``resource_governed``
    removes the static byte ceiling; callers can instead use ``reserve_bytes``
    to admit every retained chunk against current runtime resources.
    """

    if resource_governed and max_bytes is not None:
        raise ValueError(
            "max_bytes cannot be combined with resource_governed"
        )
    if resource_governed and reserve_bytes is None:
        raise ValueError(
            "resource_governed requires a reserve_bytes callback"
        )
    if resource_governed:
        limit = None
    elif max_bytes is None:
        limit = upstream_error_body_max_bytes()
    else:
        limit = int(max_bytes)
    if limit is not None and limit <= 0:
        raise ValueError("max_bytes must be positive")
    if reservation_multiplier <= 0:
        raise ValueError("reservation_multiplier must be positive")
    extensions = getattr(response, "extensions", None)
    if isinstance(extensions, dict) and extensions.get(
        "uni_api_body_already_bounded"
    ):
        content = bytes(getattr(response, "content", b""))
        return LimitedResponseBody(
            body=content if limit is None else content[:limit],
            truncated=limit is not None and len(content) > limit,
            observed_bytes_at_least=len(content),
        )
    headers = getattr(response, "headers", None)
    content_encoding = ""
    if headers is not None:
        try:
            content_encoding = str(headers.get("content-encoding") or "")
        except Exception:
            content_encoding = ""
    encoding = content_encoding.strip().lower()
    if encoding in {"", "identity"}:
        decompressor = None
    elif encoding in {"gzip", "x-gzip"}:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        close = getattr(response, "aclose", None)
        if callable(close):
            await close()
        raise UpstreamResponseEncodingUnsupported(encoding)

    # HTTPX's aiter_bytes() transparently decompresses and may materialize an
    # arbitrarily large decoded chunk before our limit sees it.  Real HTTPX
    # streaming responses expose aiter_raw(); controlled test doubles may only
    # provide aiter_bytes(), which is safe for identity bodies.
    iterator_factory = getattr(response, "aiter_raw", None)
    if bool(getattr(response, "is_stream_consumed", False)):
        # MockTransport and preloaded Response objects have already materialized
        # identity content.  Production network responses enter this helper
        # unconsumed and therefore always take the raw path above.
        if decompressor is not None:
            await getattr(response, "aclose")()
            raise UpstreamResponseEncodingUnsupported(encoding)
        iterator_factory = getattr(response, "aiter_bytes", None)
    if not callable(iterator_factory):
        if decompressor is not None:
            close = getattr(response, "aclose", None)
            if callable(close):
                await close()
            raise UpstreamResponseEncodingUnsupported(encoding)
        iterator_factory = getattr(response, "aiter_bytes", None)
    if not callable(iterator_factory):
        raise TypeError("streaming response must provide aiter_raw() or aiter_bytes()")

    body = bytearray()
    observed = 0
    raw_observed = 0
    truncated = False

    async def close_response() -> None:
        close = getattr(response, "aclose", None)
        if callable(close):
            await close()

    async def retain(decoded: bytes) -> bool:
        nonlocal observed, truncated
        observed += len(decoded)
        remaining = None if limit is None else limit - len(body)
        retained = (
            decoded
            if remaining is None
            else decoded[: max(0, remaining)]
        )
        if retained and reserve_bytes is not None:
            await reserve_bytes(len(retained) * reservation_multiplier)
        if retained:
            body.extend(retained)
        if remaining is not None and len(decoded) > remaining:
            truncated = True
            return False
        return True

    try:
        async for raw_chunk in iterator_factory():
            if isinstance(raw_chunk, str):
                raw = raw_chunk.encode("utf-8")
            else:
                raw = bytes(raw_chunk)
            raw_observed += len(raw)
            if (
                decompressor is not None
                and limit is not None
                and raw_observed
                > limit + _ENCODED_OVERHEAD_ALLOWANCE_BYTES
            ):
                # Bound compressed CPU/input work independently.  This also
                # avoids repeatedly copying a giant unconsumed_tail from a
                # non-network test transport that yields one enormous chunk.
                truncated = True
                await close_response()
                break
            if decompressor is None:
                for offset in range(0, len(raw), _DECODE_CHUNK_BYTES):
                    if not await retain(
                        raw[offset : offset + _DECODE_CHUNK_BYTES]
                    ):
                        break
            else:
                for input_offset in range(0, len(raw), _DECODE_CHUNK_BYTES):
                    pending = raw[
                        input_offset : input_offset + _DECODE_CHUNK_BYTES
                    ]
                    while pending and not truncated:
                        max_output = min(
                            _DECODE_CHUNK_BYTES,
                            max(1, limit - len(body) + 1)
                            if limit is not None
                            else _DECODE_CHUNK_BYTES,
                        )
                        previous_pending = len(pending)
                        # zlib work is CPU-bound.  More importantly, max_output
                        # makes every pre-reservation allocation small and fixed:
                        # admission runs before decoded bytes enter ``body``.
                        decoded = await _run_response_cpu(
                            decompressor.decompress,
                            pending,
                            max_output,
                        )
                        pending = decompressor.unconsumed_tail
                        if decoded and not await retain(decoded):
                            break
                        if (
                            pending
                            and not decoded
                            and len(pending) >= previous_pending
                        ):
                            raise UpstreamResponseDecodingError(
                                "upstream compressed response decoder made no progress"
                            )
                    if truncated:
                        break
                if decompressor.unused_data:
                    raise UpstreamResponseDecodingError(
                        "upstream compressed response has trailing data"
                    )
            if truncated:
                await close_response()
                break
        else:
            if decompressor is not None:
                if not decompressor.eof:
                    raise UpstreamResponseDecodingError(
                        "upstream compressed response ended before frame completion"
                    )
                while True:
                    max_output = min(
                        _DECODE_CHUNK_BYTES,
                        max(1, limit - len(body) + 1)
                        if limit is not None
                        else _DECODE_CHUNK_BYTES,
                    )
                    tail = await _run_response_cpu(
                        decompressor.flush,
                        max_output,
                    )
                    if not tail:
                        break
                    if not await retain(tail):
                        await close_response()
                        break
    except zlib.error as exc:
        await close_response()
        raise UpstreamResponseDecodingError(
            "invalid upstream compressed response"
        ) from exc
    except BaseException:
        await close_response()
        raise

    if truncated and limit is not None and observed <= limit:
        # We may stop because encoded input itself crossed the hard cap before
        # the decoder produced another byte.
        observed = limit + 1

    return LimitedResponseBody(
        body=bytes(body),
        truncated=truncated,
        observed_bytes_at_least=observed,
    )
