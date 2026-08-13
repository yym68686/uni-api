from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from uni_api.admission.json_parsing import run_json_cpu


# Keep each GIL-holding binascii operation short.  These values preserve the
# base64 4-character / 3-byte boundaries and cap transient per-task copies.
BASE64_DECODE_CHUNK_CHARS = 64 * 1024
BASE64_ENCODE_CHUNK_BYTES = 48 * 1024


class ChunkedBase64Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Base64Inspection:
    decoded_bytes: int
    digest_hex: str
    prefix: bytes
    encoded_payload: str | None


def _strict_decode_chunk(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise ChunkedBase64Error("invalid base64 payload") from exc


def _encode_chunk(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def predicted_decoded_base64_bytes(value: str) -> int:
    """Return the decoded length implied by a potentially unpadded payload."""

    if not isinstance(value, str) or not value or len(value) % 4 == 1:
        raise ChunkedBase64Error("invalid base64 length")
    existing_padding = len(value) - len(value.rstrip("="))
    if existing_padding > 2:
        raise ChunkedBase64Error("invalid base64 padding")
    added_padding = (-len(value)) % 4
    return ((len(value) + added_padding) // 4) * 3 - (
        existing_padding + added_padding
    )


async def iter_strict_base64_decoded_chunks(
    value: str,
    *,
    max_encoded_chars: int | None,
    max_decoded_bytes: int | None,
    start_index: int = 0,
) -> AsyncIterator[bytes]:
    """Strictly decode base64 in bounded work chunks."""

    payload_length = len(value) - start_index if isinstance(value, str) else -1
    if (
        not isinstance(value, str)
        or start_index < 0
        or payload_length <= 0
        or (
            max_encoded_chars is not None
            and payload_length > max_encoded_chars
        )
        or payload_length % 4 == 1
    ):
        raise ChunkedBase64Error("invalid or oversized base64 payload")

    total_decoded = 0
    value_length = len(value)
    for offset in range(
        start_index,
        value_length,
        BASE64_DECODE_CHUNK_CHARS,
    ):
        end = min(value_length, offset + BASE64_DECODE_CHUNK_CHARS)
        original_chunk = value[offset:end]
        is_final = end == value_length
        if not is_final and "=" in original_chunk:
            raise ChunkedBase64Error("base64 padding before final chunk")
        decode_chunk = original_chunk
        if is_final:
            decode_chunk += "=" * ((-len(decode_chunk)) % 4)
        decoded_chunk = await run_json_cpu(_strict_decode_chunk, decode_chunk)
        total_decoded += len(decoded_chunk)
        if (
            max_decoded_bytes is not None
            and total_decoded > max_decoded_bytes
        ):
            decoded_chunk = None
            raise ChunkedBase64Error("decoded base64 payload is too large")
        yield decoded_chunk
        decoded_chunk = None


async def inspect_base64_chunks(
    value: str,
    *,
    max_encoded_chars: int | None,
    max_decoded_bytes: int | None,
    prefix_bytes: int = 0,
    collect_encoded_payload: bool = False,
    start_index: int = 0,
) -> Base64Inspection:
    """Validate, count and hash decoded bytes with bounded event-loop stalls."""

    digest = hashlib.sha256()
    prefix = bytearray()
    decoded_bytes = 0
    encoded_parts: list[str] | None = [] if collect_encoded_payload else None
    encoded_offset = start_index

    async for decoded_chunk in iter_strict_base64_decoded_chunks(
        value,
        max_encoded_chars=max_encoded_chars,
        max_decoded_bytes=max_decoded_bytes,
        start_index=start_index,
    ):
        if encoded_parts is not None:
            encoded_end = min(
                len(value),
                encoded_offset + BASE64_DECODE_CHUNK_CHARS,
            )
            encoded_parts.append(value[encoded_offset:encoded_end])
            encoded_offset = encoded_end
        decoded_bytes += len(decoded_chunk)
        digest.update(decoded_chunk)
        if len(prefix) < prefix_bytes:
            prefix.extend(decoded_chunk[: prefix_bytes - len(prefix)])
        decoded_chunk = None

    encoded_payload = None
    if encoded_parts is not None:
        # The final unavoidable copy is linear and does no base64 work.  The
        # expensive encode/decode operations above remain finely interleaved.
        encoded_payload = "".join(encoded_parts)
        await asyncio.sleep(0)
        encoded_parts.clear()
    return Base64Inspection(
        decoded_bytes=decoded_bytes,
        digest_hex=digest.hexdigest(),
        prefix=bytes(prefix),
        encoded_payload=encoded_payload,
    )


async def encode_bytes_base64_chunks(
    value: bytes,
    *,
    prefix: str = "",
) -> str:
    """Encode bytes using bounded, cancellation-safe base64 work units."""

    encoded_parts: list[str] = [prefix] if prefix else []
    for offset in range(0, len(value), BASE64_ENCODE_CHUNK_BYTES):
        chunk = value[offset : offset + BASE64_ENCODE_CHUNK_BYTES]
        encoded_parts.append(await run_json_cpu(_encode_chunk, chunk))
        chunk = None
    encoded = "".join(encoded_parts)
    await asyncio.sleep(0)
    encoded_parts.clear()
    return encoded


async def encode_byte_stream_base64_chunks(
    chunks: AsyncIterator[bytes],
    *,
    prefix: bytes = b"",
) -> str:
    """Encode an async byte stream while preserving 3-byte boundaries."""

    pending = bytes(prefix)
    encoded_parts: list[str] = []
    async for chunk in chunks:
        combined = pending + chunk
        complete_bytes = (len(combined) // 3) * 3
        if complete_bytes:
            encoded_parts.append(
                await run_json_cpu(
                    _encode_chunk,
                    combined[:complete_bytes],
                )
            )
        pending = combined[complete_bytes:]
        combined = None
        chunk = None
    if pending:
        encoded_parts.append(await run_json_cpu(_encode_chunk, pending))
        pending = b""
    encoded = "".join(encoded_parts)
    await asyncio.sleep(0)
    encoded_parts.clear()
    return encoded
