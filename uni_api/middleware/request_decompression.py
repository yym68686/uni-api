from __future__ import annotations

import io
import os
from typing import Iterable

import zstandard as zstd
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


DEFAULT_MAX_ZSTD_REQUEST_BODY_BYTES = 64 * 1024 * 1024


class RequestBodyDecompressionMiddleware:
    """Decode supported compressed request bodies before FastAPI reads them."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int | None = None) -> None:
        self.app = app
        if max_body_bytes is None:
            max_body_bytes = _env_int(
                "ZSTD_REQUEST_MAX_BODY_BYTES",
                DEFAULT_MAX_ZSTD_REQUEST_BODY_BYTES,
            )
        self.max_body_bytes = max(0, int(max_body_bytes))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        encodings = _content_encodings(scope.get("headers") or [])
        if not encodings or _is_identity(encodings):
            await self.app(scope, receive, send)
            return
        if encodings != ["zstd"]:
            await _json_error(
                scope,
                receive,
                send,
                415,
                f"unsupported content encoding: {', '.join(encodings)}",
            )
            return

        try:
            compressed = await _read_body(receive, self.max_body_bytes)
            body = _decompress_zstd(compressed, self.max_body_bytes)
        except RequestBodyTooLarge:
            await _json_error(scope, receive, send, 413, "request body too large")
            return
        except zstd.ZstdError:
            await _json_error(scope, receive, send, 400, "invalid zstd body")
            return

        decompressed_scope = dict(scope)
        decompressed_scope["headers"] = [
            (name, value)
            for name, value in (scope.get("headers") or [])
            if name.lower() not in {b"content-encoding", b"content-length"}
        ]

        body_sent = False

        async def decompressed_receive() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(decompressed_scope, decompressed_receive, send)


class RequestBodyTooLarge(Exception):
    pass


async def _json_error(scope: Scope, receive: Receive, send: Send, status_code: int, detail: str) -> None:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    await response(scope, receive, send)


async def _read_body(receive: Receive, max_body_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"".join(chunks)
        chunk = message.get("body", b"")
        if chunk:
            total += len(chunk)
            if total > max_body_bytes:
                raise RequestBodyTooLarge()
            chunks.append(chunk)
        more_body = bool(message.get("more_body", False))
    return b"".join(chunks)


def _decompress_zstd(body: bytes, max_body_bytes: int) -> bytes:
    decoder = zstd.ZstdDecompressor()
    with decoder.stream_reader(io.BytesIO(body)) as reader:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = reader.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_body_bytes:
                raise RequestBodyTooLarge()
            chunks.append(chunk)
    return b"".join(chunks)


def _content_encodings(headers: Iterable[tuple[bytes, bytes]]) -> list[str]:
    values = [
        value.decode("latin-1")
        for name, value in headers
        if name.lower() == b"content-encoding"
    ]
    encodings: list[str] = []
    for value in values:
        encodings.extend(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        )
    return encodings


def _is_identity(encodings: list[str]) -> bool:
    return all(encoding == "identity" for encoding in encodings)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default
