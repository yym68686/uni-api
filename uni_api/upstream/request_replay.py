from __future__ import annotations

import asyncio
import os
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

import aiofiles
import httpx

from uni_api.admission import AdmissionRejected
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.admission.memory import AdaptiveMemoryGovernor, process_memory_governor
from uni_api.serialization import json

try:
    from uni_api import _uni_api_native
except ImportError:  # pragma: no cover - production images always ship it.
    _uni_api_native = None


_MIB = 1024 * 1024
_DEFAULT_SPOOL_THRESHOLD_BYTES = 1 * _MIB
_DEFAULT_SPOOL_VARIANTS = 4
_DEFAULT_TRANSPORT_CHUNK_BYTES = 256 * 1024
_NATIVE_SERIALIZER_BUFFER_BYTES = 64 * 1024


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _json_wire_upper_bound(value: Any) -> int:
    """Bound compact JSON bytes without creating a serialized value."""

    if value is None:
        return 4
    if isinstance(value, bool):
        return 5
    if isinstance(value, int):
        return len(str(value))
    if isinstance(value, float):
        return 32
    if isinstance(value, str):
        # Six bytes covers every escaped control/BMP character; UTF-8 needs at
        # most four. Quotes are charged separately.
        return 2 + len(value) * 6
    if isinstance(value, dict):
        return 2 + sum(
            _json_wire_upper_bound(str(key))
            + 1
            + _json_wire_upper_bound(item)
            + 1
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return 2 + sum(_json_wire_upper_bound(item) + 1 for item in value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _native_utf8_cache_upper_bound(value: Any) -> int:
    """Bound only UTF-8 caches Rust may cause for non-ASCII Python strings."""

    if isinstance(value, str):
        return 0 if value.isascii() else len(value) * 4
    if isinstance(value, dict):
        return sum(
            _native_utf8_cache_upper_bound(key)
            + _native_utf8_cache_upper_bound(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_native_utf8_cache_upper_bound(item) for item in value)
    return 0


def _serialize_compact_json_bytes(payload: dict[str, Any]) -> bytes:
    # Preserve stdlib JSON's accepted-value behavior, including NaN/Infinity,
    # while eliminating HTTPX's later implicit UTF-8 conversion.
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class UpstreamRequestMemoryExhausted(AdmissionRejected):
    local_admission_rejection = True

    def __init__(self) -> None:
        super().__init__("memory_hard_guard", status_code=503)


class _MemoryLease:
    def __init__(
        self,
        governor: AdaptiveMemoryGovernor,
        category: str,
        size: int,
    ) -> None:
        self._governor = governor
        self._category = category
        self.size = int(size)
        self._released = False

    @classmethod
    def reserve(
        cls,
        governor: AdaptiveMemoryGovernor,
        category: str,
        size: int,
    ) -> _MemoryLease:
        lease = cls(governor, category, max(0, int(size)))
        if lease.size and not governor.reserve_nowait(category, lease.size):
            raise UpstreamRequestMemoryExhausted()
        return lease

    def grow(self, additional_bytes: int) -> None:
        additional_bytes = max(0, int(additional_bytes))
        if not additional_bytes:
            return
        if not self._governor.reserve_nowait(self._category, additional_bytes):
            raise UpstreamRequestMemoryExhausted()
        self.size += additional_bytes

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.size:
            self._governor.release(self._category, self.size)


class DiskReplayByteStream(httpx.AsyncByteStream):
    """One request-body stream that reopens its immutable spool file."""

    def __init__(self, path: Path, *, chunk_bytes: int) -> None:
        self.path = path
        self.chunk_bytes = int(chunk_bytes)
        self._source: Any | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._source is not None:
            raise RuntimeError("disk replay stream cannot be consumed concurrently")
        source = await aiofiles.open(self.path, "rb")
        self._source = source
        try:
            while True:
                chunk = await source.read(self.chunk_bytes)
                if not chunk:
                    return
                yield chunk
        finally:
            if self._source is source:
                self._source = None
            await source.close()

    async def aclose(self) -> None:
        source = self._source
        self._source = None
        if source is not None:
            await source.close()


@dataclass(slots=True)
class PreparedUpstreamBody:
    content: bytes | DiskReplayByteStream
    content_length: int
    storage: str
    path: Path | None
    _serialization_lease: _MemoryLease
    _transport_lease: _MemoryLease

    async def aclose(self) -> None:
        try:
            close = getattr(self.content, "aclose", None)
            if callable(close):
                await close()
        finally:
            self._transport_lease.release()
            self._serialization_lease.release()


@dataclass(frozen=True, slots=True)
class _DiskVariant:
    path: Path
    content_length: int


class MessagesRequestReplayStore:
    """Finite provider-payload cache for one ``/v1/messages`` request."""

    def __init__(
        self,
        request_body_bytes: int,
        *,
        memory_governor: AdaptiveMemoryGovernor = process_memory_governor,
        threshold_bytes: int | None = None,
        max_variants: int | None = None,
        chunk_bytes: int | None = None,
        spool_directory: str | Path | None = None,
    ) -> None:
        self.request_body_bytes = max(0, int(request_body_bytes))
        self.threshold_bytes = int(
            threshold_bytes
            or _env_positive_int(
                "MESSAGES_REQUEST_SPOOL_THRESHOLD_BYTES",
                _DEFAULT_SPOOL_THRESHOLD_BYTES,
            )
        )
        self.max_variants = int(
            max_variants
            or _env_positive_int(
                "MESSAGES_REQUEST_SPOOL_MAX_VARIANTS",
                _DEFAULT_SPOOL_VARIANTS,
            )
        )
        self.chunk_bytes = int(
            chunk_bytes
            or _env_positive_int(
                "MESSAGES_REQUEST_TRANSPORT_CHUNK_BYTES",
                _DEFAULT_TRANSPORT_CHUNK_BYTES,
            )
        )
        self.spool_directory = Path(
            spool_directory
            or os.getenv(
                "MESSAGES_REQUEST_SPOOL_DIRECTORY",
                "/tmp/uni-api-messages-replay",
            )
        )
        self._memory_governor = memory_governor
        self._variants: OrderedDict[Hashable, _DiskVariant] = OrderedDict()
        self._closed = False

    @property
    def uses_disk(self) -> bool:
        return self.request_body_bytes >= self.threshold_bytes

    async def prepare(
        self,
        variant_key: Hashable,
        payload: dict[str, Any],
    ) -> PreparedUpstreamBody:
        if self._closed:
            raise RuntimeError("messages replay store is closed")
        if self.uses_disk:
            return await self._prepare_disk(variant_key, payload)
        return await self._prepare_memory(payload)

    async def _prepare_memory(
        self,
        payload: dict[str, Any],
    ) -> PreparedUpstreamBody:
        # Pre-reserve compact UTF-8 serialization plus bounded HTTPX/kernel
        # transport storage. Parsed-body ownership remains charged separately.
        serialized_upper = await asyncio.to_thread(
            _json_wire_upper_bound,
            payload,
        )
        serialization_lease = _MemoryLease.reserve(
            self._memory_governor,
            "upstream_serialized_body",
            serialized_upper * 2,
        )
        try:
            transport_lease = _MemoryLease.reserve(
                self._memory_governor,
                "upstream_transport_buffer",
                self.chunk_bytes * 2,
            )
        except BaseException:
            serialization_lease.release()
            raise
        try:
            content = await run_json_cpu(
                _serialize_compact_json_bytes,
                payload,
            )
            if len(content) * 2 > serialization_lease.size:
                serialization_lease.grow(
                    len(content) * 2 - serialization_lease.size
                )
            return PreparedUpstreamBody(
                content=content,
                content_length=len(content),
                storage="memory",
                path=None,
                _serialization_lease=serialization_lease,
                _transport_lease=transport_lease,
            )
        except BaseException:
            transport_lease.release()
            serialization_lease.release()
            raise

    async def _prepare_disk(
        self,
        variant_key: Hashable,
        payload: dict[str, Any],
    ) -> PreparedUpstreamBody:
        variant = self._variants.get(variant_key)
        serializer_bytes = 0
        if variant is None:
            serializer_bytes = (
                _NATIVE_SERIALIZER_BUFFER_BYTES
                + await asyncio.to_thread(
                    _native_utf8_cache_upper_bound,
                    payload,
                )
            )
        serialization_lease = _MemoryLease.reserve(
            self._memory_governor,
            "upstream_serialized_body",
            serializer_bytes,
        )
        try:
            transport_lease = _MemoryLease.reserve(
                self._memory_governor,
                "upstream_transport_buffer",
                self.chunk_bytes * 2,
            )
        except BaseException:
            serialization_lease.release()
            raise
        try:
            if variant is None:
                variant = await self._materialize_variant(
                    payload,
                    serialization_lease,
                )
                while len(self._variants) >= self.max_variants:
                    _, evicted = self._variants.popitem(last=False)
                    evicted.path.unlink(missing_ok=True)
                self._variants[variant_key] = variant
            else:
                self._variants.move_to_end(variant_key)
            return PreparedUpstreamBody(
                content=DiskReplayByteStream(
                    variant.path,
                    chunk_bytes=self.chunk_bytes,
                ),
                content_length=variant.content_length,
                storage="disk",
                path=variant.path,
                _serialization_lease=serialization_lease,
                _transport_lease=transport_lease,
            )
        except BaseException:
            transport_lease.release()
            serialization_lease.release()
            raise

    async def _materialize_variant(
        self,
        payload: dict[str, Any],
        lease: _MemoryLease,
    ) -> _DiskVariant:
        self.spool_directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix="messages-",
            suffix=".json",
            dir=self.spool_directory,
        )
        os.close(descriptor)
        path = Path(raw_path)
        try:
            native_writer = getattr(_uni_api_native, "write_json_file", None)
            if callable(native_writer):
                content_length = await run_json_cpu(
                    native_writer,
                    payload,
                    str(path),
                )
            else:
                # Development-only fallback. Production images use Rust's
                # fixed 64 KiB direct-to-file serializer above.
                fallback_reservation = 2 * await asyncio.to_thread(
                    _json_wire_upper_bound,
                    payload,
                )
                if fallback_reservation > lease.size:
                    lease.grow(fallback_reservation - lease.size)
                encoded = await run_json_cpu(
                    _serialize_compact_json_bytes,
                    payload,
                )
                if len(encoded) * 2 > lease.size:
                    lease.grow(len(encoded) * 2 - lease.size)
                await asyncio.to_thread(path.write_bytes, encoded)
                content_length = len(encoded)
                encoded = None
            return _DiskVariant(path=path, content_length=int(content_length))
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        variants = tuple(self._variants.values())
        self._variants.clear()
        for variant in variants:
            variant.path.unlink(missing_ok=True)
