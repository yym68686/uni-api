from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from uni_api.admission.cpu import run_cpu_phase
from uni_api.admission.core import get_request_admission_lease
from uni_api.admission.resources import startup_cpu_worker_count
from uni_api.admission.json_memory import (
    JSONMemoryComplexityError,
    estimate_json_memory_bytes,
    estimate_json_text_memory_bytes,
)
from uni_api.observability.threadpool_tasks import register_dedicated_threadpool
from uni_api.serialization import json


DEFAULT_JSON_PARSE_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
_JSON_PARSE_OFFLOAD_THRESHOLD_BYTES = 64 * 1024
_DEFAULT_JSON_PARSE_CPU_WORKERS = startup_cpu_worker_count()
try:
    JSON_PARSE_CPU_WORKERS = max(
        1,
        int(
            os.getenv(
                "JSON_PARSE_CPU_WORKERS",
                str(_DEFAULT_JSON_PARSE_CPU_WORKERS),
            )
            or str(_DEFAULT_JSON_PARSE_CPU_WORKERS)
        ),
    )
except (TypeError, ValueError):
    JSON_PARSE_CPU_WORKERS = _DEFAULT_JSON_PARSE_CPU_WORKERS
_JSON_PARSE_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=JSON_PARSE_CPU_WORKERS,
    thread_name_prefix="uni-api-json",
)
register_dedicated_threadpool("json_parse", _JSON_PARSE_CPU_EXECUTOR)


class ReusableJSONParseWorkspace:
    """One high-water temporary-memory reservation reused by a stream."""

    def __init__(self, reservation: Any | None) -> None:
        self._reservation = reservation
        self._capacity = 0
        self._closed = False

    @classmethod
    async def create(cls) -> ReusableJSONParseWorkspace:
        request_lease = get_request_admission_lease()
        reservation = (
            await request_lease.reserve_temporary_response_bytes(0)
            if request_lease is not None
            else None
        )
        return cls(reservation)

    @property
    def capacity(self) -> int:
        return self._capacity

    async def ensure(self, required_bytes: int) -> int:
        required_bytes = int(required_bytes)
        if required_bytes < 0:
            raise ValueError("required_bytes cannot be negative")
        if self._closed:
            raise RuntimeError("JSON parse workspace is closed")
        if required_bytes <= self._capacity:
            return self._capacity
        reservation = self._reservation
        if reservation is not None:
            await reservation.reserve(required_bytes - self._capacity)
        self._capacity = required_bytes
        return self._capacity

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        reservation = self._reservation
        self._reservation = None
        if reservation is not None:
            await reservation.release()


async def _finish_owner_cleanup_despite_cancellation(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def run_json_cpu(callback, *args, **kwargs):
    """Run bounded JSON CPU work without releasing ownership on cancellation."""
    return await run_cpu_phase(
        _JSON_PARSE_CPU_EXECUTOR,
        callback,
        *args,
        phase="json",
        **kwargs,
    )


class OwnedJSONValue:
    """A materialized JSON graph coupled to its exact admission ownership."""

    def __init__(self, value: Any, reservation: Any | None) -> None:
        self._value = value
        self._reservation = reservation
        self._reservation_transferred = False
        self._closed = False
        self._closing = False
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def value(self) -> Any:
        if self._closed or self._closing:
            raise RuntimeError("owned JSON value is closed")
        return self._value

    @property
    def can_close_nowait(self) -> bool:
        return bool(
            not self._closed
            and not self._closing
            and self._reservation is None
            and self._close_task is None
            and not self._lock.locked()
        )

    def close_nowait(self) -> bool:
        if not self.can_close_nowait:
            return False
        self._closing = True
        self._value = None
        self._closed = True
        self._closing = False
        return True

    def take_reservation(self):
        """Atomically transfer the live graph charge exactly once.

        This operation deliberately contains no await point.  State either
        remains attached to this owner or the caller synchronously receives
        the token; task cancellation cannot strand it between those states.
        """

        if self._closed or self._closing:
            raise RuntimeError("owned JSON value is closed")
        if self._reservation_transferred:
            raise RuntimeError("owned JSON reservation was already transferred")
        reservation = self._reservation
        self._reservation = None
        self._reservation_transferred = True
        return reservation

    async def aclose(self) -> None:
        if self._closed:
            return
        if self.close_nowait():
            return
        if self._close_task is None:
            # Establish exact-once cleanup synchronously, before cancellation
            # can strike the first lock acquisition.
            self._closing = True
            self._close_task = asyncio.create_task(self._close_once())
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await _finish_owner_cleanup_despite_cancellation(close_task)
            raise

    async def _close_once(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            # Drop our graph reference before returning the corresponding
            # memory charge.  Consumers must likewise clear aliases before
            # closing or explicitly transfer the reservation.
            self._value = None
            reservation = self._reservation
            self._reservation = None
        if reservation is not None:
            await reservation.release()

    async def __aenter__(self) -> OwnedJSONValue:
        if self._closed or self._closing:
            raise RuntimeError("owned JSON value is closed")
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


async def parse_owned_json_value(
    payload: str | bytes | bytearray | memoryview,
    *,
    max_estimated_bytes: int = DEFAULT_JSON_PARSE_MAX_ESTIMATED_BYTES,
    allow_invalid: bool = False,
    workspace: ReusableJSONParseWorkspace | None = None,
    workspace_extra_bytes: int = 0,
) -> OwnedJSONValue:
    """Parse untrusted JSON and return explicit, transferable ownership."""

    payload_is_memoryview = isinstance(payload, memoryview)
    offload = (
        len(payload) >= _JSON_PARSE_OFFLOAD_THRESHOLD_BYTES // 4
        if isinstance(payload, str)
        else len(payload) >= _JSON_PARSE_OFFLOAD_THRESHOLD_BYTES
    )
    if isinstance(payload, str):
        estimate = (
            await run_json_cpu(
                estimate_json_text_memory_bytes,
                payload,
                raw_memory_multiplier=4,
                token_memory_bytes=128,
                max_estimated_bytes=max_estimated_bytes,
            )
            if offload
            else estimate_json_text_memory_bytes(
                payload,
                raw_memory_multiplier=4,
                token_memory_bytes=128,
                max_estimated_bytes=max_estimated_bytes,
            )
        )
    else:
        estimate = (
            await run_json_cpu(
                estimate_json_memory_bytes,
                payload,
                raw_memory_multiplier=4,
                token_memory_bytes=128,
                max_estimated_bytes=max_estimated_bytes,
            )
            if offload
            else estimate_json_memory_bytes(
                payload,
                raw_memory_multiplier=4,
                token_memory_bytes=128,
                max_estimated_bytes=max_estimated_bytes,
            )
        )

    if workspace_extra_bytes < 0:
        raise ValueError("workspace_extra_bytes cannot be negative")
    if workspace is not None:
        await workspace.ensure(
            workspace_extra_bytes + estimate.estimated_bytes
        )
        reservation = None
    else:
        request_lease = get_request_admission_lease()
        reservation = (
            await request_lease.reserve_temporary_response_bytes(
                estimate.estimated_bytes
            )
            if request_lease is not None
            else None
        )
    try:
        parse_payload = payload.tobytes() if payload_is_memoryview else payload
        try:
            value: Any = (
                await run_json_cpu(json.loads, parse_payload)
                if offload
                else json.loads(parse_payload)
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not allow_invalid:
                raise
            value = parse_payload
        return OwnedJSONValue(value, reservation)
    except BaseException:
        if reservation is not None:
            await reservation.release()
        raise


@asynccontextmanager
async def parsed_json_value(
    payload: str | bytes | bytearray | memoryview,
    *,
    max_estimated_bytes: int = DEFAULT_JSON_PARSE_MAX_ESTIMATED_BYTES,
    allow_invalid: bool = False,
):
    """Materialize untrusted JSON under a live structure-aware reservation."""

    owner = await parse_owned_json_value(
        payload,
        max_estimated_bytes=max_estimated_bytes,
        allow_invalid=allow_invalid,
    )
    try:
        yield owner.value
    finally:
        await owner.aclose()
