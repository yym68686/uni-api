from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from uni_api.admission.memory import AdaptiveMemoryGovernor
from uni_api.observability.threadpool_tasks import (
    register_dedicated_threadpool,
    submit_threadpool_task,
)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_SPOOL_IO_WORKERS = _positive_int_env("IDEMPOTENCY_SPOOL_IO_WORKERS", 4)
_SPOOL_IO_MAX_PENDING = max(
    _SPOOL_IO_WORKERS,
    _positive_int_env("IDEMPOTENCY_SPOOL_IO_MAX_PENDING", 128),
)
_SPOOL_BATCH_BYTES = _positive_int_env(
    "IDEMPOTENCY_SPOOL_BATCH_BYTES",
    256 * 1024,
)
_SPOOL_IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=_SPOOL_IO_WORKERS,
    thread_name_prefix="uni-api-idempotency-spool",
)
register_dedicated_threadpool("idempotency_spool", _SPOOL_IO_EXECUTOR)
_SPOOL_IO_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Semaphore,
] = weakref.WeakKeyDictionary()
_SPOOL_IO_LIMITERS_LOCK = threading.Lock()


def _spool_io_limiter(
    loop: asyncio.AbstractEventLoop,
) -> asyncio.Semaphore:
    with _SPOOL_IO_LIMITERS_LOCK:
        limiter = _SPOOL_IO_LIMITERS.get(loop)
        if limiter is None:
            limiter = asyncio.Semaphore(_SPOOL_IO_MAX_PENDING)
            _SPOOL_IO_LIMITERS[loop] = limiter
        return limiter


class IdempotencySpoolError(RuntimeError):
    """The bounded replay spool could not retain another byte safely."""


class IdempotencySpool:
    """A small in-memory buffer which spills into a private temporary file.

    The caller owns global byte quota.  This object owns only its local Python
    heap reservation and temporary file lifecycle.  Completed response spools
    are sealed before publication, so replay readers never observe a growing
    file.
    """

    def __init__(
        self,
        *,
        directory: str,
        memory_threshold_bytes: int,
        memory_governor: AdaptiveMemoryGovernor | None,
        memory_category: str,
    ) -> None:
        if memory_threshold_bytes < 0:
            raise ValueError("memory_threshold_bytes cannot be negative")
        self._directory = str(directory)
        self._memory_threshold_bytes = int(memory_threshold_bytes)
        self._memory_governor = memory_governor
        self._memory_category = str(memory_category or "idempotency_spool")
        self._memory = bytearray()
        self._memory_reserved_bytes = 0
        self._pending_disk = bytearray()
        self._pending_disk_reserved_bytes = 0
        self._writer = None
        self._path: str | None = None
        self._size = 0
        self._sealed = False
        self._closed = False
        self._async_io_lock = asyncio.Lock()
        self._async_close_task: asyncio.Task[None] | None = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def memory_bytes(self) -> int:
        return self._memory_reserved_bytes + self._pending_disk_reserved_bytes

    @property
    def on_disk(self) -> bool:
        return self._path is not None

    @property
    def sealed(self) -> bool:
        return self._sealed

    def append(self, payload: bytes | bytearray | memoryview) -> None:
        if self._sealed or self._closed:
            raise IdempotencySpoolError("cannot append to a closed replay spool")
        chunk = bytes(payload)
        if not chunk:
            return
        if self._writer is not None:
            self._write_all(chunk)
            self._size += len(chunk)
            return

        projected = len(self._memory) + len(chunk)
        should_spill = projected > self._memory_threshold_bytes
        if not should_spill and not self._reserve_memory(len(chunk)):
            should_spill = True
        if should_spill:
            self._spill_to_disk()
            self._write_all(chunk)
        else:
            self._memory.extend(chunk)
        self._size += len(chunk)

    async def append_async(self, payload: bytes | bytearray | memoryview) -> None:
        """Append with bounded coalescing and no blocking I/O on the loop."""

        chunk = bytes(payload)
        if not chunk:
            return
        async with self._async_io_lock:
            if self._sealed or self._closed:
                raise IdempotencySpoolError(
                    "cannot append to a closed replay spool"
                )
            if self._writer is None and not self._pending_disk:
                projected = len(self._memory) + len(chunk)
                if (
                    projected <= self._memory_threshold_bytes
                    and self._reserve_memory(len(chunk))
                ):
                    self._memory.extend(chunk)
                    self._size += len(chunk)
                    return

            if len(chunk) >= _SPOOL_BATCH_BYTES:
                await self._flush_pending_disk()
                await self._run_io(self._append_storage, chunk)
                self._size += len(chunk)
                return

            if len(self._pending_disk) + len(chunk) > _SPOOL_BATCH_BYTES:
                await self._flush_pending_disk()
            if not self._reserve_pending_disk_memory(len(chunk)):
                await self._flush_pending_disk()
                await self._run_io(self._append_storage, chunk)
                self._size += len(chunk)
                return

            self._pending_disk.extend(chunk)
            self._size += len(chunk)
            if len(self._pending_disk) >= _SPOOL_BATCH_BYTES:
                await self._flush_pending_disk()

    async def seal_async(self, *, force_disk: bool) -> None:
        if self._closed:
            raise IdempotencySpoolError("cannot seal a closed replay spool")
        if self._sealed:
            return
        async with self._async_io_lock:
            await self._flush_pending_disk()
            await self._run_io(self.seal, force_disk=force_disk)

    async def close_async(self) -> None:
        if self._closed:
            return
        if self._async_close_task is None:
            self._async_close_task = asyncio.create_task(self._close_async_once())
        task = self._async_close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await self._await_future_despite_cancellation(task)
            raise

    async def _close_async_once(self) -> None:
        async with self._async_io_lock:
            self._discard_pending_disk()
            await self._run_io(self.close)

    def seal(self, *, force_disk: bool) -> None:
        if self._closed:
            raise IdempotencySpoolError("cannot seal a closed replay spool")
        if self._sealed:
            return
        if force_disk and self._writer is None:
            self._spill_to_disk()
        if self._writer is not None:
            self._writer.flush()
            self._drop_page_cache(self._writer.fileno())
            self._writer.close()
            self._writer = None
        self._sealed = True

    def iter_chunks(self, chunk_size: int = 256 * 1024) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not self._sealed or self._closed:
            raise IdempotencySpoolError("replay spool is not readable")
        if self._path is None:
            for offset in range(0, len(self._memory), chunk_size):
                yield bytes(self._memory[offset : offset + chunk_size])
            return

        # Named files allow completed entries to close their writer descriptor;
        # hundreds of cached responses therefore do not consume hundreds of FDs.
        with open(self._path, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
        self._discard_pending_disk()
        self._release_memory()
        self._memory.clear()
        path = self._path
        self._path = None
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    async def _run_io(self, callback, *args, **kwargs):
        loop = asyncio.get_running_loop()
        limiter = _spool_io_limiter(loop)
        async with limiter:
            ticket = submit_threadpool_task("idempotency_spool")

            def invoke():
                return ticket.run(callback, *args, **kwargs)

            future = loop.run_in_executor(
                _SPOOL_IO_EXECUTOR,
                invoke,
            )
            return await self._await_future_despite_cancellation(future)

    async def _flush_pending_disk(self) -> None:
        if not self._pending_disk:
            return
        payload = self._pending_disk
        reserved = self._pending_disk_reserved_bytes
        self._pending_disk = bytearray()
        self._pending_disk_reserved_bytes = 0
        try:
            await self._run_io(self._append_storage, payload)
        finally:
            if reserved and self._memory_governor is not None:
                self._memory_governor.release(self._memory_category, reserved)

    def _append_storage(self, payload: bytes | bytearray | memoryview) -> None:
        if self._writer is None:
            self._spill_to_disk()
        self._write_all(payload)

    def _reserve_pending_disk_memory(self, size: int) -> bool:
        if size <= 0:
            return True
        if (
            self._memory_governor is not None
            and not self._memory_governor.reserve_nowait(
                self._memory_category,
                size,
            )
        ):
            return False
        self._pending_disk_reserved_bytes += size
        return True

    def _discard_pending_disk(self) -> None:
        reserved = self._pending_disk_reserved_bytes
        self._pending_disk_reserved_bytes = 0
        self._pending_disk.clear()
        if reserved and self._memory_governor is not None:
            self._memory_governor.release(self._memory_category, reserved)

    @staticmethod
    async def _await_future_despite_cancellation(future):
        pending_cancel: asyncio.CancelledError | None = None
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError as exc:
                pending_cancel = pending_cancel or exc
        if pending_cancel is not None:
            try:
                future.result()
            except BaseException:
                pass
            raise pending_cancel
        return future.result()

    def _reserve_memory(self, size: int) -> bool:
        if size <= 0:
            return True
        if (
            self._memory_governor is not None
            and not self._memory_governor.reserve_nowait(
                self._memory_category,
                size,
            )
        ):
            return False
        self._memory_reserved_bytes += size
        return True

    def _release_memory(self) -> None:
        reserved = self._memory_reserved_bytes
        self._memory_reserved_bytes = 0
        if reserved and self._memory_governor is not None:
            self._memory_governor.release(self._memory_category, reserved)

    def _spill_to_disk(self) -> None:
        if self._writer is not None:
            return
        try:
            descriptor, path = tempfile.mkstemp(
                prefix="buffer-",
                suffix=".spool",
                dir=self._directory,
                text=False,
            )
            os.chmod(path, 0o600)
            writer = os.fdopen(descriptor, "w+b", buffering=0)
        except OSError as exc:
            raise IdempotencySpoolError("cannot create idempotency spool") from exc
        self._path = path
        self._writer = writer
        try:
            if self._memory:
                self._write_all(self._memory)
        except BaseException:
            self.close()
            raise
        self._memory.clear()
        self._release_memory()

    def _write_all(self, payload: bytes | bytearray | memoryview) -> None:
        writer = self._writer
        if writer is None:
            raise IdempotencySpoolError("idempotency spool writer is unavailable")
        try:
            written = writer.write(payload)
        except OSError as exc:
            raise IdempotencySpoolError("cannot write idempotency spool") from exc
        if written != len(payload):
            raise IdempotencySpoolError("short write to idempotency spool")

    @staticmethod
    def _drop_page_cache(descriptor: int) -> None:
        advice = getattr(os, "POSIX_FADV_DONTNEED", None)
        callback = getattr(os, "posix_fadvise", None)
        if advice is None or callback is None:
            return
        try:
            callback(descriptor, 0, 0, advice)
        except OSError:
            pass


def create_private_spool_directory(
    parent: str | None = None,
) -> tempfile.TemporaryDirectory:
    if parent:
        Path(parent).mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(
        prefix="uni-api-idempotency-",
        dir=parent or None,
    )
    os.chmod(directory.name, 0o700)
    return directory
