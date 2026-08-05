from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

from uni_api.admission.memory import AdaptiveMemoryGovernor


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
        self._writer = None
        self._path: str | None = None
        self._size = 0
        self._sealed = False
        self._closed = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def memory_bytes(self) -> int:
        return self._memory_reserved_bytes

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
