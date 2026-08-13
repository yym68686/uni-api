from __future__ import annotations

import asyncio
import fcntl
import os
import struct
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

from uni_api.admission.resources import (
    current_cgroup_v1_root,
    current_cgroup_v2_root,
)


_MIB = 1024 * 1024
_DEFAULT_FALLBACK_BUDGET_BYTES = 256 * _MIB
_DEFAULT_GUARD_BYTES = 512 * _MIB
_DEFAULT_GUARD_RATIO = 0.25
_DEFAULT_SAMPLE_CACHE_SECONDS = 0.05
_CGROUP_FILE_READ_LIMIT_BYTES = 64 * 1024
_SHARED_RESERVATION_CATEGORIES = (
    "total",
    "parsed_body",
    "serialized_body",
    "transport_buffer",
    "response_buffer",
    "other",
)
_SHARED_RESERVATION_BYTES = 8 * len(_SHARED_RESERVATION_CATEGORIES)
_DEFAULT_SHARED_RESERVATION_PATH = "/tmp/uni-api-shared-memory-reservation-v1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _proc_rss_bytes() -> int | None:
    try:
        lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return int(fields[1]) * 1024
        except ValueError:
            return None
    return None


class _CgroupFileReader:
    """Reuse tiny cgroup file descriptors while reading fresh values.

    Cgroup membership is fixed for a container process, while memory.current,
    memory.max, memory.high, and memory.events remain live kernel files. Keeping
    those descriptors open removes repeated pathlib/open/close work without
    caching any admission value or extending the governor's sample interval.
    """

    def __init__(self) -> None:
        self._descriptors: dict[Path, int] = {}

    def read_text(self, path: Path) -> str | None:
        descriptor = self._descriptors.get(path)
        if descriptor is None:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError:
                return None
            self._descriptors[path] = descriptor
        try:
            payload = os.pread(
                descriptor,
                _CGROUP_FILE_READ_LIMIT_BYTES,
                0,
            )
        except OSError:
            self._close_path(path)
            return None
        return payload.decode("ascii").strip()

    def read_int(self, path: Path) -> int | None:
        value = self.read_text(path)
        if not value or value == "max":
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

    def read_events(self, path: Path) -> dict[str, int]:
        events: dict[str, int] = {}
        content = self.read_text(path)
        if content is None:
            return events
        for line in content.splitlines():
            name, separator, value = line.partition(" ")
            if not separator:
                continue
            try:
                events[name] = int(value)
            except ValueError:
                continue
        return events

    def _close_path(self, path: Path) -> None:
        descriptor = self._descriptors.pop(path, None)
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except OSError:
            pass

    def close(self) -> None:
        for path in tuple(self._descriptors):
            self._close_path(path)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None = None
    events: dict[str, int] | None = None
    source: str = "unknown"


class CgroupMemorySource:
    """Read the current process cgroup without depending on Kubernetes APIs."""

    def __init__(
        self,
        root: str | Path = "/sys/fs/cgroup",
        proc_cgroup: str | Path = "/proc/self/cgroup",
    ) -> None:
        self.root = Path(root)
        self.proc_cgroup = Path(proc_cgroup)
        self._v2_root = current_cgroup_v2_root(self.root, self.proc_cgroup)
        self._v1_root = current_cgroup_v1_root(
            "memory",
            self.root,
            self.proc_cgroup,
        )
        self._reader = _CgroupFileReader()

    def sample(self) -> ProcessMemorySample:
        current = self._reader.read_int(self._v2_root / "memory.current")
        limit = self._reader.read_int(self._v2_root / "memory.max")
        high = self._reader.read_int(self._v2_root / "memory.high")
        if current is not None or limit is not None:
            return ProcessMemorySample(
                current_bytes=current,
                limit_bytes=limit,
                high_bytes=high,
                events=self._reader.read_events(
                    self._v2_root / "memory.events"
                ),
                source="cgroup-v2",
            )

        current = self._reader.read_int(
            self._v1_root / "memory.usage_in_bytes"
        )
        limit = self._reader.read_int(
            self._v1_root / "memory.limit_in_bytes"
        )
        if current is not None or limit is not None:
            # cgroup v1 commonly reports an enormous sentinel for unlimited.
            if limit is not None and limit >= (1 << 60):
                limit = None
            fail_count = self._reader.read_int(
                self._v1_root / "memory.failcnt"
            )
            return ProcessMemorySample(
                current_bytes=current,
                limit_bytes=limit,
                events={"max": fail_count} if fail_count is not None else {},
                source="cgroup-v1",
            )

        return ProcessMemorySample(
            current_bytes=_proc_rss_bytes(),
            limit_bytes=None,
            source="procfs",
        )

    def close(self) -> None:
        self._reader.close()


class SharedMemoryReservationLedger:
    """A flock-protected byte counter shared by the Rust and Python processes."""

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with self._locked():
            if reset or os.fstat(self._descriptor).st_size < _SHARED_RESERVATION_BYTES:
                self._write_locked([0] * len(_SHARED_RESERVATION_CATEGORIES))

    @staticmethod
    def _category_index(category: str) -> int:
        normalized = str(category or "other")
        if normalized in {
            "request_body",
            "pending_request_body",
            "parsed_body",
            "rust_parsed_body",
        }:
            return 1
        if normalized in {"upstream_serialized_body", "serialized_body"}:
            return 2
        if normalized in {"upstream_transport_buffer", "transport_buffer"}:
            return 3
        if normalized in {"buffered_response", "response_buffer"}:
            return 4
        return 5

    def _locked(self):
        ledger = self

        class _Lock:
            def __enter__(self):
                fcntl.flock(ledger._descriptor, fcntl.LOCK_EX)
                return ledger

            def __exit__(self, *_exc_info: object) -> None:
                fcntl.flock(ledger._descriptor, fcntl.LOCK_UN)

        return _Lock()

    def _read_locked(self) -> list[int]:
        payload = os.pread(
            self._descriptor,
            _SHARED_RESERVATION_BYTES,
            0,
        )
        if len(payload) != _SHARED_RESERVATION_BYTES:
            raise OSError("shared memory reservation ledger is truncated")
        return [
            int(value)
            for value in struct.unpack(
                f"<{len(_SHARED_RESERVATION_CATEGORIES)}Q",
                payload,
            )
        ]

    def _write_locked(self, values: list[int]) -> None:
        payload = struct.pack(
            f"<{len(_SHARED_RESERVATION_CATEGORIES)}Q",
            *(max(0, int(value)) for value in values),
        )
        if os.pwrite(self._descriptor, payload, 0) != len(payload):
            raise OSError("short write to shared memory reservation ledger")
        os.ftruncate(self._descriptor, _SHARED_RESERVATION_BYTES)

    def total(self) -> int:
        with self._locked():
            return self._read_locked()[0]

    def categories(self) -> dict[str, int]:
        with self._locked():
            values = self._read_locked()
        return dict(zip(_SHARED_RESERVATION_CATEGORIES, values, strict=True))

    def try_reserve(
        self,
        category: str,
        size: int,
        *,
        maximum_total: int,
    ) -> tuple[bool, int, int]:
        with self._locked():
            values = self._read_locked()
            before = values[0]
            after = before + int(size)
            if after > int(maximum_total):
                return False, before, before
            values[0] = after
            values[self._category_index(category)] += int(size)
            self._write_locked(values)
            return True, before, after

    def release(self, category: str, size: int) -> None:
        with self._locked():
            values = self._read_locked()
            category_index = self._category_index(category)
            if int(size) > values[0] or int(size) > values[category_index]:
                raise RuntimeError("shared memory reservation ledger underflow")
            values[0] -= int(size)
            values[category_index] -= int(size)
            self._write_locked(values)

    def close(self) -> None:
        descriptor = getattr(self, "_descriptor", None)
        if descriptor is None:
            return
        self._descriptor = None
        os.close(descriptor)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class AdaptiveMemorySnapshot:
    source: str
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None
    soft_limit_bytes: int | None
    guard_bytes: int
    capacity_bytes: int
    available_bytes: int
    reserved_bytes: int
    peak_reserved_bytes: int
    reservations: dict[str, int]
    shared_reservations: dict[str, int]
    rejected: dict[str, int]
    blocked_reservations: int
    waiting_reservations: int
    wait_timeouts: int
    events: dict[str, int]
    sample_error: str | None
    sample_sequence: int
    sampled_at_monotonic: float | None
    sample_age_ms: int | None


@dataclass(frozen=True, slots=True)
class AdaptiveMemoryReservationDecision:
    """Atomic result and cgroup facts used by one parent reservation."""

    allowed: bool
    category: str
    requested_bytes: int
    reserved_before_bytes: int
    projected_reserved_bytes: int
    reserved_after_bytes: int
    source: str
    current_bytes: int | None
    limit_bytes: int | None
    high_bytes: int | None
    soft_limit_bytes: int | None
    guard_bytes: int
    capacity_bytes: int
    available_before_bytes: int
    available_after_bytes: int
    sample_error: str | None
    sample_sequence: int
    sampled_at_monotonic: float | None
    sample_age_ms: int | None


class AdaptiveMemoryGovernor:
    """One atomic parent budget for every process-owned retained byte.

    Existing reservations deliberately remain part of the projected memory
    even though some bytes are already reflected by ``memory.current``.  This
    conservative double accounting leaves room for JSON/Pydantic
    materialization, allocator fragmentation, and allocations that occur
    between cgroup samples.
    """

    def __init__(
        self,
        *,
        source: Callable[[], ProcessMemorySample] | None = None,
        soft_limit_bytes: int | None = None,
        guard_bytes: int = _DEFAULT_GUARD_BYTES,
        guard_ratio: float = _DEFAULT_GUARD_RATIO,
        fallback_budget_bytes: int = _DEFAULT_FALLBACK_BUDGET_BYTES,
        sample_cache_seconds: float = _DEFAULT_SAMPLE_CACHE_SECONDS,
        clock: Callable[[], float] = monotonic,
        shared_ledger: SharedMemoryReservationLedger | None = None,
    ) -> None:
        if soft_limit_bytes is not None and soft_limit_bytes <= 0:
            raise ValueError("soft_limit_bytes must be positive when provided")
        if guard_bytes < 0 or not 0 <= guard_ratio < 1:
            raise ValueError("memory guard must be non-negative and below one")
        if fallback_budget_bytes <= 0:
            raise ValueError("fallback_budget_bytes must be positive")
        if sample_cache_seconds < 0:
            raise ValueError("sample_cache_seconds cannot be negative")

        cgroup_source = CgroupMemorySource()
        self._source = source or cgroup_source.sample
        self._configured_soft_limit_bytes = soft_limit_bytes
        self.guard_bytes = int(guard_bytes)
        self.guard_ratio = float(guard_ratio)
        self.fallback_budget_bytes = int(fallback_budget_bytes)
        self.sample_cache_seconds = float(sample_cache_seconds)
        self._clock = clock
        self._shared_ledger = shared_ledger
        self._shared_ledger_error: str | None = None

        self._lock = threading.RLock()
        self._sample: ProcessMemorySample | None = None
        self._sampled_at = float("-inf")
        self._sample_attempted_at = float("-inf")
        self._sample_sequence = 0
        self._sample_error: str | None = None
        self._reserved: Counter[str] = Counter()
        self._rejected: Counter[str] = Counter()
        self._peak_reserved_bytes = 0
        self._blocked_reservations = 0
        self._waiting_reservations = 0
        self._wait_timeouts = 0
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    @classmethod
    def from_environment(cls) -> AdaptiveMemoryGovernor:
        configured_soft_limit = _env_int("MEMORY_SOFT_LIMIT_BYTES", 0)
        shared_path = os.getenv(
            "UNI_API_SHARED_MEMORY_RESERVATION_PATH",
            _DEFAULT_SHARED_RESERVATION_PATH,
        )
        return cls(
            soft_limit_bytes=(
                configured_soft_limit if configured_soft_limit > 0 else None
            ),
            guard_bytes=max(0, _env_int("MEMORY_GUARD_BYTES", _DEFAULT_GUARD_BYTES)),
            guard_ratio=min(
                0.95,
                max(0.0, _env_float("MEMORY_GUARD_RATIO", _DEFAULT_GUARD_RATIO)),
            ),
            fallback_budget_bytes=max(
                1,
                _env_int(
                    "MEMORY_FALLBACK_BUDGET_BYTES",
                    _DEFAULT_FALLBACK_BUDGET_BYTES,
                ),
            ),
            sample_cache_seconds=max(
                0.0,
                _env_float(
                    "MEMORY_SAMPLE_CACHE_SECONDS",
                    _DEFAULT_SAMPLE_CACHE_SECONDS,
                ),
            ),
            shared_ledger=SharedMemoryReservationLedger(shared_path),
        )

    def _reserved_total_locked(self) -> int:
        if self._shared_ledger is None:
            return sum(self._reserved.values())
        try:
            total = self._shared_ledger.total()
            self._shared_ledger_error = None
            return total
        except Exception as exc:
            self._shared_ledger_error = f"{type(exc).__name__}: {exc}"[:512]
            return sum(self._reserved.values())

    def _shared_categories_locked(self) -> dict[str, int]:
        if self._shared_ledger is None:
            return {}
        try:
            categories = self._shared_ledger.categories()
            self._shared_ledger_error = None
            return categories
        except Exception as exc:
            self._shared_ledger_error = f"{type(exc).__name__}: {exc}"[:512]
            return {}

    def _refresh_sample_locked(self, *, force: bool = False) -> ProcessMemorySample:
        now = self._clock()
        if (
            not force
            and self._sample is not None
            and now - self._sample_attempted_at < self.sample_cache_seconds
        ):
            return self._sample
        self._sample_attempted_at = now
        try:
            sample = self._source()
            if not isinstance(sample, ProcessMemorySample):
                raise TypeError("memory source returned an invalid sample")
            self._sample = sample
            self._sample_error = None
            self._sampled_at = now
            self._sample_sequence += 1
        except Exception as exc:
            self._sample_error = f"{type(exc).__name__}: {exc}"[:512]
            if self._sample is None:
                self._sample = ProcessMemorySample(None, None, source="unavailable")
        return self._sample

    def _limits_locked(
        self,
        sample: ProcessMemorySample,
    ) -> tuple[int | None, int, int, int]:
        limit = sample.limit_bytes
        if sample.high_bytes is not None:
            limit = min(limit, sample.high_bytes) if limit is not None else sample.high_bytes
        if self._configured_soft_limit_bytes is not None:
            soft_limit = self._configured_soft_limit_bytes
            if limit is not None:
                soft_limit = min(soft_limit, limit)
            guard = max(0, (limit or soft_limit) - soft_limit)
        elif limit is not None:
            # A fixed 512 MiB safety margin is useful for the production-sized
            # Pod, but it must not consume an entire small container.  Cap only
            # the absolute component at half the effective cgroup limit; an
            # explicitly configured ratio remains authoritative.
            absolute_guard = min(self.guard_bytes, limit // 2)
            guard = max(absolute_guard, int(limit * self.guard_ratio))
            soft_limit = max(1, limit - min(guard, max(0, limit - 1)))
        else:
            guard = self.guard_bytes
            soft_limit = None

        reserved = self._reserved_total_locked()
        if soft_limit is None or sample.current_bytes is None:
            capacity = max(reserved, self.fallback_budget_bytes)
        else:
            capacity = max(reserved, soft_limit - sample.current_bytes)
        if self._sample_error is not None:
            # A stale last-good cgroup sample must never authorize expansion.
            # Preserve existing ownership, but contract all new allocations to
            # the portable finite fallback until sampling recovers.
            capacity = max(reserved, min(capacity, self.fallback_budget_bytes))
        available = (
            0
            if self._shared_ledger_error is not None
            else max(0, capacity - reserved)
        )
        return soft_limit, guard, capacity, available

    def maximum_capacity_bytes(self) -> int:
        with self._lock:
            sample = self._refresh_sample_locked(force=True)
            soft_limit, _guard, _capacity, _available = self._limits_locked(sample)
            return soft_limit or self.fallback_budget_bytes

    def reserve_nowait(self, category: str, size: int) -> bool:
        return self.reserve_nowait_decision(category, size).allowed

    def reserve_nowait_decision(
        self,
        category: str,
        size: int,
    ) -> AdaptiveMemoryReservationDecision:
        """Reserve immediately and return the exact sample used to decide."""

        size = int(size)
        if size < 0:
            raise ValueError("memory reservation cannot be negative")
        normalized = str(category or "unknown").strip() or "unknown"
        with self._lock:
            sample = self._refresh_sample_locked()
            soft_limit, guard, capacity, available_before = self._limits_locked(
                sample
            )
            reserved_before = self._reserved_total_locked()
            allowed = size <= available_before
            reserved_after = reserved_before
            if allowed and size and self._shared_ledger is not None:
                try:
                    allowed, reserved_before, reserved_after = (
                        self._shared_ledger.try_reserve(
                            normalized,
                            size,
                            maximum_total=capacity,
                        )
                    )
                    self._shared_ledger_error = None
                except Exception as exc:
                    self._shared_ledger_error = (
                        f"{type(exc).__name__}: {exc}"[:512]
                    )
                    allowed = False
            if allowed and size:
                self._reserved[normalized] += size
                self._peak_reserved_bytes = max(
                    self._peak_reserved_bytes,
                    reserved_after if self._shared_ledger is not None else reserved_before + size,
                )
            elif not allowed:
                self._rejected[normalized] += 1
            sampled_at = (
                self._sampled_at
                if self._sample_sequence > 0
                and self._sampled_at != float("-inf")
                else None
            )
            sample_age_ms = (
                max(
                    0,
                    int(round((self._clock() - sampled_at) * 1000.0)),
                )
                if sampled_at is not None
                else None
            )
            if self._shared_ledger is None:
                reserved_after = reserved_before + size if allowed else reserved_before
            return AdaptiveMemoryReservationDecision(
                allowed=allowed,
                category=normalized,
                requested_bytes=size,
                reserved_before_bytes=reserved_before,
                projected_reserved_bytes=reserved_before + size,
                reserved_after_bytes=reserved_after,
                source=sample.source,
                current_bytes=sample.current_bytes,
                limit_bytes=sample.limit_bytes,
                high_bytes=sample.high_bytes,
                soft_limit_bytes=soft_limit,
                guard_bytes=guard,
                capacity_bytes=capacity,
                available_before_bytes=available_before,
                available_after_bytes=max(
                    0,
                    available_before - size if allowed else available_before,
                ),
                sample_error=self._sample_error or self._shared_ledger_error,
                sample_sequence=self._sample_sequence,
                sampled_at_monotonic=sampled_at,
                sample_age_ms=sample_age_ms,
            )

    def _try_reserve_locked(
        self,
        category: str,
        size: int,
        *,
        record_rejection: bool,
    ) -> bool:
        sample = self._refresh_sample_locked()
        _soft_limit, _guard, capacity, available = self._limits_locked(sample)
        if size > available:
            if record_rejection:
                self._rejected[category] += 1
            return False
        if self._shared_ledger is not None:
            try:
                allowed, _before, after = self._shared_ledger.try_reserve(
                    category,
                    size,
                    maximum_total=capacity,
                )
                self._shared_ledger_error = None
            except Exception as exc:
                self._shared_ledger_error = f"{type(exc).__name__}: {exc}"[:512]
                allowed = False
                after = self._reserved_total_locked()
            if not allowed:
                if record_rejection:
                    self._rejected[category] += 1
                return False
        else:
            after = sum(self._reserved.values()) + size
        self._reserved[category] += size
        self._peak_reserved_bytes = max(
            self._peak_reserved_bytes,
            after,
        )
        return True

    async def reserve(
        self,
        category: str,
        size: int,
        *,
        timeout_seconds: float,
    ) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        normalized = str(category or "unknown").strip() or "unknown"
        size = int(size)
        if size < 0:
            raise ValueError("memory reservation cannot be negative")
        if size == 0:
            return True
        with self._lock:
            if self._try_reserve_locked(
                normalized,
                size,
                record_rejection=False,
            ):
                return True

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        waiter = (loop, event)
        started_at = loop.time()
        with self._lock:
            self._blocked_reservations += 1
            self._waiting_reservations += 1
            self._waiters.add(waiter)
        try:
            while True:
                remaining = timeout_seconds - (loop.time() - started_at)
                if remaining <= 0:
                    with self._lock:
                        self._wait_timeouts += 1
                    return False
                event.clear()
                # Recheck after registration/clear so a release in either race
                # window cannot leave this waiter asleep with available space.
                with self._lock:
                    if self._try_reserve_locked(
                        normalized,
                        size,
                        record_rejection=False,
                    ):
                        return True
                try:
                    async with asyncio.timeout(min(0.1, remaining)):
                        await event.wait()
                except TimeoutError:
                    # ``memory.current`` may fall after GC without any tracked
                    # lease release. Poll the cgroup until the caller's real
                    # deadline so that newly available headroom is observed.
                    continue
        finally:
            with self._lock:
                self._waiters.discard(waiter)
                self._waiting_reservations = max(0, self._waiting_reservations - 1)

    def release(self, category: str, size: int) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("memory release cannot be negative")
        if size == 0:
            return
        normalized = str(category or "unknown").strip() or "unknown"
        with self._lock:
            if size > self._reserved[normalized]:
                raise RuntimeError("adaptive memory reservation underflow")
            if self._shared_ledger is not None:
                self._shared_ledger.release(normalized, size)
            self._reserved[normalized] -= size
            if self._reserved[normalized] == 0:
                del self._reserved[normalized]
            waiters = tuple(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                continue

    def _snapshot_locked(
        self,
        sample: ProcessMemorySample,
    ) -> AdaptiveMemorySnapshot:
        soft_limit, guard, capacity, available = self._limits_locked(sample)
        sampled_at = (
            self._sampled_at
            if self._sample_sequence > 0 and self._sampled_at != float("-inf")
            else None
        )
        sample_age_ms = (
            max(0, int(round((self._clock() - sampled_at) * 1000.0)))
            if sampled_at is not None
            else None
        )
        return AdaptiveMemorySnapshot(
            source=sample.source,
            current_bytes=sample.current_bytes,
            limit_bytes=sample.limit_bytes,
            high_bytes=sample.high_bytes,
            soft_limit_bytes=soft_limit,
            guard_bytes=guard,
            capacity_bytes=capacity,
            available_bytes=available,
            reserved_bytes=self._reserved_total_locked(),
            peak_reserved_bytes=self._peak_reserved_bytes,
            reservations=dict(self._reserved),
            shared_reservations=self._shared_categories_locked(),
            rejected=dict(self._rejected),
            blocked_reservations=self._blocked_reservations,
            waiting_reservations=self._waiting_reservations,
            wait_timeouts=self._wait_timeouts,
            events=dict(sample.events or {}),
            sample_error=self._sample_error or self._shared_ledger_error,
            sample_sequence=self._sample_sequence,
            sampled_at_monotonic=sampled_at,
            sample_age_ms=sample_age_ms,
        )

    def snapshot_cached(self) -> AdaptiveMemorySnapshot:
        """Return the latest sample without filesystem I/O.

        Admission decision recording calls this while its asyncio lock is held.
        The sample sequence and age make the cache boundary explicit instead of
        presenting a cached cgroup value as an atomic kernel measurement.
        """

        with self._lock:
            sample = self._sample or ProcessMemorySample(
                None,
                None,
                source="unavailable",
            )
            return self._snapshot_locked(sample)

    def snapshot(self, *, force: bool = False) -> AdaptiveMemorySnapshot:
        with self._lock:
            sample = self._refresh_sample_locked(force=force)
            return self._snapshot_locked(sample)


process_memory_governor = AdaptiveMemoryGovernor.from_environment()
