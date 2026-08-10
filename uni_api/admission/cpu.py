from __future__ import annotations

import asyncio
import os
from collections import Counter
from concurrent.futures import Executor
from typing import Any, Callable

from uni_api.admission.resources import startup_cpu_worker_count


def _cpu_phase_capacity() -> int:
    detected = startup_cpu_worker_count(
        executor_groups=1,
        minimum_per_group=1,
    )
    raw = os.getenv("CPU_PHASE_TOKENS")
    if raw is None or not raw.strip():
        return detected
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CPU_PHASE_TOKENS must be an integer") from exc
    if value <= 0:
        raise ValueError("CPU_PHASE_TOKENS must be positive")
    return value


class CPUPhaseLimiter:
    """Share CPU execution tokens across independent offload executors.

    The request admission lease owns memory and transport state for the whole
    ASGI lifecycle. This limiter is intentionally narrower: a token is held
    only while one submitted CPU callback is executing or waiting to finish
    after caller cancellation.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("CPU phase capacity must be positive")
        self.capacity = int(capacity)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._active = 0
        self._waiters = 0
        self._acquired_total = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._failed_total = 0
        self._active_by_phase: Counter[str] = Counter()
        self._waiters_by_phase: Counter[str] = Counter()

    def _semaphore_for_running_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._loop is loop and self._semaphore is not None:
            return self._semaphore
        if self._active or self._waiters:
            raise RuntimeError(
                "CPU phase limiter cannot span active event loops"
            )
        # asyncio primitives bind lazily when they first contend. Recreate the
        # primitive after a prior loop has drained so reloads and isolated
        # asyncio.run() lifecycles cannot retain a closed loop.
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self.capacity)
        return self._semaphore

    async def acquire(self, phase: str) -> None:
        normalized = str(phase or "other")
        semaphore = self._semaphore_for_running_loop()
        self._waiters += 1
        self._waiters_by_phase[normalized] += 1
        try:
            await semaphore.acquire()
        finally:
            self._waiters -= 1
            self._waiters_by_phase[normalized] -= 1
            if self._waiters_by_phase[normalized] <= 0:
                del self._waiters_by_phase[normalized]
        self._active += 1
        self._active_by_phase[normalized] += 1
        self._acquired_total += 1

    def release(
        self,
        phase: str,
        *,
        cancelled: bool,
        failed: bool,
    ) -> None:
        normalized = str(phase or "other")
        if self._active <= 0 or self._active_by_phase[normalized] <= 0:
            raise RuntimeError("CPU phase token accounting underflow")
        self._active -= 1
        self._active_by_phase[normalized] -= 1
        if self._active_by_phase[normalized] <= 0:
            del self._active_by_phase[normalized]
        self._completed_total += 1
        if cancelled:
            self._cancelled_total += 1
        if failed:
            self._failed_total += 1
        semaphore = self._semaphore
        if semaphore is None:
            raise RuntimeError("CPU phase semaphore is unavailable")
        semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "active": self._active,
            "waiters": self._waiters,
            "acquired_total": self._acquired_total,
            "completed_total": self._completed_total,
            "cancelled_total": self._cancelled_total,
            "failed_total": self._failed_total,
            "active_by_phase": dict(self._active_by_phase),
            "waiters_by_phase": dict(self._waiters_by_phase),
        }


CPU_PHASE_LIMITER = CPUPhaseLimiter(_cpu_phase_capacity())


async def run_cpu_phase(
    executor: Executor,
    callback: Callable[..., Any],
    *args: Any,
    phase: str,
    **kwargs: Any,
) -> Any:
    """Run one callback under a shared, cancellation-safe CPU token."""

    await CPU_PHASE_LIMITER.acquire(phase)
    pending_cancel: asyncio.CancelledError | None = None
    failed = False
    future: asyncio.Future[Any] | None = None
    try:
        loop = asyncio.get_running_loop()

        def invoke() -> Any:
            return callback(*args, **kwargs)

        try:
            future = loop.run_in_executor(executor, invoke)
        except BaseException:
            failed = True
            raise
        owner_task = asyncio.current_task()
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError as exc:
                pending_cancel = pending_cancel or exc
            except BaseException:
                if (
                    pending_cancel is None
                    and owner_task is not None
                    and owner_task.cancelling()
                ):
                    pending_cancel = asyncio.CancelledError()
                if pending_cancel is None:
                    failed = True
                    raise
                break
        if (
            pending_cancel is None
            and owner_task is not None
            and owner_task.cancelling()
        ):
            pending_cancel = asyncio.CancelledError()
        if pending_cancel is not None:
            try:
                future.result()
            except BaseException:
                pass
            raise pending_cancel
        try:
            return future.result()
        except BaseException:
            failed = True
            raise
    finally:
        CPU_PHASE_LIMITER.release(
            phase,
            cancelled=pending_cancel is not None,
            failed=failed,
        )


def cpu_phase_snapshot() -> dict[str, Any]:
    return CPU_PHASE_LIMITER.snapshot()
