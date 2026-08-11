from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic, time
from typing import Any, Literal
from uuid import uuid4

from uni_api.admission.memory import (
    AdaptiveMemoryGovernor,
    AdaptiveMemoryReservationDecision,
    AdaptiveMemorySnapshot,
)
from uni_api.admission.observability import (
    LargeBodyAdmissionDecision,
    LargeBodyHolderSnapshot,
    ResponseBufferEvent,
    RequestBodyObservation,
)


_STREAM_RESPONSE_SUMMARY_FALLBACK_SECONDS = 5.0


class AdmissionRejected(RuntimeError):
    """A bounded admission decision that callers can translate to HTTP."""

    def __init__(self, reason: str, *, status_code: int = 503) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


class RequestBodyTooLarge(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("body_too_large", status_code=413)


class RequestBodyBudgetExhausted(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("body_budget_exhausted", status_code=503)


class LargeBodyCapacityExhausted(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("large_body_capacity_exhausted", status_code=503)


class UpstreamResponseBudgetExhausted(AdmissionRejected):
    local_admission_rejection = True

    def __init__(self, admission_branch: str) -> None:
        super().__init__("upstream_response_budget_exhausted", status_code=503)
        self.admission_branch = str(admission_branch)


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[float]
    state: str = "queued"


@dataclass(slots=True)
class _BodyMemoryWaiter:
    future: asyncio.Future[None]
    tenant_key: str
    small_lane: bool
    state: str = "queued"


class _FairBodyMemoryWaitQueue:
    """Two-lane, tenant-round-robin ordering for memory waiters."""

    def __init__(self, limit: int | None) -> None:
        if limit is not None and limit < 0:
            raise ValueError("body memory waiter limit cannot be negative")
        self.limit = int(limit) if limit is not None else None
        self._queues: dict[bool, dict[str, deque[_BodyMemoryWaiter]]] = {
            True: {},
            False: {},
        }
        self._tenant_round_robin: dict[bool, deque[str]] = {
            True: deque(),
            False: deque(),
        }
        self._active_turn: dict[bool, bool] = {True: False, False: False}
        self._promotion_enabled: dict[bool, bool] = {True: True, False: True}
        self._count = 0
        self._peak = 0
        self._grants = 0
        self._timeouts = 0
        self._queue_full = 0

    def has_waiters(self, small_lane: bool) -> bool:
        lane = bool(small_lane)
        return self._active_turn[lane] or self._count_for_lane(lane) > 0

    async def wait_turn(
        self,
        *,
        tenant_key: str,
        small_lane: bool,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise TimeoutError
        if self.limit is not None and self._count >= self.limit:
            self._queue_full += 1
            raise TimeoutError
        loop = asyncio.get_running_loop()
        normalized_tenant = str(tenant_key or "anonymous")[:96]
        waiter = _BodyMemoryWaiter(
            future=loop.create_future(),
            tenant_key=normalized_tenant,
            small_lane=bool(small_lane),
        )
        tenant_queues = self._queues[waiter.small_lane]
        tenant_queue = tenant_queues.get(normalized_tenant)
        if tenant_queue is None:
            tenant_queue = deque()
            tenant_queues[normalized_tenant] = tenant_queue
            self._tenant_round_robin[waiter.small_lane].append(
                normalized_tenant
            )
        tenant_queue.append(waiter)
        self._count += 1
        self._peak = max(self._peak, self._count)
        self._promote(waiter.small_lane)
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            self._timeouts += 1
            self._remove(waiter)
            raise
        except asyncio.CancelledError:
            self._remove(waiter)
            raise

    def finish_turn(self, small_lane: bool) -> None:
        if not self._active_turn[bool(small_lane)]:
            raise RuntimeError("body memory wait turn underflow")
        self._active_turn[bool(small_lane)] = False
        self._promote(bool(small_lane))

    def block_until_release(self, small_lane: bool) -> None:
        self._promotion_enabled[bool(small_lane)] = False

    def notify_memory_released(self) -> None:
        for lane in (True, False):
            self._promotion_enabled[lane] = True
            self._promote(lane)

    def snapshot(self) -> dict[str, int]:
        return {
            "body_memory_waiters": self._count,
            "body_memory_small_waiters": self._count_for_lane(True),
            "body_memory_large_waiters": self._count_for_lane(False),
            "body_memory_small_turn_active": int(self._active_turn[True]),
            "body_memory_large_turn_active": int(self._active_turn[False]),
            "body_memory_waiter_peak": self._peak,
            "body_memory_wait_grants": self._grants,
            "body_memory_wait_timeouts": self._timeouts,
            "body_memory_wait_queue_full": self._queue_full,
        }

    def _count_for_lane(self, small_lane: bool) -> int:
        return sum(
            len(queue) for queue in self._queues[bool(small_lane)].values()
        )

    def _promote(self, small_lane: bool) -> None:
        lane = bool(small_lane)
        if self._active_turn[lane] or not self._promotion_enabled[lane]:
            return
        round_robin = self._tenant_round_robin[lane]
        tenant_queues = self._queues[lane]
        while round_robin:
            tenant = round_robin.popleft()
            queue = tenant_queues.get(tenant)
            if not queue:
                tenant_queues.pop(tenant, None)
                continue
            waiter = queue.popleft()
            self._count -= 1
            if queue:
                round_robin.append(tenant)
            else:
                tenant_queues.pop(tenant, None)
            if waiter.state != "queued" or waiter.future.done():
                continue
            waiter.state = "granted"
            self._active_turn[lane] = True
            self._grants += 1
            waiter.future.set_result(None)
            return

    def _remove(self, waiter: _BodyMemoryWaiter) -> None:
        if waiter.state == "granted":
            waiter.state = "abandoned"
            self.finish_turn(waiter.small_lane)
            return
        if waiter.state != "queued":
            return
        waiter.state = "abandoned"
        queue = self._queues[waiter.small_lane].get(waiter.tenant_key)
        if queue is not None:
            try:
                queue.remove(waiter)
                self._count -= 1
            except ValueError:
                pass
            if not queue:
                self._queues[waiter.small_lane].pop(waiter.tenant_key, None)


@dataclass(frozen=True, slots=True)
class _LargeBodyHolder:
    claim_id: str
    lease_id: str
    observation: RequestBodyObservation
    claimed_at_monotonic: float
    claimed_at_unix_ms: int
    reserved_weighted_bytes: int


async def _finish_cleanup_despite_cancellation(task: asyncio.Task[None]) -> None:
    """Wait for a tiny ownership cleanup even if cancellation is repeated."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def _cancel_and_release_acquisition(
    acquisition: asyncio.Future[Any],
) -> None:
    """Consume an acquisition whose awaiting owner was cancelled.

    A Future may become successful in the same loop turn that its waiter is
    cancelled.  Merely propagating cancellation can therefore strand the
    lease stored in the Future's result.  This helper owns that result until it
    is either cancelled/rejected or explicitly released.
    """

    if not acquisition.done():
        acquisition.cancel()
    try:
        abandoned_lease = await acquisition
    except (asyncio.CancelledError, AdmissionRejected):
        return
    await abandoned_lease.release()


async def _await_owned_acquisition(
    acquisition: asyncio.Future[Any],
) -> Any:
    """Await a lease Future without losing a same-turn successful result."""

    try:
        return await asyncio.shield(acquisition)
    except asyncio.CancelledError:
        cleanup_task = asyncio.create_task(
            _cancel_and_release_acquisition(acquisition)
        )
        await _finish_cleanup_despite_cancellation(cleanup_task)
        raise


class AdmissionLease:
    """One request-ownership lease.

    The backing gate may own a legacy count slot or a weighted control-memory
    reservation. Releasing is idempotent and cancellation-safe.
    """

    def __init__(self, gate: Any, *, wait_ms: float) -> None:
        self._gate = gate
        self.wait_ms = max(0.0, wait_ms)
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        release_task = self._release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            # Do not return control until ownership has actually been released,
            # even if shutdown or disconnect sends cancellation more than once.
            await _finish_cleanup_despite_cancellation(release_task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._gate._release_slot()
        self._released = True

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class BoundedAdmissionGate:
    """A FIFO active-capacity gate with a hard bound on queued tasks."""

    def __init__(
        self,
        capacity: int,
        *,
        waiter_limit: int,
        wait_timeout_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if waiter_limit < 0:
            raise ValueError("waiter_limit cannot be negative")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be greater than zero")
        self.capacity = capacity
        self.waiter_limit = waiter_limit
        self.wait_timeout_seconds = wait_timeout_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiters: deque[_Waiter] = deque()
        self._acquired_total = 0
        self._cancelled_total = 0
        self._rejected: Counter[str] = Counter()

    async def begin_acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> asyncio.Future[AdmissionLease]:
        """Atomically claim an active slot or bounded waiter position.

        The returned future is already running (or already resolved), so a
        caller may safely start ancillary work only after this method returns.
        Queue-full rejection happens before any such work can begin.
        """

        timeout = self.wait_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        started_at = self._clock()
        loop = asyncio.get_running_loop()
        waiter: _Waiter | None = None

        async with self._lock:
            # A new caller must not jump ahead of already queued callers.
            if self._active < self.capacity and not self._waiters:
                self._active += 1
                self._acquired_total += 1
                resolved: asyncio.Future[AdmissionLease] = loop.create_future()
                resolved.set_result(
                    AdmissionLease(
                        self,
                        wait_ms=(self._clock() - started_at) * 1000.0,
                    )
                )
                return resolved

            if len(self._waiters) >= self.waiter_limit:
                self._rejected["queue_full"] += 1
                raise AdmissionRejected("queue_full")

            waiter = _Waiter(loop.create_future())
            self._waiters.append(waiter)

        started = asyncio.Event()

        async def wait_for_waiter() -> AdmissionLease:
            started.set()
            return await self._wait_for_waiter(
                waiter,
                started_at=started_at,
                timeout=timeout,
            )

        task = asyncio.create_task(wait_for_waiter())
        try:
            await started.wait()
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, AdmissionRejected):
                await task
            # A task cancelled before its coroutine's first step never enters
            # _wait_for_waiter(), so it cannot run that coroutine's cleanup.
            # Explicitly abandon the waiter here as the outer coroutine still
            # owns the queue position.  _abandon_waiter is idempotent for the
            # case where the child did start and already cleaned itself up.
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(waiter, cancelled=True)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return task

    async def _wait_for_waiter(
        self,
        waiter: _Waiter,
        *,
        started_at: float,
        timeout: float,
    ) -> AdmissionLease:
        try:
            granted_at = await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=timeout,
            )
        except TimeoutError:
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(
                    waiter,
                    rejection_reason="wait_timeout",
                )
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise AdmissionRejected("wait_timeout") from None
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._abandon_waiter(waiter, cancelled=True)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise

        return AdmissionLease(
            self,
            wait_ms=(granted_at - started_at) * 1000.0,
        )

    async def acquire(self, *, timeout_seconds: float | None = None) -> AdmissionLease:
        acquisition = await self.begin_acquire(timeout_seconds=timeout_seconds)
        return await _await_owned_acquisition(acquisition)

    async def try_acquire(self) -> AdmissionLease | None:
        """Acquire immediately without joining the waiter queue."""

        started_at = self._clock()
        async with self._lock:
            if self._active >= self.capacity or self._waiters:
                return None
            self._active += 1
            self._acquired_total += 1
        return AdmissionLease(
            self,
            wait_ms=(self._clock() - started_at) * 1000.0,
        )

    async def _abandon_waiter(
        self,
        waiter: _Waiter,
        *,
        rejection_reason: str | None = None,
        cancelled: bool = False,
    ) -> None:
        async with self._lock:
            if waiter.state == "queued":
                waiter.state = "abandoned"
                self._waiters.remove(waiter)
                waiter.future.cancel()
            elif waiter.state == "granted":
                # Capacity was transferred to this waiter, but cancellation or
                # timeout won the race before a lease reached the caller.
                waiter.state = "abandoned"
                self._active -= 1
            else:
                return

            if rejection_reason is not None:
                self._rejected[rejection_reason] += 1
            if cancelled:
                self._cancelled_total += 1

            self._promote_waiters_locked()

    async def _release_slot(self) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("admission gate active count underflow")
            self._active -= 1
            self._promote_waiters_locked()

    def _promote_waiters_locked(self) -> None:
        while self._active < self.capacity and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.state != "queued" or waiter.future.done():
                continue
            waiter.state = "granted"
            self._active += 1
            self._acquired_total += 1
            waiter.future.set_result(self._clock())

    def snapshot(self) -> dict[str, Any]:
        """Return an event-loop-consistent metrics snapshot without awaiting."""

        return {
            "mode": "legacy_count",
            "active": self._active,
            "waiters": sum(waiter.state == "queued" for waiter in self._waiters),
            "capacity": self.capacity,
            "waiter_limit": self.waiter_limit,
            "wait_timeout_seconds": self.wait_timeout_seconds,
            "acquired_total": self._acquired_total,
            "cancelled_total": self._cancelled_total,
            "rejected": dict(self._rejected),
        }


class MemoryResourceAdmissionGate:
    """Own request control memory without imposing a request-count limit."""

    def __init__(
        self,
        memory_governor: AdaptiveMemoryGovernor,
        *,
        control_memory_bytes: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if control_memory_bytes <= 0:
            raise ValueError("control_memory_bytes must be positive")
        self._memory_governor = memory_governor
        self.control_memory_bytes = int(control_memory_bytes)
        self._clock = clock
        self._active = 0
        self._acquired_total = 0
        self._cancelled_total = 0
        self._rejected: Counter[str] = Counter()

    def _try_acquire_nowait(self, *, started_at: float) -> AdmissionLease | None:
        if not self._memory_governor.reserve_nowait(
            "request_control",
            self.control_memory_bytes,
        ):
            return None
        self._active += 1
        self._acquired_total += 1
        return AdmissionLease(
            self,
            wait_ms=(self._clock() - started_at) * 1000.0,
        )

    async def begin_acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> asyncio.Future[AdmissionLease]:
        del timeout_seconds
        started_at = self._clock()
        lease = self._try_acquire_nowait(started_at=started_at)
        if lease is None:
            self._rejected["memory_hard_guard"] += 1
            raise AdmissionRejected("memory_hard_guard")
        resolved: asyncio.Future[AdmissionLease] = (
            asyncio.get_running_loop().create_future()
        )
        resolved.set_result(lease)
        return resolved

    async def acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AdmissionLease:
        acquisition = await self.begin_acquire(
            timeout_seconds=timeout_seconds
        )
        return acquisition.result()

    async def try_acquire(self) -> AdmissionLease | None:
        return self._try_acquire_nowait(started_at=self._clock())

    async def _release_slot(self) -> None:
        if self._active <= 0:
            raise RuntimeError("resource admission active count underflow")
        self._active -= 1
        self._memory_governor.release(
            "request_control",
            self.control_memory_bytes,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "weighted_resources",
            "active": self._active,
            "waiters": 0,
            "capacity": None,
            "waiter_limit": None,
            "wait_timeout_seconds": None,
            "control_memory_bytes_per_request": self.control_memory_bytes,
            "control_memory_reserved_bytes": (
                self._active * self.control_memory_bytes
            ),
            "acquired_total": self._acquired_total,
            "cancelled_total": self._cancelled_total,
            "rejected": dict(self._rejected),
        }


class PendingBodyReservation:
    """Bytes consumed while a request is still waiting for an active slot."""

    def __init__(
        self,
        controller: RequestAdmissionController,
        *,
        fairness_key: str = "anonymous",
        resource_only_body_admission: bool = False,
    ) -> None:
        self._controller = controller
        self._reserved_bytes = 0
        self._transferred = False
        self._released = False
        self._large_body_slot = False
        self._large_body_claim_id: str | None = None
        self._lease_id = controller._new_lease_id()
        self._body_observation = RequestBodyObservation()
        self._release_reason = "pending_released"
        self._fairness_key = str(fairness_key or "anonymous")[:96]
        self._resource_only_body_admission = bool(
            resource_only_body_admission
        )

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def observe_body(self, observation: RequestBodyObservation) -> None:
        if self._transferred or self._released:
            return
        self._body_observation = observation

    async def reserve(
        self,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._transferred or self._released:
            raise RuntimeError("pending body reservation is no longer active")
        await self._controller._reserve_pending_body_additional(
            self,
            additional_bytes,
            observation=observation,
        )
        return self._reserved_bytes

    async def transfer_to(self, lease: RequestAdmissionLease) -> int:
        if self._released:
            raise RuntimeError("pending body reservation is released")
        if self._transferred:
            return 0
        transferred = await self._controller._transfer_pending_body(
            self,
            lease,
        )
        self._transferred = True
        return transferred

    async def release(self, *, reason: str = "pending_released") -> None:
        if self._transferred or self._released:
            return
        self._release_reason = str(reason or "pending_released")
        cleanup_task = asyncio.create_task(
            self._controller._release_pending_body(self)
        )
        await _finish_cleanup_despite_cancellation(cleanup_task)
        self._released = True


class RequestAdmissionLease:
    """An active request slot plus its current body-byte reservation."""

    def __init__(
        self,
        controller: RequestAdmissionController,
        active_lease: AdmissionLease,
        *,
        reserved_body_bytes: int,
        resource_only_body_admission: bool = False,
    ) -> None:
        self._controller = controller
        self._active_lease = active_lease
        self._reserved_body_bytes = reserved_body_bytes
        self._reserved_response_bytes = 0
        self._release_requested = False
        self._released = False
        self._memory_owner_count = 0
        self._memory_finalized = False
        self._large_body_slot = False
        self._large_body_claim_id: str | None = None
        self._lease_id = controller._new_lease_id()
        self._body_observation = RequestBodyObservation()
        self._release_reason = "request_completed"
        self._release_task: asyncio.Task[None] | None = None
        self._fairness_key = "anonymous"
        self._resource_only_body_admission = bool(
            resource_only_body_admission
        )
        self._response_attempt: dict[str, Any] | None = None
        self._response_attempts_started = 0
        self._response_committed_allocations: dict[str, dict[str, Any]] = {}
        self._response_lifecycle_by_attempt: dict[str, dict[str, Any]] = {}
        self._deferred_response_summaries: dict[
            str,
            tuple[ResponseBufferEvent, asyncio.TimerHandle],
        ] = {}

    @property
    def wait_ms(self) -> float:
        return self._active_lease.wait_ms

    @property
    def reserved_body_bytes(self) -> int:
        return self._reserved_body_bytes

    @property
    def reserved_response_bytes(self) -> int:
        return self._reserved_response_bytes

    @property
    def released(self) -> bool:
        return self._released

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def observe_body(self, observation: RequestBodyObservation) -> None:
        if self._release_requested or self._released:
            return
        self._body_observation = observation

    def set_fairness_key(self, value: str) -> None:
        if self._release_requested or self._released:
            return
        self._fairness_key = str(value or "anonymous")[:96]

    def begin_response_attempt(
        self,
        entry: dict[str, Any] | None,
        *,
        routing_attempt_id: str,
        routing_attempt_index: int | None,
        provider: str,
        request_model: str,
        actual_model: str,
        transport_diagnostics: Any = None,
    ) -> None:
        """Bind response-memory ownership to the currently executing retry."""

        if self._release_requested or self._released:
            return
        self._response_attempts_started += 1
        normalized_index = (
            int(routing_attempt_index)
            if isinstance(routing_attempt_index, int)
            else self._response_attempts_started
        )
        retained_before = self._reserved_response_bytes
        lifecycle_key = (
            str(routing_attempt_id or "").strip()
            or f"attempt-{normalized_index}"
        )[:128]
        if retained_before > 0:
            for previous in self._response_lifecycle_by_attempt.values():
                if (
                    int(previous.get("committed_bytes") or 0)
                    > int(previous.get("released_bytes") or 0)
                ):
                    previous["held_across_retry"] = True
        attempt = {
            "routing_attempt_id": str(routing_attempt_id or "")[:128] or None,
            "routing_attempt_index": normalized_index,
            "provider": str(provider or "")[:256] or None,
            "request_model": str(request_model or "")[:256] or None,
            "actual_model": str(actual_model or "")[:256] or None,
            "retained_before_bytes": retained_before,
            "lifecycle_key": lifecycle_key,
            "entry": entry,
            "transport_diagnostics": transport_diagnostics,
        }
        self._response_attempt = attempt
        self._response_lifecycle_by_attempt.setdefault(
            lifecycle_key,
            {
                **attempt,
                "request_response_before": retained_before,
                "request_response_projected": retained_before,
                "request_response_after": retained_before,
                "global_response_before": self._controller._reserved_response_bytes,
                "global_response_projected": self._controller._reserved_response_bytes,
                "global_response_after": self._controller._reserved_response_bytes,
                "reserve_started_count": 0,
                "reserve_call_count": 0,
                "requested_bytes": 0,
                "commit_count": 0,
                "committed_bytes": 0,
                "rollback_count": 0,
                "rolled_back_bytes": 0,
                "release_count": 0,
                "released_bytes": 0,
                "rejection_count": 0,
                "outcome": "started",
                "held_across_retry": False,
            },
        )
        if isinstance(entry, dict):
            entry["response_buffer_reserved_before_bytes"] = retained_before
            entry["response_buffer_retained_from_prior_attempts_bytes"] = (
                retained_before
            )
            entry["response_buffer_cross_retry_retained"] = bool(
                normalized_index > 1 and retained_before > 0
            )

    def finish_response_attempt(
        self,
        *,
        outcome: str,
        keep_active: bool = False,
    ) -> None:
        attempt = self._response_attempt
        if not isinstance(attempt, dict):
            return
        entry = attempt.get("entry")
        before = int(attempt.get("retained_before_bytes") or 0)
        after = self._reserved_response_bytes
        attempt_id = str(attempt.get("routing_attempt_id") or "")
        committed_by_attempt = sum(
            int(allocation.get("bytes") or 0)
            for allocation in self._response_committed_allocations.values()
            if str(allocation.get("routing_attempt_id") or "") == attempt_id
        )
        if isinstance(entry, dict):
            entry["response_buffer_reserved_after_bytes"] = after
            entry["response_buffer_reserved_delta_bytes"] = after - before
            entry["response_buffer_committed_by_attempt_bytes"] = (
                committed_by_attempt
            )
            entry["response_buffer_retained_after_attempt"] = after > 0
            entry["response_buffer_retained_after_failed_attempt"] = bool(
                outcome not in {"succeeded", "stream_pending"} and after > 0
            )
        lifecycle_key = str(attempt.get("lifecycle_key") or "")
        lifecycle = self._response_lifecycle_by_attempt.get(lifecycle_key)
        if isinstance(lifecycle, dict):
            lifecycle["outcome"] = str(outcome)[:80]
            lifecycle["request_response_after"] = after
            lifecycle["global_response_after"] = (
                self._controller._reserved_response_bytes
            )
        diagnostics = attempt.get("transport_diagnostics")
        finalize_diagnostics = getattr(diagnostics, "finalize", None)
        if callable(finalize_diagnostics):
            try:
                finalize_diagnostics(str(outcome)[:80])
            except Exception:
                # Transport diagnostics are strictly fail-open and must never
                # alter response-memory ownership or the downstream response.
                pass
        if not keep_active:
            self._controller._complete_deferred_response_summary(
                self,
                lifecycle_key,
                outcome=str(outcome)[:80],
            )
            self._response_attempt = None

    def finish_pending_response_attempt(self, *, outcome: str) -> bool:
        """Finalize only an attempt still waiting on downstream stream end."""

        attempt = self._response_attempt
        if not isinstance(attempt, dict):
            return False
        lifecycle_key = str(attempt.get("lifecycle_key") or "")
        lifecycle = self._response_lifecycle_by_attempt.get(lifecycle_key)
        if not isinstance(lifecycle, dict):
            return False
        if str(lifecycle.get("outcome") or "") != "stream_pending":
            return False
        self.finish_response_attempt(outcome=str(outcome)[:80])
        return True

    async def reserve_body_bytes(
        self,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> int:
        """Atomically grow this request's reservation by an observed chunk."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        await self._controller._reserve_additional(
            self,
            additional_bytes,
            observation=observation,
        )
        return self._reserved_body_bytes

    async def release_body_bytes(self, released_bytes: int) -> int:
        """Release an owned request-memory phase after dropping its objects."""

        if released_bytes < 0:
            raise ValueError("released_bytes cannot be negative")
        if self._release_requested:
            return self._reserved_body_bytes
        await self._controller._release_body_bytes(self, released_bytes)
        return self._reserved_body_bytes

    async def reserve_response_bytes(self, additional_bytes: int) -> int:
        """Reserve weighted retained bytes for a buffered upstream response."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        await self._controller._reserve_response_additional(
            self,
            additional_bytes,
        )
        return self._reserved_response_bytes

    async def reserve_temporary_response_bytes(
        self,
        additional_bytes: int,
    ) -> TemporaryResponseBytesReservation:
        """Reserve live parse/materialization memory with explicit lifetime."""

        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        if self._release_requested:
            raise RuntimeError("cannot reserve bytes on a released request lease")
        reservation = TemporaryResponseBytesReservation(
            self,
            0,
            allocation_id=self._controller._new_response_allocation_id(),
        )
        activate_task = asyncio.create_task(
            self._controller._activate_temporary_response(
                reservation,
                additional_bytes,
            )
        )
        try:
            await asyncio.shield(activate_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(activate_task)
            if reservation._active:
                cleanup_task = asyncio.create_task(reservation.release())
                await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return reservation

    async def defer_memory_release(self) -> MemoryReleaseDeferral:
        """Keep request memory accounted after its active slot is released."""

        deferral = MemoryReleaseDeferral(self)
        activate_task = asyncio.create_task(
            self._controller._activate_memory_deferral(deferral)
        )
        try:
            await asyncio.shield(activate_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(activate_task)
            if deferral._active:
                cleanup_task = asyncio.create_task(deferral.release())
                await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return deferral

    async def release(self, *, reason: str = "request_completed") -> None:
        self._release_requested = True
        if self._release_task is None:
            self._release_reason = str(reason or "request_completed")
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        release_task = self._release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(release_task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._controller._release_request(self)
        self._released = True

    async def __aenter__(self) -> RequestAdmissionLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class TemporaryResponseBytesReservation:
    """A response-memory charge released as soon as its object graph dies."""

    def __init__(
        self,
        lease: RequestAdmissionLease,
        size: int,
        *,
        allocation_id: str,
    ) -> None:
        self._lease = lease
        self.size = int(size)
        self._allocation_id = allocation_id
        self._reserve_call_count = 0
        self._attempt: dict[str, Any] | None = None
        self._active = False
        self._released = False
        self._committed = False
        self._closing = False
        self._state_lock = asyncio.Lock()
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._committed or self._released:
            return
        if self._release_task is None:
            # Establish exact-once finalization before the first await.  The
            # coordinator itself waits for any in-flight grow/commit holder.
            self._closing = True
            self._release_task = asyncio.create_task(
                self._release_coordinated()
            )
        task = self._release_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(task)
            raise

    async def _release_coordinated(self) -> None:
        async with self._state_lock:
            if self._committed or self._released:
                return
            await self._release_once()

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._lease._controller._release_response_bytes(
            self._lease,
            self,
        )
        self._released = True

    async def reserve(self, additional_bytes: int) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes cannot be negative")
        async with self._state_lock:
            if self._released or self._committed or self._closing:
                raise RuntimeError("temporary response reservation is closed")
            await self._lease._controller._grow_temporary_response(
                self,
                additional_bytes,
            )
            return self.size

    async def commit(self) -> int:
        """Transfer ownership to the surrounding request lease."""

        async with self._state_lock:
            if self._released or self._closing:
                raise RuntimeError("cannot commit a released reservation")
            if self._committed:
                return self.size
            await self._lease._controller._commit_temporary_response(self)
            return self.size

    async def __aenter__(self) -> TemporaryResponseBytesReservation:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class MemoryReleaseDeferral:
    """Child ownership keeping base request memory charged after active exit."""

    def __init__(self, lease: RequestAdmissionLease) -> None:
        self._lease = lease
        self._active = False
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        task = self._release_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(task)
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._lease._controller._finish_memory_owner(self._lease)
        self._released = True


class RequestAdmissionController:
    """Own request control memory and all retained request bytes."""

    def __init__(
        self,
        *,
        capacity: int | None,
        waiter_limit: int | None,
        wait_timeout_seconds: float,
        control_memory_bytes: int = 0,
        max_body_bytes: int,
        body_budget_bytes: int,
        max_response_bytes: int | None = None,
        max_retained_bytes_per_request: int | None = None,
        large_body_threshold_weighted_bytes: int = 0,
        large_body_limit: int = 0,
        body_memory_wait_timeout_seconds: float = 0.0,
        body_memory_waiter_limit: int | None = 0,
        small_body_lane_threshold_bytes: int = 256 * 1024,
        small_body_lane_reserve_bytes: int = 0,
        memory_governor: AdaptiveMemoryGovernor | None = None,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
        decision_observer: Callable[[LargeBodyAdmissionDecision], bool | None]
        | None = None,
        response_buffer_observer: Callable[[ResponseBufferEvent], bool | None]
        | None = None,
        decision_history_limit: int = 64,
        response_event_history_limit: int = 128,
    ) -> None:
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes cannot be negative")
        if body_budget_bytes < 0:
            raise ValueError("body_budget_bytes cannot be negative")
        if max_response_bytes is not None and max_response_bytes < 0:
            raise ValueError("max_response_bytes cannot be negative")
        if large_body_threshold_weighted_bytes < 0 or large_body_limit < 0:
            raise ValueError("large body admission settings cannot be negative")
        if body_memory_wait_timeout_seconds < 0 or (
            body_memory_waiter_limit is not None
            and body_memory_waiter_limit < 0
        ):
            raise ValueError("body memory wait settings cannot be negative")
        if small_body_lane_threshold_bytes < 0 or small_body_lane_reserve_bytes < 0:
            raise ValueError("small body lane settings cannot be negative")
        if small_body_lane_reserve_bytes > body_budget_bytes:
            raise ValueError("small body lane reserve cannot exceed body budget")
        if bool(large_body_threshold_weighted_bytes) != bool(large_body_limit):
            raise ValueError(
                "large body threshold and limit must both be enabled or disabled"
            )
        if decision_history_limit <= 0:
            raise ValueError("decision_history_limit must be greater than zero")
        if response_event_history_limit <= 0:
            raise ValueError("response_event_history_limit must be greater than zero")

        self.max_body_bytes = max_body_bytes
        self.body_budget_bytes = body_budget_bytes
        self.max_response_bytes = (
            max_body_bytes if max_response_bytes is None else max_response_bytes
        )
        self.max_retained_bytes_per_request = (
            max(self.max_body_bytes, self.max_response_bytes)
            if max_retained_bytes_per_request is None
            else int(max_retained_bytes_per_request)
        )
        self.large_body_threshold_weighted_bytes = int(
            large_body_threshold_weighted_bytes
        )
        self.large_body_limit = int(large_body_limit)
        self.body_memory_wait_timeout_seconds = float(
            body_memory_wait_timeout_seconds
        )
        self.body_memory_waiter_limit = (
            int(body_memory_waiter_limit)
            if body_memory_waiter_limit is not None
            else None
        )
        self.small_body_lane_threshold_bytes = int(
            small_body_lane_threshold_bytes
        )
        self.small_body_lane_reserve_bytes = int(
            small_body_lane_reserve_bytes
        )
        self._body_memory_wait_queue = _FairBodyMemoryWaitQueue(
            self.body_memory_waiter_limit
        )
        self._large_body_holders: dict[str, _LargeBodyHolder] = {}
        if self.max_retained_bytes_per_request < 0:
            raise ValueError("max_retained_bytes_per_request cannot be negative")
        self._memory_governor = memory_governor
        self._clock = clock
        self._wall_clock = wall_clock
        self._decision_observer = decision_observer
        self._response_buffer_observer = response_buffer_observer
        self._decision_history: deque[LargeBodyAdmissionDecision] = deque(
            maxlen=int(decision_history_limit)
        )
        self._decision_sequence = 0
        self._decision_history_overwritten = 0
        self._decision_record_failures = 0
        self._decision_observer_errors = 0
        self._decision_observer_enqueue_failures = 0
        self._identity_nonce = uuid4().hex
        self._lease_sequence = 0
        self._claim_sequence = 0
        self._response_allocation_sequence = 0
        self._response_event_sequence = 0
        self._response_event_history: deque[ResponseBufferEvent] = deque(
            maxlen=int(response_event_history_limit)
        )
        self._response_event_history_overwritten = 0
        self._response_event_record_failures = 0
        self._response_event_observer_errors = 0
        self._response_event_observer_enqueue_failures = 0
        self._response_rejected_by_branch: Counter[str] = Counter()
        self._response_rejection_timestamps: deque[float] = deque()
        if capacity is None:
            if memory_governor is None:
                raise ValueError(
                    "weighted request admission requires a memory governor"
                )
            self._active_gate = MemoryResourceAdmissionGate(
                memory_governor,
                control_memory_bytes=control_memory_bytes,
                clock=clock,
            )
        else:
            if waiter_limit is None:
                raise ValueError(
                    "legacy request admission requires a waiter limit"
                )
            self._active_gate = BoundedAdmissionGate(
                capacity,
                waiter_limit=waiter_limit,
                wait_timeout_seconds=wait_timeout_seconds,
                clock=clock,
            )
        self._body_lock = asyncio.Lock()
        self._reserved_body_bytes = 0
        self._pending_body_reserved_bytes = 0
        self._reserved_response_bytes = 0
        self._staged_body_release_count = 0
        self._staged_body_released_bytes = 0
        self._deferred_memory_leases: set[RequestAdmissionLease] = set()
        self._body_rejected: Counter[str] = Counter()
        self._response_rejected: Counter[str] = Counter()

    def _new_lease_id(self) -> str:
        self._lease_sequence += 1
        return f"{self._identity_nonce}-lease-{self._lease_sequence:x}"

    def _new_claim_id_locked(self) -> str:
        self._claim_sequence += 1
        return f"{self._identity_nonce}-claim-{self._claim_sequence:x}"

    def _new_response_allocation_id(self) -> str:
        self._response_allocation_sequence += 1
        return (
            f"{self._identity_nonce}-response-"
            f"{self._response_allocation_sequence:x}"
        )

    async def acquire(
        self,
        *,
        initial_body_bytes: int = 0,
        timeout_seconds: float | None = None,
        resource_only_body_admission: bool = False,
    ) -> RequestAdmissionLease:
        if initial_body_bytes < 0:
            raise ValueError("initial_body_bytes cannot be negative")

        if resource_only_body_admission:
            acquisition = await self.begin_acquire(
                timeout_seconds=timeout_seconds,
                resource_only_body_admission=True,
            )
        else:
            # Keep the ordinary path source-compatible with instrumentation
            # wrappers that implement the long-standing timeout-only call.
            acquisition = await self.begin_acquire(
                timeout_seconds=timeout_seconds,
            )
        lease = await _await_owned_acquisition(acquisition)
        if initial_body_bytes == 0:
            return lease
        try:
            await lease.reserve_body_bytes(initial_body_bytes)
        except BaseException:
            await lease.release()
            raise
        return lease

    async def begin_acquire(
        self,
        *,
        timeout_seconds: float | None = None,
        resource_only_body_admission: bool = False,
    ) -> asyncio.Future[RequestAdmissionLease]:
        active_acquisition = await self._active_gate.begin_acquire(
            timeout_seconds=timeout_seconds
        )
        loop = asyncio.get_running_loop()
        if active_acquisition.done():
            active_lease = active_acquisition.result()
            resolved: asyncio.Future[RequestAdmissionLease] = loop.create_future()
            resolved.set_result(
                RequestAdmissionLease(
                    self,
                    active_lease,
                    reserved_body_bytes=0,
                    resource_only_body_admission=(
                        resource_only_body_admission
                    ),
                )
            )
            return resolved

        started = asyncio.Event()

        async def convert() -> RequestAdmissionLease:
            started.set()
            active_lease = await _await_owned_acquisition(active_acquisition)
            return RequestAdmissionLease(
                self,
                active_lease,
                reserved_body_bytes=0,
                resource_only_body_admission=resource_only_body_admission,
            )

        task = asyncio.create_task(convert())
        try:
            await started.wait()
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, AdmissionRejected):
                await task
            # As above, cancellation may happen before convert() executes its
            # first line.  In that window only this outer coroutine can return
            # the inner waiter/lease, so never rely solely on convert()'s
            # cancellation handler.
            cleanup_task = asyncio.create_task(
                _cancel_and_release_acquisition(active_acquisition)
            )
            await _finish_cleanup_despite_cancellation(cleanup_task)
            raise
        return task

    async def try_acquire(
        self,
        *,
        resource_only_body_admission: bool = False,
    ) -> RequestAdmissionLease | None:
        active_lease = await self._active_gate.try_acquire()
        if active_lease is None:
            return None
        return RequestAdmissionLease(
            self,
            active_lease,
            reserved_body_bytes=0,
            resource_only_body_admission=resource_only_body_admission,
        )

    def pending_body_reservation(
        self,
        *,
        fairness_key: str = "anonymous",
        resource_only_body_admission: bool = False,
    ) -> PendingBodyReservation:
        return PendingBodyReservation(
            self,
            fairness_key=fairness_key,
            resource_only_body_admission=resource_only_body_admission,
        )

    async def _reserve_pending_body_additional(
        self,
        reservation: PendingBodyReservation,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> None:
        await self._reserve_body_owner(
            reservation,
            additional_bytes,
            observation=observation,
            pending=True,
        )

    async def _transfer_pending_body(
        self,
        reservation: PendingBodyReservation,
        lease: RequestAdmissionLease,
    ) -> int:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    raise RuntimeError("cannot transfer bytes to a released request lease")
                transferred = reservation._reserved_bytes
                request_before = lease._reserved_body_bytes
                next_request_bytes = request_before + transferred
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                if (
                    not lease._resource_only_body_admission
                    and next_request_bytes > self.max_body_bytes
                ):
                    raise RuntimeError("pending body transfer exceeds request limit")
                if (
                    reservation._resource_only_body_admission
                    != lease._resource_only_body_admission
                ):
                    raise RuntimeError(
                        "pending body admission mode changed during transfer"
                    )
                if transferred > self._pending_body_reserved_bytes:
                    raise RuntimeError("pending body reservation underflow")
                lease._body_observation = reservation._body_observation
                if reservation._large_body_slot:
                    if lease._large_body_slot:
                        raise RuntimeError("large request admission double transfer")
                    claim_id = reservation._large_body_claim_id
                    if not claim_id or claim_id not in self._large_body_holders:
                        raise RuntimeError("large request holder missing during transfer")
                    lease._large_body_slot = True
                    lease._large_body_claim_id = claim_id
                    reservation._large_body_slot = False
                    reservation._large_body_claim_id = None
                    holder = self._large_body_holders[claim_id]
                    self._large_body_holders[claim_id] = replace(
                        holder,
                        lease_id=lease._lease_id,
                        observation=lease._body_observation,
                        reserved_weighted_bytes=next_request_bytes,
                    )
                else:
                    self._ensure_large_body_slot_available_locked(
                        lease,
                        next_request_bytes,
                        request_before=request_before,
                        global_body_before=global_body_before,
                        global_response_before=global_response_before,
                        decision_events=decision_events,
                    )
                lease._reserved_body_bytes = next_request_bytes
                reservation._reserved_bytes = 0
                self._pending_body_reserved_bytes -= transferred
                self._claim_large_body_slot_locked(
                    lease,
                    next_request_bytes,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    decision_events=decision_events,
                )
                self._update_large_body_holder_locked(lease, next_request_bytes)
                return transferred
        finally:
            self._publish_decision_events(decision_events)

    async def _release_pending_body(
        self,
        reservation: PendingBodyReservation,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                released = reservation._reserved_bytes
                request_before = released
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                reservation._reserved_bytes = 0
                if released > self._pending_body_reserved_bytes:
                    raise RuntimeError("pending body reservation underflow")
                if released > self._reserved_body_bytes:
                    raise RuntimeError("request body reservation underflow")
                self._pending_body_reserved_bytes -= released
                self._reserved_body_bytes -= released
                self._release_parent_memory("request_body", released)
                self._release_large_body_slot_locked(
                    reservation,
                    request_before=request_before,
                    global_body_before=global_body_before,
                    global_response_before=global_response_before,
                    release_reason=reservation._release_reason,
                    release_finalizer="pending_release",
                    decision_events=decision_events,
                )
        finally:
            self._publish_decision_events(decision_events)

    async def _release_body_bytes(
        self,
        lease: RequestAdmissionLease,
        released_bytes: int,
    ) -> None:
        released = int(released_bytes)
        if released == 0:
            return
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    return
                if released > lease._reserved_body_bytes:
                    raise RuntimeError("request body staged release underflow")
                request_before = lease._reserved_body_bytes
                global_body_before = self._reserved_body_bytes
                global_response_before = self._reserved_response_bytes
                if released > self._reserved_body_bytes:
                    raise RuntimeError("global request body staged release underflow")
                lease._reserved_body_bytes -= released
                self._reserved_body_bytes -= released
                self._staged_body_release_count += 1
                self._staged_body_released_bytes += released
                self._release_parent_memory("request_body", released)
                if lease._reserved_body_bytes == 0:
                    self._release_large_body_slot_locked(
                        lease,
                        request_before=request_before,
                        global_body_before=global_body_before,
                        global_response_before=global_response_before,
                        release_reason="request_body_objects_released",
                        release_finalizer="staged_body_release",
                        decision_events=decision_events,
                    )
                else:
                    self._update_large_body_holder_locked(
                        lease,
                        lease._reserved_body_bytes,
                    )
        finally:
            self._publish_decision_events(decision_events)

    async def _reserve_additional(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None = None,
    ) -> None:
        await self._reserve_body_owner(
            lease,
            additional_bytes,
            observation=observation,
            pending=False,
        )

    async def _reserve_body_owner(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        additional_bytes: int,
        *,
        observation: RequestBodyObservation | None,
        pending: bool,
    ) -> None:
        """Reserve without awaiting the parent governor under ``_body_lock``."""

        additional = int(additional_bytes)
        if additional < 0:
            raise ValueError("additional_bytes cannot be negative")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.body_memory_wait_timeout_seconds
        wait_lane = True
        while True:
            decision_events: list[LargeBodyAdmissionDecision] = []
            unavailable_reason: str | None = None
            parent_reserved = False
            try:
                async with self._body_lock:
                    request_before = self._owner_reserved_body_bytes(owner, pending)
                    if observation is not None:
                        owner._body_observation = observation
                    self._validate_body_owner_active(owner, pending)
                    next_request_bytes = request_before + additional
                    self._validate_body_hard_limits_locked(
                        owner,
                        next_request_bytes,
                        pending=pending,
                    )
                    wait_lane = self._is_small_body_lane(next_request_bytes)
                    unavailable_reason = self._body_unavailable_reason_locked(
                        owner,
                        next_request_bytes,
                        additional,
                    )
                    if (
                        unavailable_reason is None
                        and request_before == 0
                        and self._body_memory_wait_queue.has_waiters(wait_lane)
                    ):
                        unavailable_reason = "body_budget_exhausted"
                    if unavailable_reason is None:
                        parent_reserved = self._reserve_parent_memory(
                            "request_body",
                            additional,
                        )
                        if not parent_reserved:
                            unavailable_reason = "body_budget_exhausted"
                    if unavailable_reason is None:
                        self._commit_body_owner_locked(
                            owner,
                            next_request_bytes,
                            additional,
                            pending=pending,
                            request_before=request_before,
                            decision_events=decision_events,
                        )
                        return
            finally:
                self._publish_decision_events(decision_events)

            remaining = deadline - loop.time()
            if remaining <= 0 or self.body_memory_waiter_limit == 0:
                await self._raise_body_wait_rejection(
                    owner,
                    additional,
                    pending=pending,
                    reason=unavailable_reason or "body_budget_exhausted",
                )
            if unavailable_reason != "body_budget_exhausted" or (
                self._reserved_body_bytes
                + self._reserved_response_bytes
                + additional
                > self._body_budget_for_lane(wait_lane)
            ):
                self._body_memory_wait_queue.block_until_release(wait_lane)
            try:
                await self._body_memory_wait_queue.wait_turn(
                    tenant_key=owner._fairness_key,
                    small_lane=wait_lane,
                    timeout_seconds=remaining,
                )
            except TimeoutError:
                await self._raise_body_wait_rejection(
                    owner,
                    additional,
                    pending=pending,
                    reason=unavailable_reason or "body_budget_exhausted",
                )
            outcome = await self._commit_body_owner_after_wait_turn(
                owner,
                additional,
                pending=pending,
                wait_lane=wait_lane,
                deadline=deadline,
            )
            if outcome == "committed":
                return
            if outcome == "timeout":
                await self._raise_body_wait_rejection(
                    owner,
                    additional,
                    pending=pending,
                    reason="body_budget_exhausted",
                )

    async def _commit_body_owner_after_wait_turn(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        additional_bytes: int,
        *,
        pending: bool,
        wait_lane: bool,
        deadline: float,
    ) -> Literal["committed", "retry", "timeout"]:
        parent_reserved = False
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            if self._memory_governor is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0 or not await self._memory_governor.reserve(
                    "request_body",
                    additional_bytes,
                    timeout_seconds=remaining,
                ):
                    return "timeout"
                parent_reserved = True
            async with self._body_lock:
                request_before = self._owner_reserved_body_bytes(owner, pending)
                self._validate_body_owner_active(owner, pending)
                next_request_bytes = request_before + additional_bytes
                self._validate_body_hard_limits_locked(
                    owner,
                    next_request_bytes,
                    pending=pending,
                )
                unavailable_reason = self._body_unavailable_reason_locked(
                    owner,
                    next_request_bytes,
                    additional_bytes,
                )
                if unavailable_reason is not None:
                    return "retry"
                self._commit_body_owner_locked(
                    owner,
                    next_request_bytes,
                    additional_bytes,
                    pending=pending,
                    request_before=request_before,
                    decision_events=decision_events,
                )
                parent_reserved = False
                return "committed"
        finally:
            self._publish_decision_events(decision_events)
            if parent_reserved:
                self._release_parent_memory("request_body", additional_bytes)
            self._body_memory_wait_queue.finish_turn(wait_lane)

    def _owner_reserved_body_bytes(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        pending: bool,
    ) -> int:
        return owner._reserved_bytes if pending else owner._reserved_body_bytes

    @staticmethod
    def _validate_body_owner_active(
        owner: RequestAdmissionLease | PendingBodyReservation,
        pending: bool,
    ) -> None:
        if pending:
            if owner._transferred or owner._released:
                raise RuntimeError("pending body reservation is no longer active")
            return
        if owner._release_requested or owner._released:
            raise RuntimeError("cannot reserve bytes on a released request lease")

    def _validate_body_hard_limits_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        *,
        pending: bool,
    ) -> None:
        if (
            not owner._resource_only_body_admission
            and next_request_bytes > self.max_body_bytes
        ):
            self._body_rejected["body_too_large"] += 1
            raise RequestBodyTooLarge()
        if (
            not pending
            and not owner._resource_only_body_admission
            and next_request_bytes + owner._reserved_response_bytes
            > self.max_retained_bytes_per_request
        ):
            self._body_rejected["body_budget_exhausted"] += 1
            raise RequestBodyBudgetExhausted()

    def _is_small_body_lane(self, next_request_bytes: int) -> bool:
        return bool(
            self.small_body_lane_threshold_bytes
            and next_request_bytes <= self.small_body_lane_threshold_bytes
        )

    def _body_budget_for_lane(self, small_lane: bool) -> int:
        if small_lane:
            return self.body_budget_bytes
        return max(0, self.body_budget_bytes - self.small_body_lane_reserve_bytes)

    def _body_unavailable_reason_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        additional_bytes: int,
    ) -> str | None:
        if (
            self._reserved_body_bytes
            + self._reserved_response_bytes
            + additional_bytes
            > self._body_budget_for_lane(
                self._is_small_body_lane(next_request_bytes)
            )
        ):
            return "body_budget_exhausted"
        if (
            self.large_body_limit
            and not owner._large_body_slot
            and next_request_bytes > self.large_body_threshold_weighted_bytes
            and len(self._large_body_holders) >= self.large_body_limit
        ):
            return "large_body_capacity_exhausted"
        return None

    def _commit_body_owner_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        additional_bytes: int,
        *,
        pending: bool,
        request_before: int,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        global_body_before = self._reserved_body_bytes
        global_response_before = self._reserved_response_bytes
        if pending:
            owner._reserved_bytes = next_request_bytes
            self._pending_body_reserved_bytes += additional_bytes
        else:
            owner._reserved_body_bytes = next_request_bytes
        self._reserved_body_bytes += additional_bytes
        self._claim_large_body_slot_locked(
            owner,
            next_request_bytes,
            request_before=request_before,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            decision_events=decision_events,
        )
        self._update_large_body_holder_locked(owner, next_request_bytes)

    async def _raise_body_wait_rejection(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        additional_bytes: int,
        *,
        pending: bool,
        reason: str,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        try:
            async with self._body_lock:
                request_before = self._owner_reserved_body_bytes(owner, pending)
                if reason == "large_body_capacity_exhausted":
                    self._ensure_large_body_slot_available_locked(
                        owner,
                        request_before + additional_bytes,
                        request_before=request_before,
                        global_body_before=self._reserved_body_bytes,
                        global_response_before=self._reserved_response_bytes,
                        decision_events=decision_events,
                    )
                self._body_rejected["body_budget_exhausted"] += 1
                raise RequestBodyBudgetExhausted()
        finally:
            self._publish_decision_events(decision_events)

    def _response_lifecycle_locked(
        self,
        lease: RequestAdmissionLease,
        attempt: dict[str, Any] | None,
    ) -> dict[str, Any]:
        attempt = attempt if isinstance(attempt, dict) else {}
        key = str(
            attempt.get("lifecycle_key")
            or attempt.get("routing_attempt_id")
            or "unattributed"
        )[:128]
        lifecycle = lease._response_lifecycle_by_attempt.get(key)
        if lifecycle is None:
            lifecycle = {
                **attempt,
                "lifecycle_key": key,
                "request_response_before": lease._reserved_response_bytes,
                "request_response_projected": lease._reserved_response_bytes,
                "request_response_after": lease._reserved_response_bytes,
                "global_response_before": self._reserved_response_bytes,
                "global_response_projected": self._reserved_response_bytes,
                "global_response_after": self._reserved_response_bytes,
                "reserve_started_count": 0,
                "reserve_call_count": 0,
                "requested_bytes": 0,
                "commit_count": 0,
                "committed_bytes": 0,
                "rollback_count": 0,
                "rolled_back_bytes": 0,
                "release_count": 0,
                "released_bytes": 0,
                "rejection_count": 0,
                "outcome": "unattributed",
                "held_across_retry": False,
            }
            lease._response_lifecycle_by_attempt[key] = lifecycle
        return lifecycle

    @staticmethod
    def _increment_lifecycle(
        lifecycle: dict[str, Any],
        key: str,
        value: int,
    ) -> None:
        lifecycle[key] = int(lifecycle.get(key) or 0) + int(value)

    def _update_response_lifecycle_locked(
        self,
        lease: RequestAdmissionLease,
        attempt: dict[str, Any] | None,
        *,
        reserve_started: int = 0,
        reserve_calls: int = 0,
        requested_bytes: int = 0,
        commits: int = 0,
        committed_bytes: int = 0,
        rollbacks: int = 0,
        rolled_back_bytes: int = 0,
        releases: int = 0,
        released_bytes: int = 0,
        rejections: int = 0,
        request_response_projected: int | None = None,
        request_response_after: int | None = None,
        global_response_projected: int | None = None,
        global_response_after: int | None = None,
    ) -> dict[str, Any]:
        lifecycle = self._response_lifecycle_locked(lease, attempt)
        for key, value in (
            ("reserve_started_count", reserve_started),
            ("reserve_call_count", reserve_calls),
            ("requested_bytes", requested_bytes),
            ("commit_count", commits),
            ("committed_bytes", committed_bytes),
            ("rollback_count", rollbacks),
            ("rolled_back_bytes", rolled_back_bytes),
            ("release_count", releases),
            ("released_bytes", released_bytes),
            ("rejection_count", rejections),
        ):
            if value:
                self._increment_lifecycle(lifecycle, key, value)
        if request_response_projected is not None:
            lifecycle["request_response_projected"] = max(
                int(lifecycle.get("request_response_projected") or 0),
                int(request_response_projected),
            )
        if request_response_after is not None:
            lifecycle["request_response_after"] = int(request_response_after)
        if global_response_projected is not None:
            lifecycle["global_response_projected"] = max(
                int(lifecycle.get("global_response_projected") or 0),
                int(global_response_projected),
            )
        if global_response_after is not None:
            lifecycle["global_response_after"] = int(global_response_after)
        return lifecycle

    def _response_parent_sample(
        self,
    ) -> AdaptiveMemorySnapshot | None:
        if self._memory_governor is None:
            return None
        return self._memory_governor.snapshot()

    def _admit_response_bytes_locked(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
        *,
        allocation_id: str,
        allocation_kind: str,
        allocation_reserved_before: int,
        allocation_reserve_call_count: int,
        attempt: dict[str, Any] | None,
        events: list[ResponseBufferEvent],
    ) -> AdaptiveMemoryReservationDecision | AdaptiveMemorySnapshot | None:
        request_response_before = lease._reserved_response_bytes
        request_response_projected = request_response_before + additional_bytes
        global_response_before = self._reserved_response_bytes
        global_response_projected = global_response_before + additional_bytes
        global_retained_before = self._reserved_body_bytes + global_response_before
        parent: AdaptiveMemoryReservationDecision | AdaptiveMemorySnapshot | None = (
            self._response_parent_sample()
        )

        branch: str | None = None
        if request_response_projected > self.max_response_bytes:
            branch = "per_request_response_limit"
        elif (
            lease._reserved_body_bytes + request_response_projected
            > self.max_retained_bytes_per_request
        ):
            branch = "per_request_retained_limit"
        elif (
            global_retained_before + additional_bytes
            > self.body_budget_bytes
        ):
            branch = "global_hard_budget"
        elif self._memory_governor is not None:
            parent = self._memory_governor.reserve_nowait_decision(
                "buffered_response",
                additional_bytes,
            )
            if not parent.allowed:
                branch = "parent_governor"

        if branch is None:
            return parent

        reason = (
            "upstream_response_too_large"
            if branch == "per_request_response_limit"
            else "upstream_response_budget_exhausted"
        )
        self._response_rejected[reason] += 1
        self._response_rejected_by_branch[branch] += 1
        rejected_at = self._safe_decision_monotonic()
        self._response_rejection_timestamps.append(rejected_at)
        rejection_cutoff = rejected_at - 60.0
        while (
            self._response_rejection_timestamps
            and self._response_rejection_timestamps[0] < rejection_cutoff
        ):
            self._response_rejection_timestamps.popleft()
        self._update_response_lifecycle_locked(
            lease,
            attempt,
            reserve_calls=1,
            requested_bytes=additional_bytes,
            rejections=1,
            request_response_projected=request_response_projected,
            request_response_after=request_response_before,
            global_response_projected=global_response_projected,
            global_response_after=global_response_before,
        )
        self._record_response_buffer_event_locked(
            events,
            lease,
            event="reject",
            outcome="rejected",
            admission_branch=branch,
            allocation_id=allocation_id,
            allocation_kind=allocation_kind,
            requested_bytes=additional_bytes,
            allocation_reserved_before=allocation_reserved_before,
            allocation_reserved_after=allocation_reserved_before,
            allocation_reserve_call_count=allocation_reserve_call_count,
            request_response_before=request_response_before,
            request_response_projected=request_response_projected,
            request_response_after=request_response_before,
            global_response_before=global_response_before,
            global_response_projected=global_response_projected,
            global_response_after=global_response_before,
            parent=parent,
            attempt=attempt,
            rejection_count=1,
        )
        raise UpstreamResponseBudgetExhausted(branch)

    def _record_response_buffer_event_locked(
        self,
        events: list[ResponseBufferEvent],
        lease: RequestAdmissionLease,
        **kwargs: Any,
    ) -> None:
        try:
            event = self._append_response_buffer_event_locked(
                lease,
                **kwargs,
            )
        except Exception:
            self._response_event_record_failures += 1
            return
        events.append(event)

    def _append_response_buffer_event_locked(
        self,
        lease: RequestAdmissionLease,
        *,
        event: str,
        outcome: str,
        admission_branch: str | None,
        allocation_id: str,
        allocation_kind: str,
        requested_bytes: int,
        allocation_reserved_before: int,
        allocation_reserved_after: int,
        allocation_reserve_call_count: int,
        request_response_before: int,
        request_response_projected: int,
        request_response_after: int,
        global_response_before: int,
        global_response_projected: int,
        global_response_after: int,
        parent: AdaptiveMemoryReservationDecision | AdaptiveMemorySnapshot | None,
        attempt: dict[str, Any] | None,
        reserve_started_count: int = 0,
        commit_count: int = 0,
        committed_bytes: int = 0,
        rollback_count: int = 0,
        rolled_back_bytes: int = 0,
        release_count: int = 0,
        released_bytes: int = 0,
        rejection_count: int = 0,
    ) -> ResponseBufferEvent:
        self._response_event_sequence += 1
        attempt = attempt if isinstance(attempt, dict) else {}
        attempt_index = attempt.get("routing_attempt_index")
        if not isinstance(attempt_index, int):
            attempt_index = None
        retained_from_prior = int(attempt.get("retained_before_bytes") or 0)
        crosses_retry = bool(
            retained_from_prior > 0 and (attempt_index or 0) > 1
        ) or bool(attempt.get("held_across_retry"))
        allocation_attempt_index = attempt_index or 0
        if event == "release" and allocation_attempt_index:
            crosses_retry = crosses_retry or (
                lease._response_attempts_started > allocation_attempt_index
            )

        parent_decision = (
            parent
            if isinstance(parent, AdaptiveMemoryReservationDecision)
            else None
        )
        parent_snapshot = (
            parent if isinstance(parent, AdaptiveMemorySnapshot) else None
        )
        parent_reserved_before = (
            parent_decision.reserved_before_bytes
            if parent_decision is not None
            else getattr(parent_snapshot, "reserved_bytes", None)
        )
        parent_projected = (
            parent_decision.projected_reserved_bytes
            if parent_decision is not None
            else (
                int(parent_reserved_before)
                + (0 if event == "attempt_summary" else requested_bytes)
                if parent_reserved_before is not None
                else None
            )
        )
        parent_reserved_after = (
            parent_decision.reserved_after_bytes
            if parent_decision is not None
            else parent_reserved_before
        )
        parent_available_before = (
            parent_decision.available_before_bytes
            if parent_decision is not None
            else getattr(parent_snapshot, "available_bytes", None)
        )
        parent_available_after = (
            parent_decision.available_after_bytes
            if parent_decision is not None
            else parent_available_before
        )
        request_body = lease._reserved_body_bytes
        global_body = self._reserved_body_bytes
        created = ResponseBufferEvent(
            schema_version=1,
            sequence=self._response_event_sequence,
            event=str(event),
            outcome=str(outcome),
            admission_branch=admission_branch,
            occurred_at_unix_ms=self._safe_decision_wall_time_ms(),
            request_self_lease_id=lease._lease_id,
            request_self_request_id=lease._body_observation.request_id,
            request_self_trace_id=lease._body_observation.trace_id,
            routing_attempt_id=(
                str(attempt.get("routing_attempt_id") or "")[:128] or None
            ),
            routing_attempt_index=attempt_index,
            provider=str(attempt.get("provider") or "")[:256] or None,
            request_model=(
                str(attempt.get("request_model") or "")[:256] or None
            ),
            actual_model=(
                str(attempt.get("actual_model") or "")[:256] or None
            ),
            allocation_id=str(allocation_id)[:128],
            allocation_kind=str(allocation_kind)[:64],
            requested_bytes=int(requested_bytes),
            allocation_reserved_before_bytes=int(allocation_reserved_before),
            allocation_reserved_after_bytes=int(allocation_reserved_after),
            allocation_reserve_call_count=int(allocation_reserve_call_count),
            request_response_reserved_before_bytes=int(request_response_before),
            request_response_reserved_projected_bytes=int(
                request_response_projected
            ),
            request_response_reserved_after_bytes=int(request_response_after),
            request_retained_reserved_before_bytes=(
                request_body + int(request_response_before)
            ),
            request_retained_reserved_projected_bytes=(
                request_body + int(request_response_projected)
            ),
            request_retained_reserved_after_bytes=(
                request_body + int(request_response_after)
            ),
            runtime_global_response_reserved_before_bytes=int(
                global_response_before
            ),
            runtime_global_response_reserved_projected_bytes=int(
                global_response_projected
            ),
            runtime_global_response_reserved_after_bytes=int(
                global_response_after
            ),
            runtime_global_retained_reserved_before_bytes=(
                global_body + int(global_response_before)
            ),
            runtime_global_retained_reserved_projected_bytes=(
                global_body + int(global_response_projected)
            ),
            runtime_global_retained_reserved_after_bytes=(
                global_body + int(global_response_after)
            ),
            retained_from_prior_attempts_bytes=retained_from_prior,
            crosses_retry_boundary=crosses_retry,
            request_response_limit_bytes=self.max_response_bytes,
            request_retained_limit_bytes=self.max_retained_bytes_per_request,
            runtime_global_hard_budget_bytes=self.body_budget_bytes,
            parent_governor_allowed=(
                parent_decision.allowed if parent_decision is not None else None
            ),
            parent_governor_reserved_before_bytes=parent_reserved_before,
            parent_governor_projected_reserved_bytes=parent_projected,
            parent_governor_reserved_after_bytes=parent_reserved_after,
            parent_governor_available_before_bytes=parent_available_before,
            parent_governor_available_after_bytes=parent_available_after,
            cgroup_memory_source=getattr(parent, "source", None),
            cgroup_memory_current_bytes_sampled=getattr(
                parent, "current_bytes", None
            ),
            cgroup_memory_limit_bytes_sampled=getattr(
                parent, "limit_bytes", None
            ),
            cgroup_memory_high_bytes_sampled=getattr(
                parent, "high_bytes", None
            ),
            cgroup_memory_soft_limit_bytes_sampled=getattr(
                parent, "soft_limit_bytes", None
            ),
            cgroup_memory_guard_bytes_sampled=getattr(
                parent, "guard_bytes", None
            ),
            cgroup_memory_capacity_bytes_sampled=getattr(
                parent, "capacity_bytes", None
            ),
            cgroup_memory_sample_sequence=getattr(
                parent, "sample_sequence", None
            ),
            cgroup_memory_sample_age_ms_at_decision=getattr(
                parent, "sample_age_ms", None
            ),
            cgroup_memory_sample_error=getattr(parent, "sample_error", None),
            reserve_started_count=int(reserve_started_count),
            commit_count=int(commit_count),
            committed_bytes=int(committed_bytes),
            rollback_count=int(rollback_count),
            rolled_back_bytes=int(rolled_back_bytes),
            release_count=int(release_count),
            released_bytes=int(released_bytes),
            rejection_count=int(rejection_count),
        )
        if len(self._response_event_history) == self._response_event_history.maxlen:
            self._response_event_history_overwritten += 1
        self._response_event_history.append(created)
        return created

    def _publish_response_buffer_events(
        self,
        events: list[ResponseBufferEvent],
    ) -> None:
        observer = self._response_buffer_observer
        if observer is None:
            return
        for event in events:
            try:
                accepted = observer(event)
            except Exception:
                self._response_event_observer_errors += 1
                continue
            if accepted is False:
                self._response_event_observer_enqueue_failures += 1

    def _replace_response_event_history(
        self,
        original: ResponseBufferEvent,
        updated: ResponseBufferEvent,
    ) -> None:
        for index, event in enumerate(self._response_event_history):
            if event.sequence == original.sequence:
                self._response_event_history[index] = updated
                return
        if len(self._response_event_history) == self._response_event_history.maxlen:
            self._response_event_history_overwritten += 1
        self._response_event_history.append(updated)

    def _defer_response_summary_locked(
        self,
        lease: RequestAdmissionLease,
        lifecycle_key: str,
        event: ResponseBufferEvent,
    ) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        handle = loop.call_later(
            _STREAM_RESPONSE_SUMMARY_FALLBACK_SECONDS,
            self._expire_deferred_response_summary,
            lease,
            lifecycle_key,
        )
        lease._deferred_response_summaries[lifecycle_key] = (event, handle)
        return True

    def _complete_deferred_response_summary(
        self,
        lease: RequestAdmissionLease,
        lifecycle_key: str,
        *,
        outcome: str,
    ) -> None:
        pending = lease._deferred_response_summaries.pop(
            lifecycle_key,
            None,
        )
        if pending is None:
            return
        event, handle = pending
        handle.cancel()
        finalized = replace(event, outcome=str(outcome)[:80])
        self._replace_response_event_history(event, finalized)
        lease._response_lifecycle_by_attempt.pop(lifecycle_key, None)
        self._publish_response_buffer_events([finalized])

    def _expire_deferred_response_summary(
        self,
        lease: RequestAdmissionLease,
        lifecycle_key: str,
    ) -> None:
        pending = lease._deferred_response_summaries.pop(
            lifecycle_key,
            None,
        )
        if pending is None:
            return
        event, _handle = pending
        finalized = replace(event, outcome="stream_terminal_unobserved")
        self._replace_response_event_history(event, finalized)
        lease._response_lifecycle_by_attempt.pop(lifecycle_key, None)
        self._publish_response_buffer_events([finalized])

    def recent_response_buffer_events(self) -> tuple[ResponseBufferEvent, ...]:
        return tuple(self._response_event_history)

    async def _reserve_response_additional(
        self,
        lease: RequestAdmissionLease,
        additional_bytes: int,
    ) -> None:
        events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    raise RuntimeError(
                        "cannot reserve bytes on a released request lease"
                    )
                attempt = dict(lease._response_attempt or {})
                lifecycle = self._response_lifecycle_locked(lease, attempt)
                ledger_key = f"direct:{lifecycle['lifecycle_key']}"
                allocation = lease._response_committed_allocations.get(
                    ledger_key
                )
                allocation_id = (
                    str(allocation.get("allocation_id"))
                    if isinstance(allocation, dict)
                    else self._new_response_allocation_id()
                )
                allocation_reserved_before = (
                    int(allocation.get("bytes") or 0)
                    if isinstance(allocation, dict)
                    else 0
                )
                allocation_reserve_call_count = (
                    int(allocation.get("reserve_call_count") or 0) + 1
                    if isinstance(allocation, dict)
                    else 1
                )
                self._admit_response_bytes_locked(
                    lease,
                    additional_bytes,
                    allocation_id=allocation_id,
                    allocation_kind="request_committed",
                    allocation_reserved_before=allocation_reserved_before,
                    allocation_reserve_call_count=(
                        allocation_reserve_call_count
                    ),
                    attempt=attempt,
                    events=events,
                )
                request_response_before = lease._reserved_response_bytes
                global_response_before = self._reserved_response_bytes
                lease._reserved_response_bytes += additional_bytes
                self._reserved_response_bytes += additional_bytes
                if not isinstance(allocation, dict):
                    allocation = {
                        **attempt,
                        "allocation_id": allocation_id,
                        "allocation_kind": "request_committed",
                        "bytes": 0,
                        "reserve_call_count": 0,
                    }
                    lease._response_committed_allocations[ledger_key] = allocation
                allocation["bytes"] = int(allocation.get("bytes") or 0) + (
                    additional_bytes
                )
                allocation["reserve_call_count"] = int(
                    allocation.get("reserve_call_count") or 0
                ) + 1
                self._update_response_lifecycle_locked(
                    lease,
                    attempt,
                    reserve_started=1,
                    reserve_calls=1,
                    requested_bytes=additional_bytes,
                    commits=1,
                    committed_bytes=additional_bytes,
                    request_response_projected=(
                        request_response_before + additional_bytes
                    ),
                    request_response_after=lease._reserved_response_bytes,
                    global_response_projected=(
                        global_response_before + additional_bytes
                    ),
                    global_response_after=self._reserved_response_bytes,
                )
        finally:
            self._publish_response_buffer_events(events)

    async def _activate_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
        additional_bytes: int,
    ) -> None:
        lease = reservation._lease
        events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                if lease._release_requested or lease._released:
                    raise RuntimeError(
                        "cannot reserve bytes on a released request lease"
                    )
                attempt = dict(lease._response_attempt or {})
                self._admit_response_bytes_locked(
                    lease,
                    additional_bytes,
                    allocation_id=reservation._allocation_id,
                    allocation_kind="temporary",
                    allocation_reserved_before=0,
                    allocation_reserve_call_count=1,
                    attempt=attempt,
                    events=events,
                )
                request_response_before = lease._reserved_response_bytes
                global_response_before = self._reserved_response_bytes
                lease._reserved_response_bytes += additional_bytes
                self._reserved_response_bytes += additional_bytes
                lease._memory_owner_count += 1
                reservation.size = additional_bytes
                reservation._reserve_call_count = 1
                reservation._attempt = attempt
                reservation._active = True
                self._update_response_lifecycle_locked(
                    lease,
                    attempt,
                    reserve_started=1,
                    reserve_calls=1,
                    requested_bytes=additional_bytes,
                    request_response_projected=(
                        request_response_before + additional_bytes
                    ),
                    request_response_after=lease._reserved_response_bytes,
                    global_response_projected=(
                        global_response_before + additional_bytes
                    ),
                    global_response_after=self._reserved_response_bytes,
                )
        finally:
            self._publish_response_buffer_events(events)

    async def _grow_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
        additional_bytes: int,
    ) -> None:
        lease = reservation._lease
        events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                if (
                    not reservation._active
                    or reservation._released
                    or reservation._committed
                ):
                    raise RuntimeError("temporary response reservation is closed")
                request_response_before = lease._reserved_response_bytes
                global_response_before = self._reserved_response_bytes
                self._admit_response_bytes_locked(
                    lease,
                    additional_bytes,
                    allocation_id=reservation._allocation_id,
                    allocation_kind="temporary",
                    allocation_reserved_before=reservation.size,
                    allocation_reserve_call_count=(
                        reservation._reserve_call_count + 1
                    ),
                    attempt=reservation._attempt,
                    events=events,
                )
                lease._reserved_response_bytes += additional_bytes
                self._reserved_response_bytes += additional_bytes
                reservation.size += additional_bytes
                reservation._reserve_call_count += 1
                self._update_response_lifecycle_locked(
                    lease,
                    reservation._attempt,
                    reserve_calls=1,
                    requested_bytes=additional_bytes,
                    request_response_projected=(
                        request_response_before + additional_bytes
                    ),
                    request_response_after=lease._reserved_response_bytes,
                    global_response_projected=(
                        global_response_before + additional_bytes
                    ),
                    global_response_after=self._reserved_response_bytes,
                )
                # Successful chunk growth is aggregated into the bounded
                # per-attempt summary. Rejections are always emitted at the
                # exact failing call by _admit_response_bytes_locked().
        finally:
            self._publish_response_buffer_events(events)

    async def _activate_memory_deferral(
        self,
        deferral: MemoryReleaseDeferral,
    ) -> None:
        lease = deferral._lease
        async with self._body_lock:
            if lease._release_requested or lease._memory_finalized:
                raise RuntimeError("cannot defer memory on a released request lease")
            lease._memory_owner_count += 1
            deferral._active = True

    async def _release_request(self, lease: RequestAdmissionLease) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        response_events: list[ResponseBufferEvent] = []
        ownership_cleanup_complete = False
        try:
            async with self._body_lock:
                if lease._memory_owner_count:
                    self._deferred_memory_leases.add(lease)
                else:
                    self._finalize_request_memory_locked(
                        lease,
                        decision_events=decision_events,
                        response_events=response_events,
                        release_finalizer="request_release",
                    )
            ownership_cleanup_complete = True
        finally:
            if ownership_cleanup_complete:
                try:
                    # Return business capacity before doing any exporter work.
                    await lease._active_lease.release()
                finally:
                    self._publish_decision_events(decision_events)
                    self._publish_response_buffer_events(response_events)
            else:
                self._publish_decision_events(decision_events)
                self._publish_response_buffer_events(response_events)

    async def _release_response_bytes(
        self,
        lease: RequestAdmissionLease,
        reservation: TemporaryResponseBytesReservation,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        response_events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                released_bytes = reservation.size
                if released_bytes < 0:
                    raise ValueError("released_bytes cannot be negative")
                if released_bytes > lease._reserved_response_bytes:
                    raise RuntimeError("request response reservation underflow")
                if released_bytes > self._reserved_response_bytes:
                    raise RuntimeError("upstream response reservation underflow")
                request_response_before = lease._reserved_response_bytes
                global_response_before = self._reserved_response_bytes
                lease._reserved_response_bytes -= released_bytes
                self._reserved_response_bytes -= released_bytes
                self._release_parent_memory("buffered_response", released_bytes)
                self._update_response_lifecycle_locked(
                    lease,
                    reservation._attempt,
                    rollbacks=1,
                    rolled_back_bytes=released_bytes,
                    request_response_projected=(
                        request_response_before - released_bytes
                    ),
                    request_response_after=lease._reserved_response_bytes,
                    global_response_projected=(
                        global_response_before - released_bytes
                    ),
                    global_response_after=self._reserved_response_bytes,
                )
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    response_events=response_events,
                    release_finalizer="temporary_response_release",
                )
        finally:
            self._publish_decision_events(decision_events)
            self._publish_response_buffer_events(response_events)

    async def _commit_temporary_response(
        self,
        reservation: TemporaryResponseBytesReservation,
    ) -> None:
        lease = reservation._lease
        decision_events: list[LargeBodyAdmissionDecision] = []
        response_events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                if not reservation._active or reservation._released:
                    raise RuntimeError("temporary response reservation is closed")
                if reservation._committed:
                    return
                reservation._committed = True
                allocation = {
                    **(reservation._attempt or {}),
                    "allocation_id": reservation._allocation_id,
                    "allocation_kind": "temporary_committed",
                    "bytes": reservation.size,
                    "reserve_call_count": reservation._reserve_call_count,
                }
                lease._response_committed_allocations[
                    reservation._allocation_id
                ] = allocation
                self._update_response_lifecycle_locked(
                    lease,
                    reservation._attempt,
                    commits=1,
                    committed_bytes=reservation.size,
                    request_response_projected=lease._reserved_response_bytes,
                    request_response_after=lease._reserved_response_bytes,
                    global_response_projected=self._reserved_response_bytes,
                    global_response_after=self._reserved_response_bytes,
                )
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    response_events=response_events,
                    release_finalizer="temporary_response_commit",
                )
        finally:
            self._publish_decision_events(decision_events)
            self._publish_response_buffer_events(response_events)

    async def _finish_memory_owner(
        self,
        lease: RequestAdmissionLease,
    ) -> None:
        decision_events: list[LargeBodyAdmissionDecision] = []
        response_events: list[ResponseBufferEvent] = []
        try:
            async with self._body_lock:
                self._finish_memory_owner_locked(
                    lease,
                    decision_events=decision_events,
                    response_events=response_events,
                    release_finalizer="deferred_memory_release",
                )
        finally:
            self._publish_decision_events(decision_events)
            self._publish_response_buffer_events(response_events)

    def _finish_memory_owner_locked(
        self,
        lease: RequestAdmissionLease,
        *,
        decision_events: list[LargeBodyAdmissionDecision],
        response_events: list[ResponseBufferEvent],
        release_finalizer: str,
    ) -> None:
        if lease._memory_owner_count <= 0:
            raise RuntimeError("request memory owner underflow")
        lease._memory_owner_count -= 1
        if lease._release_requested and lease._memory_owner_count == 0:
            self._finalize_request_memory_locked(
                lease,
                decision_events=decision_events,
                response_events=response_events,
                release_finalizer=release_finalizer,
            )

    def _finalize_request_memory_locked(
        self,
        lease: RequestAdmissionLease,
        *,
        decision_events: list[LargeBodyAdmissionDecision],
        response_events: list[ResponseBufferEvent],
        release_finalizer: str,
    ) -> None:
        if lease._memory_finalized:
            return
        reserved_body_bytes = lease._reserved_body_bytes
        reserved_response_bytes = lease._reserved_response_bytes
        global_body_before = self._reserved_body_bytes
        global_response_before = self._reserved_response_bytes
        if reserved_body_bytes > self._reserved_body_bytes:
            raise RuntimeError("request body reservation underflow")
        if reserved_response_bytes > self._reserved_response_bytes:
            raise RuntimeError("upstream response reservation underflow")
        self._reserved_body_bytes -= reserved_body_bytes
        self._release_parent_memory("request_body", reserved_body_bytes)
        lease._reserved_body_bytes = 0
        allocation_total = sum(
            int(allocation.get("bytes") or 0)
            for allocation in lease._response_committed_allocations.values()
        )
        if allocation_total != reserved_response_bytes:
            missing = reserved_response_bytes - allocation_total
            if missing < 0:
                raise RuntimeError("response allocation ledger overflow")
            if missing:
                synthetic_id = self._new_response_allocation_id()
                lease._response_committed_allocations[synthetic_id] = {
                    **(lease._response_attempt or {}),
                    "allocation_id": synthetic_id,
                    "allocation_kind": "legacy_unattributed",
                    "bytes": missing,
                    "reserve_call_count": 1,
                }
        for allocation in tuple(
            lease._response_committed_allocations.values()
        ):
            released = int(allocation.get("bytes") or 0)
            if released < 0 or released > lease._reserved_response_bytes:
                raise RuntimeError("response allocation ledger underflow")
            request_response_before = lease._reserved_response_bytes
            current_global_response = self._reserved_response_bytes
            lease._reserved_response_bytes -= released
            self._reserved_response_bytes -= released
            self._release_parent_memory("buffered_response", released)
            self._update_response_lifecycle_locked(
                lease,
                allocation,
                releases=1,
                released_bytes=released,
                request_response_projected=(
                    request_response_before - released
                ),
                request_response_after=lease._reserved_response_bytes,
                global_response_projected=(
                    current_global_response - released
                ),
                global_response_after=self._reserved_response_bytes,
            )
        if lease._reserved_response_bytes:
            raise RuntimeError("response allocation ledger did not fully release")
        lease._response_committed_allocations.clear()
        parent_after = (
            self._memory_governor.snapshot_cached()
            if self._memory_governor is not None
            else None
        )
        completed_lifecycle_keys: list[str] = []
        for lifecycle_key, lifecycle in tuple(
            lease._response_lifecycle_by_attempt.items()
        ):
            if lifecycle_key == "unattributed":
                # Temporary allocations used by request JSON/payload parsing
                # happen before a provider routing attempt exists. Rejections
                # are emitted immediately with their exact admission branch;
                # successful preprocessing churn is not a response-attempt
                # lifecycle event and would otherwise double event volume.
                completed_lifecycle_keys.append(lifecycle_key)
                continue
            committed_bytes = int(lifecycle.get("committed_bytes") or 0)
            released_bytes = int(lifecycle.get("released_bytes") or 0)
            summary_events: list[ResponseBufferEvent] = []
            self._record_response_buffer_event_locked(
                summary_events,
                lease,
                event="attempt_summary",
                outcome=str(lifecycle.get("outcome") or "finished")[:80],
                admission_branch=None,
                allocation_id=(
                    f"{lease._lease_id}:"
                    f"{lifecycle.get('lifecycle_key') or 'unattributed'}"
                )[:128],
                allocation_kind="attempt_aggregate",
                requested_bytes=int(lifecycle.get("requested_bytes") or 0),
                allocation_reserved_before=0,
                allocation_reserved_after=max(
                    0,
                    committed_bytes - released_bytes,
                ),
                allocation_reserve_call_count=int(
                    lifecycle.get("reserve_call_count") or 0
                ),
                request_response_before=int(
                    lifecycle.get("request_response_before") or 0
                ),
                request_response_projected=int(
                    lifecycle.get("request_response_projected") or 0
                ),
                request_response_after=0,
                global_response_before=int(
                    lifecycle.get("global_response_before") or 0
                ),
                global_response_projected=int(
                    lifecycle.get("global_response_projected") or 0
                ),
                global_response_after=self._reserved_response_bytes,
                parent=parent_after,
                attempt=lifecycle,
                reserve_started_count=int(
                    lifecycle.get("reserve_started_count") or 0
                ),
                commit_count=int(lifecycle.get("commit_count") or 0),
                committed_bytes=committed_bytes,
                rollback_count=int(lifecycle.get("rollback_count") or 0),
                rolled_back_bytes=int(
                    lifecycle.get("rolled_back_bytes") or 0
                ),
                release_count=int(lifecycle.get("release_count") or 0),
                released_bytes=released_bytes,
                rejection_count=int(lifecycle.get("rejection_count") or 0),
            )
            summary_event = summary_events[0]
            active_attempt = lease._response_attempt
            is_active_stream_pending = bool(
                summary_event.outcome == "stream_pending"
                and isinstance(active_attempt, dict)
                and str(active_attempt.get("lifecycle_key") or "")
                == lifecycle_key
            )
            if is_active_stream_pending and self._defer_response_summary_locked(
                lease,
                lifecycle_key,
                summary_event,
            ):
                continue
            if is_active_stream_pending:
                fallback_event = replace(
                    summary_event,
                    outcome="stream_terminal_unobserved",
                )
                self._replace_response_event_history(
                    summary_event,
                    fallback_event,
                )
                response_events.append(fallback_event)
            else:
                response_events.append(summary_event)
            completed_lifecycle_keys.append(lifecycle_key)
        for lifecycle_key in completed_lifecycle_keys:
            lease._response_lifecycle_by_attempt.pop(lifecycle_key, None)
        self._release_large_body_slot_locked(
            lease,
            request_before=reserved_body_bytes,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            release_reason=lease._release_reason,
            release_finalizer=release_finalizer,
            decision_events=decision_events,
        )
        lease._memory_finalized = True
        self._deferred_memory_leases.discard(lease)

    def _claim_large_body_slot_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        if len(self._large_body_holders) >= self.large_body_limit:
            raise RuntimeError("large request capacity changed after availability check")
        active_before = len(self._large_body_holders)
        claim_id = self._new_claim_id_locked()
        holder = _LargeBodyHolder(
            claim_id=claim_id,
            lease_id=lease._lease_id,
            observation=lease._body_observation,
            claimed_at_monotonic=self._safe_decision_monotonic(),
            claimed_at_unix_ms=self._safe_decision_wall_time_ms(),
            reserved_weighted_bytes=next_request_bytes,
        )
        self._large_body_holders[claim_id] = holder
        lease._large_body_slot = True
        lease._large_body_claim_id = claim_id
        self._append_large_body_decision_locked(
            decision_events,
            decision="claim",
            reason="threshold_crossed",
            owner=lease,
            request_before=request_before,
            attempted_after=next_request_bytes,
            committed_after=next_request_bytes,
            active_before=active_before,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            holder=holder,
        )

    def _ensure_large_body_slot_available_locked(
        self,
        lease: RequestAdmissionLease | PendingBodyReservation,
        next_request_bytes: int,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if (
            not self.large_body_limit
            or lease._large_body_slot
            or next_request_bytes <= self.large_body_threshold_weighted_bytes
        ):
            return
        if len(self._large_body_holders) >= self.large_body_limit:
            self._body_rejected["large_body_capacity_exhausted"] += 1
            self._append_large_body_decision_locked(
                decision_events,
                decision="reject",
                reason="large_body_capacity_exhausted",
                owner=lease,
                request_before=request_before,
                attempted_after=next_request_bytes,
                committed_after=request_before,
                active_before=len(self._large_body_holders),
                global_body_before=global_body_before,
                global_response_before=global_response_before,
                blocking_holders=tuple(self._large_body_holders.values()),
            )
            raise LargeBodyCapacityExhausted()

    def _update_large_body_holder_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        reserved_weighted_bytes: int,
    ) -> None:
        if not owner._large_body_slot:
            return
        claim_id = owner._large_body_claim_id
        if not claim_id or claim_id not in self._large_body_holders:
            raise RuntimeError("large request holder missing")
        holder = self._large_body_holders[claim_id]
        self._large_body_holders[claim_id] = replace(
            holder,
            lease_id=owner._lease_id,
            observation=owner._body_observation,
            reserved_weighted_bytes=reserved_weighted_bytes,
        )

    def _release_large_body_slot_locked(
        self,
        owner: RequestAdmissionLease | PendingBodyReservation,
        *,
        request_before: int,
        global_body_before: int,
        global_response_before: int,
        release_reason: str,
        release_finalizer: str,
        decision_events: list[LargeBodyAdmissionDecision],
    ) -> None:
        if not owner._large_body_slot:
            return
        claim_id = owner._large_body_claim_id
        if not claim_id:
            raise RuntimeError("large request claim id missing")
        holder = self._large_body_holders.get(claim_id)
        if holder is None:
            raise RuntimeError("large request holder missing during release")
        active_before = len(self._large_body_holders)
        del self._large_body_holders[claim_id]
        owner._large_body_slot = False
        owner._large_body_claim_id = None
        self._append_large_body_decision_locked(
            decision_events,
            decision="release",
            reason="slot_released",
            owner=owner,
            request_before=request_before,
            attempted_after=request_before,
            committed_after=0,
            active_before=active_before,
            global_body_before=global_body_before,
            global_response_before=global_response_before,
            release_reason=release_reason,
            release_finalizer=release_finalizer,
            holder=holder,
        )

    def _safe_decision_monotonic(self) -> float:
        try:
            return float(self._clock())
        except Exception:
            self._decision_record_failures += 1
            return monotonic()

    def _safe_decision_wall_time_ms(self) -> int:
        try:
            return int(float(self._wall_clock()) * 1000.0)
        except Exception:
            self._decision_record_failures += 1
            return int(time() * 1000.0)

    def _append_large_body_decision_locked(
        self,
        decision_events: list[LargeBodyAdmissionDecision],
        **kwargs: Any,
    ) -> None:
        """Record observability without changing admission ownership semantics."""

        try:
            event = self._record_large_body_decision_locked(**kwargs)
        except Exception:
            self._decision_record_failures += 1
            return
        decision_events.append(event)

    def _holder_snapshot_locked(
        self,
        holder: _LargeBodyHolder,
        *,
        now_monotonic: float,
    ) -> LargeBodyHolderSnapshot:
        return LargeBodyHolderSnapshot(
            claim_id=holder.claim_id,
            lease_id=holder.lease_id,
            request_id=holder.observation.request_id,
            trace_id=holder.observation.trace_id,
            claimed_at_unix_ms=holder.claimed_at_unix_ms,
            held_ms=max(
                0,
                int(round((now_monotonic - holder.claimed_at_monotonic) * 1000.0)),
            ),
            request_self_body_reserved_weighted_bytes=(
                holder.reserved_weighted_bytes
            ),
        )

    def _record_large_body_decision_locked(
        self,
        *,
        decision: str,
        reason: str,
        owner: RequestAdmissionLease | PendingBodyReservation,
        request_before: int,
        attempted_after: int,
        committed_after: int,
        active_before: int,
        global_body_before: int,
        global_response_before: int,
        release_reason: str | None = None,
        release_finalizer: str | None = None,
        holder: _LargeBodyHolder | None = None,
        blocking_holders: tuple[_LargeBodyHolder, ...] = (),
    ) -> LargeBodyAdmissionDecision:
        now_monotonic = self._safe_decision_monotonic()
        try:
            parent = (
                self._memory_governor.snapshot_cached()
                if self._memory_governor is not None
                else None
            )
        except Exception:
            self._decision_record_failures += 1
            parent = None
        effective_body_budget = self.body_budget_bytes
        if parent is not None:
            effective_body_budget = min(
                effective_body_budget,
                parent.capacity_bytes,
            )
        observation = owner._body_observation
        self._decision_sequence += 1
        event = LargeBodyAdmissionDecision(
            schema_version=1,
            sequence=self._decision_sequence,
            decision=decision,
            reason=reason,
            occurred_at_unix_ms=self._safe_decision_wall_time_ms(),
            release_reason=release_reason,
            release_finalizer=release_finalizer,
            request_self_lease_id=owner._lease_id,
            request_self_request_id=observation.request_id,
            request_self_trace_id=observation.trace_id,
            request_self_method=observation.method,
            request_self_path=observation.path,
            request_self_declared_content_length_bytes=(
                observation.declared_content_length_bytes
            ),
            request_self_wire_bytes=observation.wire_bytes,
            request_self_decoded_bytes=observation.decoded_bytes,
            request_self_decoder_workspace_bytes=(
                observation.decoder_workspace_bytes
            ),
            request_self_json_raw_bytes=observation.json_raw_bytes,
            request_self_json_structural_item_count=(
                observation.json_structural_item_count
            ),
            request_self_json_depth=observation.json_depth,
            request_self_json_peak_depth=observation.json_peak_depth,
            request_self_json_scalar_bytes=observation.json_scalar_bytes,
            request_self_json_estimated_bytes=observation.json_estimated_bytes,
            request_self_json_raw_memory_multiplier=(
                observation.json_raw_memory_multiplier
            ),
            request_self_json_structural_item_memory_bytes=(
                observation.json_structural_item_memory_bytes
            ),
            request_self_body_reserved_weighted_before_bytes=request_before,
            request_self_body_reserved_weighted_attempted_after_bytes=(
                attempted_after
            ),
            request_self_body_reserved_weighted_committed_after_bytes=(
                committed_after
            ),
            runtime_global_large_body_threshold_weighted_bytes=(
                self.large_body_threshold_weighted_bytes
            ),
            runtime_global_large_body_active_before=active_before,
            runtime_global_large_body_active_after=len(self._large_body_holders),
            runtime_global_large_body_limit=self.large_body_limit,
            runtime_global_request_body_reserved_weighted_before_bytes=(
                global_body_before
            ),
            runtime_global_request_body_reserved_weighted_after_bytes=(
                self._reserved_body_bytes
            ),
            runtime_global_upstream_response_reserved_weighted_before_bytes=(
                global_response_before
            ),
            runtime_global_upstream_response_reserved_weighted_after_bytes=(
                self._reserved_response_bytes
            ),
            runtime_global_retained_reserved_weighted_before_bytes=(
                global_body_before + global_response_before
            ),
            runtime_global_retained_reserved_weighted_after_bytes=(
                self._reserved_body_bytes + self._reserved_response_bytes
            ),
            runtime_global_request_body_budget_weighted_bytes=effective_body_budget,
            runtime_global_request_body_budget_hard_weighted_bytes=self.body_budget_bytes,
            runtime_global_cgroup_memory_source=getattr(parent, "source", None),
            runtime_global_cgroup_memory_current_bytes_sampled=getattr(
                parent, "current_bytes", None
            ),
            runtime_global_cgroup_memory_limit_bytes_sampled=getattr(parent, "limit_bytes", None),
            runtime_global_cgroup_memory_high_bytes_sampled=getattr(parent, "high_bytes", None),
            runtime_global_cgroup_memory_soft_limit_bytes_sampled=getattr(
                parent, "soft_limit_bytes", None
            ),
            runtime_global_cgroup_memory_guard_bytes_sampled=getattr(parent, "guard_bytes", None),
            runtime_global_cgroup_memory_capacity_bytes_sampled=getattr(
                parent, "capacity_bytes", None
            ),
            runtime_global_cgroup_memory_available_bytes_sampled=getattr(
                parent, "available_bytes", None
            ),
            runtime_global_cgroup_memory_reserved_bytes_sampled=getattr(
                parent, "reserved_bytes", None
            ),
            runtime_global_cgroup_memory_sample_sequence=getattr(
                parent, "sample_sequence", None
            ),
            runtime_global_cgroup_memory_sample_age_ms_at_decision=getattr(
                parent, "sample_age_ms", None
            ),
            runtime_global_cgroup_memory_sample_error=getattr(
                parent, "sample_error", None
            ),
            holder=(
                self._holder_snapshot_locked(holder, now_monotonic=now_monotonic)
                if holder is not None
                else None
            ),
            blocking_holders=tuple(
                self._holder_snapshot_locked(
                    blocking_holder,
                    now_monotonic=now_monotonic,
                )
                for blocking_holder in blocking_holders
            ),
        )
        if len(self._decision_history) == self._decision_history.maxlen:
            self._decision_history_overwritten += 1
        self._decision_history.append(event)
        return event

    def _publish_decision_events(
        self,
        events: list[LargeBodyAdmissionDecision],
    ) -> None:
        observer = self._decision_observer
        if observer is None:
            return
        for event in events:
            try:
                accepted = observer(event)
            except Exception:
                self._decision_observer_errors += 1
                continue
            if accepted is False:
                self._decision_observer_enqueue_failures += 1

    def recent_large_body_decisions(self) -> tuple[LargeBodyAdmissionDecision, ...]:
        return tuple(self._decision_history)

    def record_rejection(self, reason: str) -> None:
        normalized = str(reason or "").strip()
        if normalized:
            self._body_rejected[normalized] += 1

    def _reserve_parent_memory(self, category: str, size: int) -> bool:
        if self._memory_governor is None:
            return True
        return self._memory_governor.reserve_nowait(category, size)

    def _release_parent_memory(self, category: str, size: int) -> None:
        if self._memory_governor is not None and size:
            self._memory_governor.release(category, size)
        if size:
            self._body_memory_wait_queue.notify_memory_released()

    def snapshot(self) -> dict[str, Any]:
        gate_snapshot = self._active_gate.snapshot()
        rejected = Counter(gate_snapshot["rejected"])
        rejected.update(self._body_rejected)
        rejected.update(self._response_rejected)
        parent_snapshot = (
            self._memory_governor.snapshot()
            if self._memory_governor is not None
            else None
        )
        effective_body_budget = self.body_budget_bytes
        if parent_snapshot is not None:
            effective_body_budget = min(
                effective_body_budget,
                parent_snapshot.capacity_bytes,
            )
        now_monotonic = self._clock()
        rejection_cutoff = now_monotonic - 60.0
        while (
            self._response_rejection_timestamps
            and self._response_rejection_timestamps[0] < rejection_cutoff
        ):
            self._response_rejection_timestamps.popleft()
        holder_ages_ms = [
            max(
                0,
                int(round((now_monotonic - holder.claimed_at_monotonic) * 1000.0)),
            )
            for holder in self._large_body_holders.values()
        ]
        rejection_decision_total = sum(int(value or 0) for value in rejected.values())
        retained_reserved = self._reserved_body_bytes + self._reserved_response_bytes
        soft_remaining = max(0, effective_body_budget - retained_reserved)
        soft_remaining_ratio = (
            soft_remaining / effective_body_budget
            if effective_body_budget > 0
            else 0.0
        )
        response_rejections_1m = len(self._response_rejection_timestamps)
        return {
            **gate_snapshot,
            **self._body_memory_wait_queue.snapshot(),
            "reserved_body_bytes": self._reserved_body_bytes,
            "staged_body_release_count": self._staged_body_release_count,
            "staged_body_released_bytes": self._staged_body_released_bytes,
            "pending_body_reserved_bytes": self._pending_body_reserved_bytes,
            "reserved_response_bytes": self._reserved_response_bytes,
            "reserved_retained_bytes": (
                self._reserved_body_bytes + self._reserved_response_bytes
            ),
            "deferred_memory_requests": len(self._deferred_memory_leases),
            "deferred_memory_bytes": sum(
                lease._reserved_body_bytes + lease._reserved_response_bytes
                for lease in self._deferred_memory_leases
            ),
            "body_budget": effective_body_budget,
            "body_budget_hard": self.body_budget_bytes,
            "max_body_bytes": self.max_body_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_retained_bytes_per_request": self.max_retained_bytes_per_request,
            "body_memory_wait_timeout_seconds": (
                self.body_memory_wait_timeout_seconds
            ),
            "body_memory_waiter_limit": self.body_memory_waiter_limit,
            "small_body_lane_threshold_bytes": (
                self.small_body_lane_threshold_bytes
            ),
            "small_body_lane_reserve_bytes": self.small_body_lane_reserve_bytes,
            "large_body_threshold_weighted_bytes": (
                self.large_body_threshold_weighted_bytes
            ),
            "large_body_limit": self.large_body_limit,
            "large_body_active": len(self._large_body_holders),
            "large_body_oldest_holder_age_ms": (
                max(holder_ages_ms) if holder_ages_ms else 0
            ),
            "large_body_decision_events_recorded_total": self._decision_sequence,
            "large_body_decision_history_overwritten_total": (
                self._decision_history_overwritten
            ),
            "large_body_decision_record_failures_total": (
                self._decision_record_failures
            ),
            "large_body_decision_observer_errors_total": (
                self._decision_observer_errors
            ),
            "large_body_decision_observer_enqueue_failures_total": (
                self._decision_observer_enqueue_failures
            ),
            "memory_parent": parent_snapshot,
            "rejected": dict(rejected),
            "rejection_decisions": dict(rejected),
            "rejection_decision_total": rejection_decision_total,
            "response_rejection_decisions_by_branch": dict(
                self._response_rejected_by_branch
            ),
            "response_rejection_decision_total": sum(
                self._response_rejected_by_branch.values()
            ),
            "response_rejections_1m": response_rejections_1m,
            "response_rejection_rate_per_second_1m": (
                response_rejections_1m / 60.0
            ),
            "response_budget_soft_remaining_bytes": soft_remaining,
            "response_budget_soft_remaining_ratio": soft_remaining_ratio,
            "response_budget_soft_headroom_alert": bool(
                effective_body_budget > 0 and soft_remaining_ratio <= 0.10
            ),
            "response_rejection_rate_alert": response_rejections_1m >= 5,
            "response_buffer_events_recorded_total": (
                self._response_event_sequence
            ),
            "response_buffer_event_history_overwritten_total": (
                self._response_event_history_overwritten
            ),
            "response_buffer_event_record_failures_total": (
                self._response_event_record_failures
            ),
            "response_buffer_event_observer_errors_total": (
                self._response_event_observer_errors
            ),
            "response_buffer_event_observer_enqueue_failures_total": (
                self._response_event_observer_enqueue_failures
            ),
        }


_current_request_admission_lease: contextvars.ContextVar[
    RequestAdmissionLease | None
] = contextvars.ContextVar("uni_api_request_admission_lease", default=None)


def bind_request_admission_lease(
    lease: RequestAdmissionLease | None,
) -> contextvars.Token[RequestAdmissionLease | None]:
    return _current_request_admission_lease.set(lease)


def reset_request_admission_lease(
    token: contextvars.Token[RequestAdmissionLease | None],
) -> None:
    _current_request_admission_lease.reset(token)


def get_request_admission_lease() -> RequestAdmissionLease | None:
    return _current_request_admission_lease.get()
