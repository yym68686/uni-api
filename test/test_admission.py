import asyncio

import pytest

import uni_api.admission.core as admission_core
from uni_api.admission import (
    AdmissionRejected,
    BoundedAdmissionGate,
    RequestAdmissionController,
    RequestBodyBudgetExhausted,
    RequestBodyTooLarge,
    RequestBodyObservation,
    UpstreamResponseBudgetExhausted,
)
from uni_api.admission.memory import AdaptiveMemoryGovernor, ProcessMemorySample


async def _wait_until(predicate, *, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def test_gate_immediate_acquire_and_idempotent_release():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )

        lease = await gate.acquire()
        assert lease.wait_ms >= 0
        snapshot = gate.snapshot()
        assert {
            key: snapshot[key]
            for key in (
                "active",
                "waiters",
                "capacity",
                "waiter_limit",
                "acquired_total",
                "cancelled_total",
                "rejected",
            )
        } == {
            "active": 1,
            "waiters": 0,
            "capacity": 1,
            "waiter_limit": 1,
            "acquired_total": 1,
            "cancelled_total": 0,
            "rejected": {},
        }

        await asyncio.gather(lease.release(), lease.release(), lease.release())
        assert lease.released is True
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_grants_waiters_in_fifo_order():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=2,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        acquired_order = []
        release_first = asyncio.Event()

        async def wait_for_lease(name, release_event=None):
            lease = await gate.acquire()
            acquired_order.append(name)
            if release_event is not None:
                await release_event.wait()
            await lease.release()

        first = asyncio.create_task(wait_for_lease("first", release_first))
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)
        second = asyncio.create_task(wait_for_lease("second"))
        await _wait_until(lambda: gate.snapshot()["waiters"] == 2)

        await holder.release()
        await _wait_until(lambda: acquired_order == ["first"])
        assert gate.snapshot()["waiters"] == 1

        release_first.set()
        await asyncio.gather(first, second)
        assert acquired_order == ["first", "second"]
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_cancellation_before_waiter_task_first_step_returns_queue_position(
    monkeypatch,
):
    """The outer begin_acquire coroutine owns a not-yet-started child task."""

    class CancelBeforeChildStarts:
        def set(self):
            return None

        async def wait(self):
            raise asyncio.CancelledError

    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()

        with monkeypatch.context() as patch:
            # core.py and this test refer to the same asyncio module object.
            # Raising before Event.wait suspends guarantees the child task has
            # not taken its first coroutine step.
            patch.setattr(asyncio, "Event", CancelBeforeChildStarts)
            with pytest.raises(asyncio.CancelledError):
                await gate.begin_acquire()

        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["cancelled_total"] == 1
        await holder.release()
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["acquired_total"] == 1

    asyncio.run(run())


def test_gate_acquire_cancellation_same_turn_as_grant_releases_result(monkeypatch):
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        original_begin_acquire = gate.begin_acquire

        async def begin_and_cancel_owner_on_grant(*, timeout_seconds=None):
            acquisition = await original_begin_acquire(
                timeout_seconds=timeout_seconds
            )
            owner = asyncio.current_task()
            assert owner is not None
            acquisition.add_done_callback(lambda _done: owner.cancel())
            return acquisition

        monkeypatch.setattr(gate, "begin_acquire", begin_and_cancel_owner_on_grant)
        owner = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)
        await holder.release()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["waiters"] == 0

    asyncio.run(run())


def test_request_acquire_cancellation_same_turn_as_grant_releases_result(
    monkeypatch,
):
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        holder = await controller.acquire()
        original_begin_acquire = controller.begin_acquire

        async def begin_and_cancel_owner_on_grant(*, timeout_seconds=None):
            acquisition = await original_begin_acquire(
                timeout_seconds=timeout_seconds
            )
            owner = asyncio.current_task()
            assert owner is not None
            acquisition.add_done_callback(lambda _done: owner.cancel())
            return acquisition

        monkeypatch.setattr(
            controller,
            "begin_acquire",
            begin_and_cancel_owner_on_grant,
        )
        owner = asyncio.create_task(controller.acquire())
        await _wait_until(lambda: controller.snapshot()["waiters"] == 1)
        await holder.release()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["waiters"] == 0

    asyncio.run(run())


def test_gate_rejects_when_waiter_queue_is_full():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

        with pytest.raises(AdmissionRejected) as exc_info:
            await gate.acquire()
        assert exc_info.value.reason == "queue_full"
        assert exc_info.value.status_code == 503
        assert gate.snapshot()["rejected"] == {"queue_full": 1}

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await holder.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_times_out_without_leaking_a_waiter_or_slot():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=0.02,
        )
        holder = await gate.acquire()

        with pytest.raises(AdmissionRejected) as exc_info:
            await gate.acquire()
        assert exc_info.value.reason == "wait_timeout"
        assert gate.snapshot()["active"] == 1
        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["rejected"] == {"wait_timeout": 1}

        await holder.release()
        followup = await gate.acquire()
        await followup.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_cancellation_while_queued_does_not_leak_capacity():
    async def run():
        gate = BoundedAdmissionGate(
            1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        holder = await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert gate.snapshot()["waiters"] == 0
        assert gate.snapshot()["cancelled_total"] == 1

        await holder.release()
        followup = await gate.acquire()
        await followup.release()
        assert gate.snapshot()["active"] == 0

    asyncio.run(run())


def test_gate_grant_cancellation_race_transfers_capacity_or_returns_a_lease():
    async def run():
        # Repeat the handoff race. If cancellation arrives before acquire()
        # returns, the gate must reclaim the grant. If it arrives afterwards,
        # the returned lease remains explicit ownership and is released here.
        for _ in range(50):
            gate = BoundedAdmissionGate(
                1,
                waiter_limit=1,
                wait_timeout_seconds=1,
            )
            holder = await gate.acquire()
            queued = asyncio.create_task(gate.acquire())
            await _wait_until(lambda: gate.snapshot()["waiters"] == 1)

            release = asyncio.create_task(holder.release())
            queued.cancel()
            await release
            try:
                delivered_lease = await queued
            except asyncio.CancelledError:
                delivered_lease = None
            if delivered_lease is not None:
                await delivered_lease.release()

            await _wait_until(lambda: gate.snapshot()["active"] == 0)
            followup = await gate.acquire()
            await followup.release()

    asyncio.run(run())


def test_production_admission_envelope_is_exactly_64_active_plus_936_waiters():
    async def run():
        gate = BoundedAdmissionGate(
            64,
            waiter_limit=936,
            wait_timeout_seconds=2,
        )
        holders = await asyncio.gather(*(gate.acquire() for _ in range(64)))
        waiters = [asyncio.create_task(gate.acquire()) for _ in range(936)]
        await _wait_until(lambda: gate.snapshot()["waiters"] == 936)

        snapshot = gate.snapshot()
        assert snapshot["active"] == 64
        assert snapshot["waiters"] == 936
        with pytest.raises(AdmissionRejected) as rejected:
            await gate.acquire()
        assert rejected.value.reason == "queue_full"

        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        await asyncio.gather(*(holder.release() for holder in holders))
        snapshot = gate.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["waiters"] == 0
        assert snapshot["cancelled_total"] == 936

    asyncio.run(run())


def test_request_controller_enforces_per_request_body_limit():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=20,
        )
        lease = await controller.acquire(initial_body_bytes=4)
        assert await lease.reserve_body_bytes(2) == 6

        with pytest.raises(RequestBodyTooLarge) as exc_info:
            await lease.reserve_body_bytes(5)
        assert exc_info.value.status_code == 413
        assert exc_info.value.reason == "body_too_large"
        assert lease.reserved_body_bytes == 6
        assert controller.snapshot()["reserved_body_bytes"] == 6

        await lease.release()
        assert lease.reserved_body_bytes == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0
        assert controller.snapshot()["rejected"] == {"body_too_large": 1}

    asyncio.run(run())


def test_resource_only_body_admission_uses_global_budget_not_request_limit():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=4,
            body_budget_bytes=32,
        )
        lease = await controller.acquire(
            resource_only_body_admission=True,
        )

        assert await lease.reserve_body_bytes(16) == 16
        assert controller.snapshot()["reserved_body_bytes"] == 16
        assert controller.snapshot()["rejected"] == {}

        await lease.release()
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_request_controller_enforces_global_body_budget_and_recovers_on_release():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
        )
        first = await controller.acquire(initial_body_bytes=6)
        second = await controller.acquire(initial_body_bytes=4)

        with pytest.raises(RequestBodyBudgetExhausted) as exc_info:
            await second.reserve_body_bytes(1)
        assert exc_info.value.status_code == 503
        assert exc_info.value.reason == "body_budget_exhausted"
        assert controller.snapshot()["reserved_body_bytes"] == 10

        await first.release()
        assert await second.reserve_body_bytes(1) == 5
        assert controller.snapshot()["reserved_body_bytes"] == 5

        await asyncio.gather(second.release(), second.release())
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["body_budget"] == 10
        assert snapshot["rejected"] == {"body_budget_exhausted": 1}

    asyncio.run(run())


def test_body_budget_exhaustion_waits_for_release_when_enabled():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
            body_memory_wait_timeout_seconds=0.5,
            body_memory_waiter_limit=2,
        )
        first = await controller.acquire(initial_body_bytes=6)
        second = await controller.acquire(initial_body_bytes=4)
        reservation = asyncio.create_task(second.reserve_body_bytes(1))
        async with asyncio.timeout(1):
            while controller.snapshot()["body_memory_waiters"] != 1:
                await asyncio.sleep(0)

        await first.release()
        assert await reservation == 5
        snapshot = controller.snapshot()
        assert snapshot["body_memory_wait_timeouts"] == 0
        assert snapshot["rejected"] == {}
        await second.release()

    asyncio.run(run())


def test_existing_body_owner_can_finish_while_new_request_waits():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
            body_memory_wait_timeout_seconds=0.5,
            body_memory_waiter_limit=2,
        )
        existing = await controller.acquire(initial_body_bytes=8)
        newcomer = await controller.acquire()
        waiting = asyncio.create_task(newcomer.reserve_body_bytes(8))
        async with asyncio.timeout(1):
            while controller.snapshot()["body_memory_waiters"] != 1:
                await asyncio.sleep(0)

        # Incremental decoding for the existing body must not queue behind a
        # new request that cannot proceed until this owner finishes/releases.
        assert await existing.reserve_body_bytes(2) == 10
        await existing.release()
        assert await waiting == 8
        await newcomer.release()

    asyncio.run(run())


def test_parent_memory_wait_does_not_hold_controller_body_lock():
    async def run():
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(100, 1000, source="fake"),
            guard_bytes=0,
            guard_ratio=0,
            sample_cache_seconds=60,
        )
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=900,
            body_budget_bytes=1000,
            body_memory_wait_timeout_seconds=0.5,
            body_memory_waiter_limit=2,
            memory_governor=governor,
        )
        owner = await controller.acquire(initial_body_bytes=900)
        waiter = await controller.acquire()
        blocked = asyncio.create_task(waiter.reserve_body_bytes(1))
        async with asyncio.timeout(1):
            while governor.snapshot().waiting_reservations != 1:
                await asyncio.sleep(0)

        # This release needs _body_lock. If reserve() awaited the parent while
        # holding that lock, the pair would deadlock until timeout.
        await owner.release()
        assert await blocked == 1
        await waiter.release()

    asyncio.run(run())


def test_cancelled_parent_memory_wait_releases_fair_turn():
    async def run():
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(100, 1000, source="fake"),
            guard_bytes=0,
            guard_ratio=0,
            sample_cache_seconds=60,
        )
        controller = RequestAdmissionController(
            capacity=3,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=900,
            body_budget_bytes=1000,
            body_memory_wait_timeout_seconds=1,
            body_memory_waiter_limit=2,
            memory_governor=governor,
        )
        owner = await controller.acquire(initial_body_bytes=900)
        cancelled_lease = await controller.acquire()
        cancelled = asyncio.create_task(cancelled_lease.reserve_body_bytes(1))
        async with asyncio.timeout(1):
            while governor.snapshot().waiting_reservations != 1:
                await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert controller.snapshot()["body_memory_small_turn_active"] == 0

        await owner.release()
        next_lease = await controller.acquire(initial_body_bytes=1)
        await asyncio.gather(cancelled_lease.release(), next_lease.release())

    asyncio.run(run())


def test_small_lane_and_tenant_round_robin_prevent_large_request_monopoly():
    async def run():
        controller = RequestAdmissionController(
            capacity=6,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=100,
            body_budget_bytes=100,
            body_memory_wait_timeout_seconds=0.5,
            body_memory_waiter_limit=4,
            small_body_lane_threshold_bytes=10,
            small_body_lane_reserve_bytes=20,
        )
        large = await controller.acquire(initial_body_bytes=80)
        small_one = await controller.acquire(initial_body_bytes=10)
        small_two = await controller.acquire(initial_body_bytes=10)
        assert controller.snapshot()["reserved_body_bytes"] == 100

        order = []
        first_a = await controller.acquire()
        second_a = await controller.acquire()
        first_b = await controller.acquire()
        first_a.set_fairness_key("tenant-a")
        second_a.set_fairness_key("tenant-a")
        first_b.set_fairness_key("tenant-b")

        async def reserve(lease, name):
            await lease.reserve_body_bytes(5)
            order.append(name)

        tasks = [
            asyncio.create_task(reserve(first_a, "a1")),
            asyncio.create_task(reserve(second_a, "a2")),
            asyncio.create_task(reserve(first_b, "b1")),
        ]
        async with asyncio.timeout(1):
            while controller.snapshot()["body_memory_waiters"] != 3:
                await asyncio.sleep(0)
        await asyncio.gather(small_one.release(), small_two.release())
        await asyncio.gather(*tasks)
        assert order == ["a1", "b1", "a2"]

        await asyncio.gather(
            large.release(),
            first_a.release(),
            second_a.release(),
            first_b.release(),
        )

    asyncio.run(run())


def test_request_and_response_share_adaptive_parent_without_leaking():
    async def run():
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(100, 1000, source="fake"),
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=0,
        )
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=800,
            body_budget_bytes=900,
            max_response_bytes=800,
            max_retained_bytes_per_request=800,
            memory_governor=governor,
        )
        first = await controller.acquire(initial_body_bytes=500)
        second = await controller.acquire()
        await second.reserve_response_bytes(300)
        assert governor.snapshot().reservations == {
            "request_body": 500,
            "buffered_response": 300,
        }

        with pytest.raises(UpstreamResponseBudgetExhausted):
            await second.reserve_response_bytes(1)
        assert second.reserved_response_bytes == 300

        await first.release()
        assert governor.snapshot().reservations == {"buffered_response": 300}
        await second.release()
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(run())


def test_response_buffer_lifecycle_records_cross_retry_ownership():
    async def run():
        events = []
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(100, 1_000, source="fake"),
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=60,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=100,
            body_budget_bytes=800,
            max_response_bytes=100,
            memory_governor=governor,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        lease = await controller.acquire()
        lease.observe_body(
            RequestBodyObservation(request_id="request-1", trace_id="trace-1")
        )
        first_entry = {}
        lease.begin_response_attempt(
            first_entry,
            routing_attempt_id="attempt-1",
            routing_attempt_index=1,
            provider="provider-a",
            request_model="model-x",
            actual_model="model-x",
        )
        first = await lease.reserve_temporary_response_bytes(0)
        await first.reserve(6)
        await first.commit()
        lease.finish_response_attempt(outcome="failed")

        second_entry = {}
        lease.begin_response_attempt(
            second_entry,
            routing_attempt_id="attempt-2",
            routing_attempt_index=2,
            provider="provider-b",
            request_model="model-x",
            actual_model="model-x",
        )
        second = await lease.reserve_temporary_response_bytes(0)
        await second.release()
        lease.finish_response_attempt(outcome="succeeded")
        await lease.release()

        assert first_entry["response_buffer_reserved_before_bytes"] == 0
        assert first_entry["response_buffer_reserved_after_bytes"] == 6
        assert first_entry["response_buffer_retained_after_failed_attempt"] is True
        assert second_entry["response_buffer_retained_from_prior_attempts_bytes"] == 6
        assert second_entry["response_buffer_cross_retry_retained"] is True

        assert [event.event for event in events] == [
            "attempt_summary",
            "attempt_summary",
        ]
        first_summary, second_summary = events
        assert first_summary.routing_attempt_id == "attempt-1"
        assert first_summary.reserve_started_count == 1
        assert first_summary.allocation_reserve_call_count == 2
        assert first_summary.commit_count == 1
        assert first_summary.committed_bytes == 6
        assert first_summary.release_count == 1
        assert first_summary.released_bytes == 6
        assert first_summary.request_self_request_id == "request-1"
        assert first_summary.crosses_retry_boundary is True
        assert first_summary.request_response_reserved_after_bytes == 0
        assert second_summary.routing_attempt_id == "attempt-2"
        assert second_summary.rollback_count == 1
        assert second_summary.crosses_retry_boundary is True
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(run())


def test_response_buffer_lifecycle_is_bounded_per_attempt_for_stream_frames():
    async def run():
        events = []
        transport_outcomes = []

        class Diagnostics:
            def finalize(self, outcome):
                transport_outcomes.append(outcome)

        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1_024,
            body_budget_bytes=1_024,
            max_response_bytes=1_024,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        lease = await controller.acquire()
        lease.begin_response_attempt(
            {},
            routing_attempt_id="stream-attempt",
            routing_attempt_index=1,
            provider="provider-a",
            request_model="model-x",
            actual_model="model-x",
            transport_diagnostics=Diagnostics(),
        )
        lease.finish_response_attempt(
            outcome="stream_pending",
            keep_active=True,
        )
        for _ in range(100):
            reservation = await lease.reserve_temporary_response_bytes(1)
            await reservation.release()
        lease.finish_response_attempt(outcome="stream_completed")
        await lease.release()

        assert len(events) == 1
        summary = events[0]
        assert summary.event == "attempt_summary"
        assert summary.outcome == "stream_completed"
        assert summary.reserve_started_count == 100
        assert summary.allocation_reserve_call_count == 100
        assert summary.rollback_count == 100
        assert summary.rolled_back_bytes == 100
        assert transport_outcomes == ["stream_pending", "stream_completed"]

    asyncio.run(run())


def test_stream_terminal_outcome_wins_after_request_release_is_deferred():
    async def run():
        events = []
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1_024,
            body_budget_bytes=1_024,
            max_response_bytes=1_024,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        lease = await controller.acquire()
        lease.begin_response_attempt(
            {},
            routing_attempt_id="stream-attempt",
            routing_attempt_index=1,
            provider="provider-a",
            request_model="model-x",
            actual_model="model-x",
        )
        lease.finish_response_attempt(
            outcome="stream_pending",
            keep_active=True,
        )
        reservation = await lease.reserve_temporary_response_bytes(1)

        await lease.release()
        await reservation.release()
        lease.finish_response_attempt(outcome="stream_completed")

        assert len(events) == 1
        assert events[0].outcome == "stream_completed"

    asyncio.run(run())


def test_stream_summary_timeout_is_reported_explicitly(monkeypatch):
    monkeypatch.setattr(
        admission_core,
        "_STREAM_RESPONSE_SUMMARY_FALLBACK_SECONDS",
        0,
    )

    async def run():
        events = []
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1_024,
            body_budget_bytes=1_024,
            max_response_bytes=1_024,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        lease = await controller.acquire()
        lease.begin_response_attempt(
            {},
            routing_attempt_id="stream-attempt",
            routing_attempt_index=1,
            provider="provider-a",
            request_model="model-x",
            actual_model="model-x",
        )
        lease.finish_response_attempt(
            outcome="stream_pending",
            keep_active=True,
        )
        reservation = await lease.reserve_temporary_response_bytes(1)

        await lease.release()
        await reservation.release()
        await asyncio.sleep(0)

        assert len(events) == 1
        assert events[0].outcome == "stream_terminal_unobserved"

    asyncio.run(run())


def test_response_buffer_observer_failure_never_changes_memory_ownership():
    def broken_observer(_event):
        raise RuntimeError("exporter unavailable")

    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=16,
            body_budget_bytes=16,
            max_response_bytes=16,
            response_buffer_observer=broken_observer,
        )
        lease = await controller.acquire()
        lease.begin_response_attempt(
            {},
            routing_attempt_id="attempt-1",
            routing_attempt_index=1,
            provider="provider-a",
            request_model="model-x",
            actual_model="model-x",
        )
        reservation = await lease.reserve_temporary_response_bytes(4)
        await reservation.release()
        await lease.release()

        snapshot = controller.snapshot()
        assert snapshot["reserved_response_bytes"] == 0
        assert snapshot["active"] == 0
        assert snapshot["response_buffer_event_observer_errors_total"] == 1

    asyncio.run(run())


def test_successful_preprocessing_reservations_do_not_emit_attempt_summaries():
    async def run():
        events = []
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=16,
            body_budget_bytes=16,
            max_response_bytes=16,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        lease = await controller.acquire()
        reservation = await lease.reserve_temporary_response_bytes(4)
        await reservation.release()
        await lease.release()

        assert events == []
        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


def test_response_rejections_report_exact_low_cardinality_branches():
    async def rejected_event(
        *,
        max_response=100,
        max_retained=100,
        body_budget=800,
        initial_body=0,
        existing_global_body=0,
        requested=1,
        current_memory=100,
    ):
        events = []
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(
                current_memory,
                1_000,
                source="fake",
            ),
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=60,
        )
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=800,
            body_budget_bytes=body_budget,
            max_response_bytes=max_response,
            max_retained_bytes_per_request=max_retained,
            memory_governor=governor,
            response_buffer_observer=lambda event: events.append(event) or True,
        )
        body_owner = None
        if existing_global_body:
            body_owner = await controller.acquire(
                initial_body_bytes=existing_global_body
            )
        lease = await controller.acquire(initial_body_bytes=initial_body)
        lease.observe_body(RequestBodyObservation(request_id="reject-request"))
        with pytest.raises(UpstreamResponseBudgetExhausted) as caught:
            await lease.reserve_response_bytes(requested)
        event = events[-1]
        await lease.release()
        if body_owner is not None:
            await body_owner.release()
        return caught.value, event, controller.snapshot()

    async def run():
        response_limit, response_event, _ = await rejected_event(
            max_response=5,
            requested=6,
        )
        assert response_limit.admission_branch == "per_request_response_limit"
        assert response_event.requested_bytes == 6
        assert response_event.request_response_reserved_before_bytes == 0
        assert response_event.request_response_reserved_projected_bytes == 6

        retained_limit, retained_event, _ = await rejected_event(
            max_response=10,
            max_retained=5,
            initial_body=4,
            requested=2,
        )
        assert retained_limit.admission_branch == "per_request_retained_limit"
        assert retained_event.request_retained_reserved_before_bytes == 4
        assert retained_event.request_retained_reserved_projected_bytes == 6

        global_limit, global_event, _ = await rejected_event(
            max_response=10,
            max_retained=10,
            body_budget=5,
            existing_global_body=4,
            requested=2,
        )
        assert global_limit.admission_branch == "global_hard_budget"
        assert global_event.runtime_global_retained_reserved_before_bytes == 4
        assert global_event.runtime_global_retained_reserved_projected_bytes == 6

        parent_limit, parent_event, parent_snapshot = await rejected_event(
            max_response=200,
            max_retained=200,
            body_budget=800,
            requested=101,
            current_memory=800,
        )
        assert parent_limit.admission_branch == "parent_governor"
        assert parent_event.parent_governor_allowed is False
        assert parent_event.cgroup_memory_source == "fake"
        assert parent_event.cgroup_memory_current_bytes_sampled == 800
        assert parent_event.cgroup_memory_sample_sequence == 1
        assert parent_event.cgroup_memory_sample_age_ms_at_decision is not None
        assert parent_snapshot["response_rejection_decisions_by_branch"] == {
            "parent_governor": 1
        }

    asyncio.run(run())


def test_pending_transfer_and_temporary_commit_do_not_double_charge_parent():
    async def run():
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(100, 1000, source="fake"),
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=0,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=800,
            body_budget_bytes=900,
            max_response_bytes=800,
            memory_governor=governor,
        )
        pending = controller.pending_body_reservation()
        await pending.reserve(200)
        lease = await controller.acquire()
        assert await pending.transfer_to(lease) == 200
        assert governor.snapshot().reservations == {"request_body": 200}

        temporary = await lease.reserve_temporary_response_bytes(300)
        assert governor.snapshot().reserved_bytes == 500
        await temporary.commit()
        assert governor.snapshot().reserved_bytes == 500

        await lease.release()
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(run())


def test_per_request_body_and_response_total_remains_bounded():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=100,
            body_budget_bytes=200,
            max_response_bytes=100,
            max_retained_bytes_per_request=100,
        )
        lease = await controller.acquire(initial_body_bytes=60)
        await lease.reserve_response_bytes(40)
        with pytest.raises(UpstreamResponseBudgetExhausted):
            await lease.reserve_response_bytes(1)
        assert lease.reserved_body_bytes == 60
        assert lease.reserved_response_bytes == 40
        await lease.release()

    asyncio.run(run())


def test_body_and_buffered_response_share_one_weighted_memory_budget():
    async def run():
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=20,
            body_budget_bytes=20,
            max_response_bytes=20,
        )
        body_owner = await controller.acquire(initial_body_bytes=8)
        response_owner = await controller.acquire()
        assert await response_owner.reserve_response_bytes(12) == 12
        assert controller.snapshot()["reserved_retained_bytes"] == 20

        with pytest.raises(UpstreamResponseBudgetExhausted):
            await response_owner.reserve_response_bytes(1)
        assert controller.snapshot()["reserved_response_bytes"] == 12

        await body_owner.release()
        assert await response_owner.reserve_response_bytes(1) == 13
        await response_owner.release()

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["reserved_response_bytes"] == 0
        assert snapshot["reserved_retained_bytes"] == 0

    asyncio.run(run())


def test_request_controller_failed_initial_reservations_release_active_slot():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=5,
            body_budget_bytes=4,
        )

        with pytest.raises(RequestBodyTooLarge):
            await controller.acquire(initial_body_bytes=6)
        assert controller.snapshot()["active"] == 0

        with pytest.raises(RequestBodyBudgetExhausted):
            await controller.acquire(initial_body_bytes=5)
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0
        assert snapshot["rejected"] == {
            "body_too_large": 1,
            "body_budget_exhausted": 1,
        }

    asyncio.run(run())


def test_request_lease_context_manager_releases_bytes_on_cancellation():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=10,
            body_budget_bytes=10,
        )
        entered = asyncio.Event()

        async def request_task():
            async with await controller.acquire(initial_body_bytes=7):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(request_task())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_body_bytes"] == 0

    asyncio.run(run())


def test_temporary_response_reservation_releases_before_request_end():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        request_lease = await controller.acquire()
        temporary = await request_lease.reserve_temporary_response_bytes(400)
        assert controller.snapshot()["reserved_response_bytes"] == 400

        await temporary.release()
        assert controller.snapshot()["reserved_response_bytes"] == 0
        assert request_lease.reserved_response_bytes == 0

        await request_lease.release()
        assert controller.snapshot()["active"] == 0

    asyncio.run(run())


def test_request_releases_active_but_defers_memory_for_live_child_owner():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        request_lease = await controller.acquire(initial_body_bytes=200)
        temporary = await request_lease.reserve_temporary_response_bytes(300)

        await request_lease.release()
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_retained_bytes"] == 500
        assert snapshot["deferred_memory_requests"] == 1
        assert snapshot["deferred_memory_bytes"] == 500

        await temporary.release()
        snapshot = controller.snapshot()
        assert snapshot["reserved_retained_bytes"] == 0
        assert snapshot["deferred_memory_requests"] == 0
        assert snapshot["deferred_memory_bytes"] == 0

    asyncio.run(run())


def test_explicit_cleanup_deferral_keeps_base_request_memory_accounted():
    async def run():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        request_lease = await controller.acquire(initial_body_bytes=700)
        deferral = await request_lease.defer_memory_release()

        await request_lease.release()
        assert controller.snapshot()["deferred_memory_bytes"] == 700

        await deferral.release()
        assert controller.snapshot()["reserved_retained_bytes"] == 0

    asyncio.run(run())


def test_admission_configuration_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        BoundedAdmissionGate(0, waiter_limit=1, wait_timeout_seconds=1)
    with pytest.raises(ValueError):
        BoundedAdmissionGate(1, waiter_limit=-1, wait_timeout_seconds=1)
    with pytest.raises(ValueError):
        BoundedAdmissionGate(1, waiter_limit=1, wait_timeout_seconds=0)
    with pytest.raises(ValueError):
        RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            max_body_bytes=-1,
            body_budget_bytes=1,
        )
