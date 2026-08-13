import asyncio
import os
from pathlib import Path

from uni_api.admission.memory import (
    AdaptiveMemoryGovernor,
    CgroupMemorySource,
    ProcessMemorySample,
    SharedMemoryReservationLedger,
)


def test_shared_ledger_makes_cross_process_reservations_visible(tmp_path):
    path = tmp_path / "shared-memory-reservations"
    first_ledger = SharedMemoryReservationLedger(path, reset=True)
    second_ledger = SharedMemoryReservationLedger(path)
    memory = FakeMemory(current=100, limit=1000)
    first = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=0,
        shared_ledger=first_ledger,
    )
    second = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=0,
        shared_ledger=second_ledger,
    )

    assert first.reserve_nowait("request_body", 500)
    assert second.snapshot().reserved_bytes == 500
    assert second.snapshot().available_bytes == 300
    assert first_ledger.categories()["parsed_body"] == 500
    assert not second.reserve_nowait("upstream_serialized_body", 301)
    assert second.reserve_nowait("upstream_serialized_body", 300)
    assert first.snapshot().reserved_bytes == 800
    assert first_ledger.categories()["serialized_body"] == 300

    second.release("upstream_serialized_body", 300)
    first.release("request_body", 500)
    assert first_ledger.total() == 0
    assert first_ledger.categories() == {
        "total": 0,
        "parsed_body": 0,
        "serialized_body": 0,
        "transport_buffer": 0,
        "response_buffer": 0,
        "other": 0,
    }


class FakeMemory:
    def __init__(
        self,
        *,
        current: int | None,
        limit: int | None,
        high: int | None = None,
    ) -> None:
        self.current = current
        self.limit = limit
        self.high = high
        self.events: dict[str, int] = {}

    def sample(self) -> ProcessMemorySample:
        return ProcessMemorySample(
            current_bytes=self.current,
            limit_bytes=self.limit,
            high_bytes=self.high,
            events=dict(self.events),
            source="fake",
        )


def test_dynamic_parent_budget_uses_limit_current_reservations_and_guard():
    memory = FakeMemory(current=100, limit=1000)
    governor = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        fallback_budget_bytes=50,
        sample_cache_seconds=0,
    )

    snapshot = governor.snapshot()
    assert snapshot.soft_limit_bytes == 900
    assert snapshot.capacity_bytes == 800
    assert snapshot.available_bytes == 800

    assert governor.reserve_nowait("body", 500)
    snapshot = governor.snapshot()
    assert snapshot.reserved_bytes == 500
    assert snapshot.available_bytes == 300
    assert not governor.reserve_nowait("stream", 301)

    memory.current = 700
    snapshot = governor.snapshot(force=True)
    assert snapshot.capacity_bytes == 500
    assert snapshot.available_bytes == 0
    assert not governor.reserve_nowait("body", 1)

    governor.release("body", 500)
    snapshot = governor.snapshot(force=True)
    assert snapshot.reserved_bytes == 0
    assert snapshot.available_bytes == 200


def test_reservation_decision_is_the_atomic_sample_used_for_rejection():
    clock = [10.0]
    governor = AdaptiveMemoryGovernor(
        source=lambda: ProcessMemorySample(
            current_bytes=700,
            limit_bytes=1_000,
            source="fake",
        ),
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=60,
        clock=lambda: clock[0],
    )

    allowed = governor.reserve_nowait_decision("body", 150)
    assert allowed.allowed is True
    assert allowed.reserved_before_bytes == 0
    assert allowed.projected_reserved_bytes == 150
    assert allowed.reserved_after_bytes == 150
    assert allowed.current_bytes == 700
    assert allowed.soft_limit_bytes == 900
    assert allowed.available_before_bytes == 200
    assert allowed.available_after_bytes == 50
    assert allowed.sample_sequence == 1
    assert allowed.sample_age_ms == 0

    clock[0] = 10.025
    rejected = governor.reserve_nowait_decision("buffered_response", 51)
    assert rejected.allowed is False
    assert rejected.reserved_before_bytes == 150
    assert rejected.projected_reserved_bytes == 201
    assert rejected.reserved_after_bytes == 150
    assert rejected.available_before_bytes == 50
    assert rejected.sample_sequence == 1
    assert rejected.sample_age_ms == 25


def test_cached_snapshot_is_io_free_and_reports_sample_sequence_and_age():
    clock = [10.0]
    calls = 0

    def sample() -> ProcessMemorySample:
        nonlocal calls
        calls += 1
        return ProcessMemorySample(
            current_bytes=100 + calls,
            limit_bytes=1_000,
            source="fake",
        )

    governor = AdaptiveMemoryGovernor(
        source=sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=60,
        clock=lambda: clock[0],
    )
    empty = governor.snapshot_cached()
    assert calls == 0
    assert empty.source == "unavailable"
    assert empty.sample_sequence == 0
    assert empty.sample_age_ms is None

    sampled = governor.snapshot(force=True)
    assert calls == 1
    assert sampled.current_bytes == 101
    assert sampled.sample_sequence == 1
    assert sampled.sample_age_ms == 0

    clock[0] = 10.125
    cached = governor.snapshot_cached()
    assert calls == 1
    assert cached.current_bytes == 101
    assert cached.sample_sequence == 1
    assert cached.sample_age_ms == 125


def test_failed_refresh_does_not_make_a_stale_cgroup_sample_look_fresh():
    clock = [10.0]
    fail = [False]

    def sample() -> ProcessMemorySample:
        if fail[0]:
            raise OSError("bounded cgroup read failure")
        return ProcessMemorySample(
            current_bytes=100,
            limit_bytes=1_000,
            source="fake",
        )

    governor = AdaptiveMemoryGovernor(
        source=sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=60,
        clock=lambda: clock[0],
    )
    assert governor.snapshot(force=True).sample_sequence == 1

    clock[0] = 20.0
    fail[0] = True
    stale = governor.snapshot(force=True)
    assert stale.current_bytes == 100
    assert stale.sample_sequence == 1
    assert stale.sample_age_ms == 10_000
    assert stale.sample_error == "OSError: bounded cgroup read failure"


def test_existing_reservations_survive_dynamic_downscale():
    memory = FakeMemory(current=100, limit=1000)
    governor = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=0,
    )
    assert governor.reserve_nowait("body", 700)

    memory.current = 950
    snapshot = governor.snapshot(force=True)
    assert snapshot.capacity_bytes == 700
    assert snapshot.available_bytes == 0
    assert snapshot.reservations == {"body": 700}

    governor.release("body", 700)
    assert governor.snapshot(force=True).reserved_bytes == 0


def test_finite_memory_high_is_the_effective_parent_ceiling():
    memory = FakeMemory(current=100, limit=2000, high=1000)
    governor = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        sample_cache_seconds=0,
    )
    snapshot = governor.snapshot()
    assert snapshot.limit_bytes == 2000
    assert snapshot.high_bytes == 1000
    assert snapshot.soft_limit_bytes == 900
    assert snapshot.capacity_bytes == 800


def test_default_guard_scales_safely_across_small_and_production_limits():
    mib = 1024 * 1024
    expected = {
        256: (128, 128, 52),
        512: (256, 256, 180),
        4032: (1008, 3024, 2948),
    }
    for limit_mib, (guard_mib, soft_mib, capacity_mib) in expected.items():
        governor = AdaptiveMemoryGovernor(
            source=lambda limit=limit_mib: ProcessMemorySample(
                current_bytes=76 * mib,
                limit_bytes=limit * mib,
                source="fake",
            ),
            sample_cache_seconds=0,
        )
        snapshot = governor.snapshot()
        assert snapshot.guard_bytes == guard_mib * mib
        assert snapshot.soft_limit_bytes == soft_mib * mib
        assert snapshot.capacity_bytes == capacity_mib * mib
        assert governor.reserve_nowait("request_body", 1)
        governor.release("request_body", 1)


def test_resolves_nested_cgroup_v1_memory_controller(tmp_path):
    nested = tmp_path / "memory" / "kubepods" / "pod" / "container"
    nested.mkdir(parents=True)
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text(
        "2:cpu,cpuacct:/kubepods/pod/container\n"
        "3:memory:/kubepods/pod/container\n",
        encoding="ascii",
    )
    (nested / "memory.usage_in_bytes").write_text("100\n", encoding="ascii")
    (nested / "memory.limit_in_bytes").write_text("1000\n", encoding="ascii")
    (nested / "memory.failcnt").write_text("7\n", encoding="ascii")

    sample = CgroupMemorySource(tmp_path, proc_cgroup).sample()
    assert sample.current_bytes == 100
    assert sample.limit_bytes == 1000
    assert sample.events == {"max": 7}
    assert sample.source == "cgroup-v1"


def test_cgroup_v2_source_reuses_descriptors_but_reads_fresh_values(
    tmp_path,
    monkeypatch,
):
    nested = tmp_path / "kubepods.slice" / "pod.scope"
    nested.mkdir(parents=True)
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text(
        "0::/kubepods.slice/pod.scope\n",
        encoding="ascii",
    )
    paths = {
        "memory.current": "100\n",
        "memory.max": "1000\n",
        "memory.high": "900\n",
        "memory.events": "high 1\nmax 2\n",
    }
    for name, content in paths.items():
        (nested / name).write_text(content, encoding="ascii")

    opened = []
    real_open = os.open

    def recording_open(path, flags):
        opened.append(Path(path))
        return real_open(path, flags)

    monkeypatch.setattr("uni_api.admission.memory.os.open", recording_open)
    source = CgroupMemorySource(tmp_path, proc_cgroup)
    try:
        first = source.sample()
        assert first.current_bytes == 100
        assert first.limit_bytes == 1000
        assert first.high_bytes == 900
        assert first.events == {"high": 1, "max": 2}

        (nested / "memory.current").write_text("250\n", encoding="ascii")
        (nested / "memory.max").write_text("1200\n", encoding="ascii")
        (nested / "memory.high").write_text("1100\n", encoding="ascii")
        (nested / "memory.events").write_text(
            "high 3\nmax 4\n",
            encoding="ascii",
        )

        second = source.sample()
        assert second.current_bytes == 250
        assert second.limit_bytes == 1200
        assert second.high_bytes == 1100
        assert second.events == {"high": 3, "max": 4}
        assert opened == [nested / name for name in paths]
    finally:
        source.close()


def test_unlimited_environment_preserves_finite_fallback_budget():
    memory = FakeMemory(current=100, limit=None)
    governor = AdaptiveMemoryGovernor(
        source=memory.sample,
        fallback_budget_bytes=256,
        sample_cache_seconds=0,
    )
    assert governor.snapshot().capacity_bytes == 256
    assert governor.reserve_nowait("body", 256)
    assert not governor.reserve_nowait("body", 1)
    governor.release("body", 256)


def test_async_waiter_is_woken_by_cross_category_release():
    async def scenario():
        memory = FakeMemory(current=100, limit=1000)
        governor = AdaptiveMemoryGovernor(
            source=memory.sample,
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=0,
        )
        assert governor.reserve_nowait("body", 800)

        waiting = asyncio.create_task(
            governor.reserve("stream", 100, timeout_seconds=1)
        )
        await asyncio.sleep(0)
        assert governor.snapshot().waiting_reservations == 1
        governor.release("body", 100)
        assert await waiting is True
        assert governor.snapshot().reservations == {"body": 700, "stream": 100}
        governor.release("stream", 100)
        governor.release("body", 700)

    asyncio.run(scenario())


def test_async_waiter_timeout_does_not_leak_reservation():
    async def scenario():
        memory = FakeMemory(current=100, limit=1000)
        governor = AdaptiveMemoryGovernor(
            source=memory.sample,
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=0,
        )
        assert governor.reserve_nowait("body", 800)
        assert not await governor.reserve("stream", 1, timeout_seconds=0.01)
        snapshot = governor.snapshot()
        assert snapshot.waiting_reservations == 0
        assert snapshot.wait_timeouts == 1
        assert snapshot.reservations == {"body": 800}
        governor.release("body", 800)

    asyncio.run(scenario())


def test_async_waiter_observes_cgroup_memory_drop_without_tracked_release():
    async def scenario():
        memory = FakeMemory(current=100, limit=1000)
        governor = AdaptiveMemoryGovernor(
            source=memory.sample,
            guard_bytes=100,
            guard_ratio=0,
            sample_cache_seconds=0,
        )
        assert governor.reserve_nowait("body", 800)
        waiting = asyncio.create_task(
            governor.reserve("stream", 50, timeout_seconds=1)
        )
        await asyncio.sleep(0)
        memory.current = 0
        assert await asyncio.wait_for(waiting, timeout=0.3) is True
        governor.release("stream", 50)
        governor.release("body", 800)

    asyncio.run(scenario())


def test_sampling_failure_preserves_existing_ownership_but_forbids_expansion():
    class FailingMemory(FakeMemory):
        fail = False

        def sample(self) -> ProcessMemorySample:
            if self.fail:
                raise OSError("cgroup unavailable")
            return super().sample()

    memory = FailingMemory(current=100, limit=2000)
    governor = AdaptiveMemoryGovernor(
        source=memory.sample,
        guard_bytes=100,
        guard_ratio=0,
        fallback_budget_bytes=256,
        sample_cache_seconds=0,
    )
    assert governor.reserve_nowait("body", 300)
    memory.fail = True
    snapshot = governor.snapshot(force=True)
    assert snapshot.sample_error == "OSError: cgroup unavailable"
    assert snapshot.capacity_bytes == 300
    assert snapshot.available_bytes == 0
    assert not governor.reserve_nowait("body", 1)
    governor.release("body", 300)

    snapshot = governor.snapshot(force=True)
    assert snapshot.capacity_bytes == 256
    assert snapshot.available_bytes == 256
