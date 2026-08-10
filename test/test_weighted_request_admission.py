import asyncio

import pytest

from uni_api.admission import AdmissionRejected, RequestAdmissionController
from uni_api.admission.memory import (
    AdaptiveMemoryGovernor,
    ProcessMemorySample,
)


def _controller(*, memory_bytes: int, control_bytes: int):
    governor = AdaptiveMemoryGovernor(
        source=lambda: ProcessMemorySample(
            current_bytes=0,
            limit_bytes=memory_bytes,
            source="test",
        ),
        guard_bytes=0,
        guard_ratio=0,
        sample_cache_seconds=0,
    )
    controller = RequestAdmissionController(
        capacity=None,
        waiter_limit=None,
        wait_timeout_seconds=1,
        control_memory_bytes=control_bytes,
        max_body_bytes=memory_bytes,
        body_budget_bytes=memory_bytes,
        large_body_threshold_weighted_bytes=0,
        large_body_limit=0,
        body_memory_wait_timeout_seconds=1,
        body_memory_waiter_limit=None,
        memory_governor=governor,
    )
    return controller, governor


def test_weighted_request_admission_has_no_request_count_capacity():
    async def scenario():
        controller, governor = _controller(
            memory_bytes=64 * 1024 * 1024,
            control_bytes=64 * 1024,
        )
        leases = [await controller.acquire() for _ in range(1000)]

        snapshot = controller.snapshot()
        assert snapshot["mode"] == "weighted_resources"
        assert snapshot["capacity"] is None
        assert snapshot["waiter_limit"] is None
        assert snapshot["active"] == 1000
        assert snapshot["control_memory_reserved_bytes"] == 64 * 1024 * 1000

        await asyncio.gather(*(lease.release() for lease in leases))
        assert controller.snapshot()["active"] == 0
        assert governor.snapshot().reservations == {}

    asyncio.run(scenario())


def test_weighted_request_admission_rejects_only_at_memory_guard():
    async def scenario():
        controller, governor = _controller(
            memory_bytes=1024 * 1024,
            control_bytes=256 * 1024,
        )
        leases = [await controller.acquire() for _ in range(4)]

        with pytest.raises(AdmissionRejected, match="memory_hard_guard"):
            await controller.acquire()

        snapshot = controller.snapshot()
        assert snapshot["active"] == 4
        assert snapshot["rejected"] == {"memory_hard_guard": 1}
        await asyncio.gather(*(lease.release() for lease in leases))
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(scenario())
