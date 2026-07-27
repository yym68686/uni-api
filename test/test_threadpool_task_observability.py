import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from uni_api.observability.threadpool_tasks import ThreadpoolTaskRegistry


def test_threadpool_registry_classifies_lifecycle_without_task_data():
    registry = ThreadpoolTaskRegistry()
    ticket = registry.submitted("json_parse")

    queued = registry.snapshot()["categories"]["json_parse"]
    assert queued["submitted_total"] == 1
    assert queued["queued"] == 1
    assert registry.snapshot()["lifecycle_semantics"] == (
        "explicit_task_tag_wall_thread_cpu_v1"
    )

    def callback():
        active = registry.active_categories()
        return active[threading.get_native_id()]

    assert ticket.run(callback) == "json_parse"
    metrics = registry.snapshot()["categories"]["json_parse"]
    assert metrics["started_total"] == 1
    assert metrics["completed_total"] == 1
    assert metrics.get("failed_total", 0) == 0
    assert metrics["queued"] == 0
    assert metrics["inflight"] == 0
    assert metrics["active_threads"] == 0
    assert metrics["queue_wait_ns_total"] >= 0
    assert metrics["wall_ns_total"] >= 0
    assert metrics["cpu_ns_total"] >= 0


def test_threadpool_registry_bounds_category_and_counts_failures():
    registry = ThreadpoolTaskRegistry()
    ticket = registry.submitted("request-body-or-secret-controlled-category")

    with pytest.raises(RuntimeError, match="sentinel"):
        ticket.run(lambda: (_ for _ in ()).throw(RuntimeError("sentinel")))

    snapshot = registry.snapshot()
    assert set(snapshot["categories"]) == {"other"}
    metrics = snapshot["categories"]["other"]
    assert metrics["completed_total"] == 1
    assert metrics["failed_total"] == 1
    assert "sentinel" not in repr(snapshot)


def test_cancelled_waiter_does_not_leave_phantom_queued_task():
    registry = ThreadpoolTaskRegistry()
    ticket = registry.submitted("json_serialization")

    assert ticket.cancel_if_queued() is True
    assert ticket.cancel_if_queued() is False
    metrics = registry.snapshot()["categories"]["json_serialization"]
    assert metrics["queued"] == 0
    assert metrics["cancelled_total"] == 1

    # Exercise the executor-claim race: the callback can still start after the
    # asyncio waiter observed cancellation, and must remain fully accounted.
    assert ticket.run(lambda: "completed") == "completed"
    metrics = registry.snapshot()["categories"]["json_serialization"]
    assert metrics["cancelled_task_started_total"] == 1
    assert metrics["completed_total"] == 1
    assert metrics["queued"] == 0


def test_dedicated_executor_state_is_sampled_without_wrapping_tasks():
    registry = ThreadpoolTaskRegistry()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="uni-api-json-test",
    )
    try:
        registry.register_dedicated("json_parse", executor)
        assert executor.submit(lambda: 1).result() == 1
        snapshot = registry.snapshot()
    finally:
        executor.shutdown(wait=True)

    assert snapshot["dedicated_executors"]["json_parse"] == {
        "queue_depth": 0,
        "threads": 1,
        "alive_threads": 1,
    }
    assert snapshot["categories"] == {}
