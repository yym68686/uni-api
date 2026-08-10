import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import uni_api.admission.cpu as cpu_module
from uni_api.admission.cpu import CPUPhaseLimiter


def test_cpu_phase_tokens_are_shared_across_executor_groups(monkeypatch):
    async def scenario():
        limiter = CPUPhaseLimiter(1)
        monkeypatch.setattr(cpu_module, "CPU_PHASE_LIMITER", limiter)
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()

        def first_callback():
            first_started.set()
            release_first.wait(timeout=2)
            return "first"

        def second_callback():
            second_started.set()
            return "second"

        first_executor = ThreadPoolExecutor(max_workers=1)
        second_executor = ThreadPoolExecutor(max_workers=1)
        try:
            first = asyncio.create_task(
                cpu_module.run_cpu_phase(
                    first_executor,
                    first_callback,
                    phase="request_body",
                )
            )
            assert await asyncio.to_thread(first_started.wait, 1)
            second = asyncio.create_task(
                cpu_module.run_cpu_phase(
                    second_executor,
                    second_callback,
                    phase="json",
                )
            )
            await asyncio.sleep(0.05)

            snapshot = limiter.snapshot()
            assert snapshot["active"] == 1
            assert snapshot["waiters"] == 1
            assert second_started.is_set() is False

            release_first.set()
            assert await first == "first"
            assert await second == "second"
            assert limiter.snapshot()["completed_total"] == 2
            assert limiter.snapshot()["active"] == 0
            assert limiter.snapshot()["waiters"] == 0
        finally:
            release_first.set()
            first_executor.shutdown(wait=True)
            second_executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_cancelled_cpu_phase_waiter_does_not_consume_a_token(monkeypatch):
    async def scenario():
        limiter = CPUPhaseLimiter(1)
        monkeypatch.setattr(cpu_module, "CPU_PHASE_LIMITER", limiter)
        first_started = threading.Event()
        release_first = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)

        def blocking_callback():
            first_started.set()
            release_first.wait(timeout=2)

        try:
            first = asyncio.create_task(
                cpu_module.run_cpu_phase(
                    executor,
                    blocking_callback,
                    phase="json",
                )
            )
            assert await asyncio.to_thread(first_started.wait, 1)
            waiter = asyncio.create_task(
                cpu_module.run_cpu_phase(
                    executor,
                    lambda: None,
                    phase="upstream_response",
                )
            )
            await asyncio.sleep(0.05)
            assert limiter.snapshot()["waiters"] == 1

            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass
            assert limiter.snapshot()["active"] == 1
            assert limiter.snapshot()["waiters"] == 0

            release_first.set()
            await first
            assert limiter.snapshot()["active"] == 0
        finally:
            release_first.set()
            executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_cpu_phase_limiter_rebinds_after_an_event_loop_drains():
    limiter = CPUPhaseLimiter(1)

    async def contend_once():
        await limiter.acquire("json")
        waiter = asyncio.create_task(limiter.acquire("request_body"))
        await asyncio.sleep(0)
        assert limiter.snapshot()["waiters"] == 1
        limiter.release("json", cancelled=False, failed=False)
        await waiter
        limiter.release("request_body", cancelled=False, failed=False)

    asyncio.run(contend_once())
    asyncio.run(contend_once())

    assert limiter.snapshot()["completed_total"] == 4
