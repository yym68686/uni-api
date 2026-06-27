from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from core.log_config import logger


BACKGROUND_STREAM_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()


def background_stream_cleanup_snapshot() -> dict[str, int]:
    tasks = list(BACKGROUND_STREAM_CLEANUP_TASKS)
    return {
        "pending": sum(1 for task in tasks if not task.done()),
        "done": sum(1 for task in tasks if task.done()),
        "total": len(tasks),
    }


async def wait_background_stream_cleanup_tasks(timeout: float = 5.0) -> dict[str, int]:
    tasks = [task for task in list(BACKGROUND_STREAM_CLEANUP_TASKS) if not task.done()]
    if not tasks:
        return background_stream_cleanup_snapshot()

    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        logger.warning("Cancelled %s detached stream cleanup task(s) during shutdown", len(pending))

    BACKGROUND_STREAM_CLEANUP_TASKS.difference_update(done)
    BACKGROUND_STREAM_CLEANUP_TASKS.difference_update(pending)
    snapshot = background_stream_cleanup_snapshot()
    snapshot["completed_during_wait"] = len(done)
    snapshot["cancelled_during_wait"] = len(pending)
    return snapshot


def drain_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    uncancel = getattr(current_task, "uncancel", None)
    if callable(uncancel):
        while current_task is not None and current_task.cancelling():
            uncancel()


def track_background_stream_cleanup_task(task: asyncio.Task[Any], *, label: str) -> None:
    BACKGROUND_STREAM_CLEANUP_TASKS.add(task)

    def cleanup_done(done: asyncio.Task[Any]) -> None:
        BACKGROUND_STREAM_CLEANUP_TASKS.discard(done)
        if done.cancelled():
            logger.warning("%s cleanup task was cancelled after detach", label)
            return
        try:
            done.result()
        except BaseException as exc:
            logger.warning(
                "%s cleanup failed after detach",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(cleanup_done)


async def await_stream_cleanup_safely(awaitable: Any, *, label: str) -> bool:
    if awaitable is None or not hasattr(awaitable, "__await__"):
        return True

    drain_current_task_cancellation()
    cleanup_task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(cleanup_task)
        return True
    except asyncio.CancelledError:
        logger.warning(
            "%s cleanup was cancelled; waiting for cleanup to finish",
            label,
        )
        drain_current_task_cancellation()
        try:
            await asyncio.shield(cleanup_task)
            return True
        except asyncio.CancelledError:
            drain_current_task_cancellation()
            logger.warning(
                "%s cleanup was cancelled again; detached cleanup will continue",
                label,
            )
            track_background_stream_cleanup_task(cleanup_task, label=label)
            return True
        except GeneratorExit as final_exc:
            logger.warning(
                "%s cleanup was interrupted by generator close; detached cleanup will continue",
                label,
                exc_info=(type(final_exc), final_exc, final_exc.__traceback__),
            )
            track_background_stream_cleanup_task(cleanup_task, label=label)
            return True
        except BaseException as final_exc:
            logger.warning(
                "%s cleanup failed after cancellation",
                label,
                exc_info=(type(final_exc), final_exc, final_exc.__traceback__),
            )
            return False
    except GeneratorExit as exc:
        logger.warning(
            "%s cleanup was interrupted by generator close; detached cleanup will continue",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        track_background_stream_cleanup_task(cleanup_task, label=label)
        return True
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


async def close_async_iterator_safely(iterator: Any, *, label: str) -> bool:
    aclose = getattr(iterator, "aclose", None)
    if not callable(aclose):
        return True
    try:
        close_result = aclose()
    except RuntimeError as exc:
        logger.debug("%s async iterator close skipped: %s", label, exc)
        return True
    except BaseException as exc:
        logger.warning(
            "%s async iterator close failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return await await_stream_cleanup_safely(close_result, label=label)


async def call_cleanup_safely(cleanup: Any, *, label: str) -> bool:
    try:
        result = cleanup()
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return await await_stream_cleanup_safely(result, label=label)


async def force_release_httpcore_pool_request_safely(stream: Any, *, label: str) -> bool:
    pool = getattr(stream, "_pool", None)
    pool_request = getattr(stream, "_pool_request", None)
    if pool is None or pool_request is None:
        return True

    requests = getattr(pool, "_requests", None)
    pool_connections = getattr(pool, "_connections", None)
    connection = getattr(pool_request, "connection", None)
    if not isinstance(requests, list):
        requests = []
    if pool_request not in requests and connection is None:
        return True

    try:
        closing: list[Any] = []
        lock = getattr(pool, "_optional_thread_lock", None)
        if lock is not None:
            with lock:
                if pool_request in requests:
                    requests.remove(pool_request)
                if isinstance(pool_connections, list) and connection in pool_connections:
                    pool_connections.remove(connection)
                assign_requests = getattr(pool, "_assign_requests_to_connections", None)
                closing = list(assign_requests()) if callable(assign_requests) else closing
        else:
            if pool_request in requests:
                requests.remove(pool_request)
            if isinstance(pool_connections, list) and connection in pool_connections:
                pool_connections.remove(connection)
            assign_requests = getattr(pool, "_assign_requests_to_connections", None)
            closing = list(assign_requests()) if callable(assign_requests) else closing

        if connection is not None and all(candidate is not connection for candidate in closing):
            closing.append(connection)

        close_connections = getattr(pool, "_close_connections", None)
        if callable(close_connections):
            return await await_stream_cleanup_safely(
                close_connections(closing),
                label=label,
            )
        cleanup_ok = True
        for connection_to_close in closing:
            aclose = getattr(connection_to_close, "aclose", None)
            if callable(aclose):
                cleanup_ok = await call_cleanup_safely(
                    aclose,
                    label=f"{label} connection",
                ) and cleanup_ok
        return cleanup_ok
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


async def force_close_response_httpcore_stream_chain_safely(response: Any | None, *, label: str) -> bool:
    if response is None:
        return True

    stream = getattr(response, "stream", None)
    candidates: list[Any] = []
    current = stream
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        candidates.append(current)
        current = getattr(current, "_stream", None)

    cleanup_ok = True
    for candidate in candidates:
        aclose = getattr(candidate, "aclose", None)
        if callable(aclose):
            cleanup_ok = await call_cleanup_safely(
                aclose,
                label=label,
            ) and cleanup_ok
        cleanup_ok = await force_release_httpcore_pool_request_safely(
            candidate,
            label=label,
        ) and cleanup_ok
    return cleanup_ok


async def yield_from_stream(stream: AsyncIterator[Any], *, label: str) -> AsyncIterator[Any]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await close_async_iterator_safely(stream, label=label)
