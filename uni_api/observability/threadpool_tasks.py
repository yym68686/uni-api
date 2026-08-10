from __future__ import annotations

import threading
import time
import weakref
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable


THREADPOOL_TASK_CATEGORIES = frozenset(
    {
        "json_parse",
        "json_serialization",
        "network_procfs",
        "on_cpu_profile",
        "request_body_decode",
        "upstream_response_decode",
        "idempotency_spool",
    }
)


def _category(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in THREADPOOL_TASK_CATEGORIES else "other"


class ThreadpoolTaskRegistry:
    """Bounded task-class metrics for worker threads.

    Only fixed category names and numeric counters are retained. Callbacks,
    arguments, return values, exceptions, and request data never enter the
    registry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, Counter[str]] = {}
        self._active_by_native_id: dict[int, str] = {}
        self._dedicated_executors: dict[str, weakref.ReferenceType[Any]] = {}

    def register_dedicated(self, category: str, executor: Any) -> None:
        normalized = _category(category)
        try:
            reference = weakref.ref(executor)
        except TypeError:
            return
        with self._lock:
            self._dedicated_executors[normalized] = reference

    def submitted(self, category: str) -> ThreadpoolTaskTicket:
        normalized = _category(category)
        submitted_ns = time.perf_counter_ns()
        with self._lock:
            metrics = self._metrics.setdefault(normalized, Counter())
            metrics["submitted_total"] += 1
            metrics["queued"] += 1
        return ThreadpoolTaskTicket(
            registry=self,
            category=normalized,
            submitted_ns=submitted_ns,
        )

    def _run(
        self,
        ticket: ThreadpoolTaskTicket,
        callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.thread_time_ns()
        native_id = threading.get_native_id()
        previous_category = None
        with self._lock:
            metrics = self._metrics.setdefault(ticket.category, Counter())
            if ticket._state not in {"queued", "cancelled"}:
                raise RuntimeError("threadpool task ticket can only run once")
            if ticket._state == "queued":
                metrics["queued"] = max(0, metrics["queued"] - 1)
            else:
                # The asyncio waiter can race with an executor worker that has
                # already claimed the callback. Preserve both facts instead
                # of leaving a phantom queued task.
                metrics["cancelled_task_started_total"] += 1
            ticket._state = "running"
            metrics["started_total"] += 1
            metrics["inflight"] += 1
            metrics["queue_wait_ns_total"] += max(
                0,
                started_ns - ticket.submitted_ns,
            )
            previous_category = self._active_by_native_id.get(native_id)
            self._active_by_native_id[native_id] = ticket.category
        failed = False
        try:
            return callback(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            finished_ns = time.perf_counter_ns()
            cpu_finished_ns = time.thread_time_ns()
            with self._lock:
                metrics = self._metrics.setdefault(ticket.category, Counter())
                metrics["completed_total"] += 1
                if failed:
                    metrics["failed_total"] += 1
                metrics["inflight"] = max(0, metrics["inflight"] - 1)
                metrics["wall_ns_total"] += max(0, finished_ns - started_ns)
                metrics["cpu_ns_total"] += max(
                    0,
                    cpu_finished_ns - cpu_started_ns,
                )
                ticket._state = "completed"
                if previous_category is None:
                    self._active_by_native_id.pop(native_id, None)
                else:
                    self._active_by_native_id[native_id] = previous_category

    def _cancel_if_queued(self, ticket: ThreadpoolTaskTicket) -> bool:
        with self._lock:
            if ticket._state != "queued":
                return False
            ticket._state = "cancelled"
            metrics = self._metrics.setdefault(ticket.category, Counter())
            metrics["queued"] = max(0, metrics["queued"] - 1)
            metrics["cancelled_total"] += 1
            return True

    def active_categories(self) -> dict[int, str]:
        with self._lock:
            return dict(self._active_by_native_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = {
                category: dict(metrics)
                for category, metrics in sorted(self._metrics.items())
            }
            active = Counter(self._active_by_native_id.values())
            dedicated_references = dict(self._dedicated_executors)
        for category, metrics in rows.items():
            metrics.setdefault("queued", 0)
            metrics.setdefault("inflight", 0)
            metrics["active_threads"] = int(active.get(category, 0))
        dedicated: dict[str, dict[str, int]] = {}
        for category, reference in sorted(dedicated_references.items()):
            executor = reference()
            if executor is None:
                continue
            row: dict[str, int] = {}
            queue = getattr(executor, "_work_queue", None)
            qsize = getattr(queue, "qsize", None)
            if callable(qsize):
                try:
                    row["queue_depth"] = max(0, int(qsize()))
                except (NotImplementedError, OSError, TypeError, ValueError):
                    pass
            threads = getattr(executor, "_threads", None)
            if isinstance(threads, set):
                try:
                    captured_threads = tuple(threads)
                    row["threads"] = len(captured_threads)
                    row["alive_threads"] = sum(
                        1 for thread in captured_threads if thread.is_alive()
                    )
                except (RuntimeError, TypeError):
                    pass
            if row:
                dedicated[category] = row
        return {
            "schema_version": 1,
            "lifecycle_semantics": "explicit_task_tag_wall_thread_cpu_v1",
            "categories": rows,
            "dedicated_executors": dedicated,
        }


@dataclass(slots=True)
class ThreadpoolTaskTicket:
    registry: ThreadpoolTaskRegistry
    category: str
    submitted_ns: int
    _state: str = "queued"

    def run(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.registry._run(self, callback, args, kwargs)

    def cancel_if_queued(self) -> bool:
        return self.registry._cancel_if_queued(self)


_REGISTRY = ThreadpoolTaskRegistry()


def submit_threadpool_task(category: str) -> ThreadpoolTaskTicket:
    return _REGISTRY.submitted(category)


def register_dedicated_threadpool(category: str, executor: Any) -> None:
    _REGISTRY.register_dedicated(category, executor)


def threadpool_task_snapshot() -> dict[str, Any]:
    return _REGISTRY.snapshot()


def active_threadpool_task_categories() -> dict[int, str]:
    return _REGISTRY.active_categories()
