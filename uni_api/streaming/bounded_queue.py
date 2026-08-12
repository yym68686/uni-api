from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Any, Deque

from uni_api.admission.memory import AdaptiveMemoryGovernor
from uni_api.streaming.usage import StreamUsageSnapshot


class StreamQueueClosed(Exception):
    """Raised when a consumer reaches the end of a closed stream queue."""


class StreamQueueItemTooLarge(ValueError):
    """Raised when one item cannot fit in the configured byte budget."""

    status_code = 503
    reason = "upstream_stream_item_too_large"
    retry_after_seconds = 1
    local_admission_rejection = True


class StreamQueuePutTimeout(TimeoutError):
    """Raised when a slow consumer keeps the queue full for too long."""

    status_code = 503
    reason = "upstream_stream_queue_wait_timeout"
    retry_after_seconds = 1
    local_admission_rejection = True


class StreamBufferBudgetTimeout(RuntimeError):
    """Raised when the process-wide retained-byte budget stays exhausted."""

    status_code = 503
    reason = "upstream_stream_buffer_budget_exhausted"
    retry_after_seconds = 1
    local_admission_rejection = True


async def _reserve_adaptive_parent_safely(
    governor: AdaptiveMemoryGovernor,
    category: str,
    size: int,
    timeout_seconds: float,
) -> bool:
    """Own or release a same-turn parent reservation under cancellation."""

    task = asyncio.create_task(
        governor.reserve(
            category,
            size,
            timeout_seconds=timeout_seconds,
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            reserved = task.result()
        except asyncio.CancelledError:
            reserved = False
        if reserved:
            governor.release(category, size)
        raise


async def _release_leases_despite_cancellation(
    leases: list[RetainedByteLease],
) -> None:
    if not leases:
        return

    async def release_all() -> None:
        await asyncio.gather(*(lease.release() for lease in leases))

    task = asyncio.create_task(release_all())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


@dataclass(frozen=True, slots=True)
class StreamQueueSnapshot:
    # ``items``/``bytes`` include dequeued items whose downstream send has not
    # completed yet.  They are retained-memory gauges, not deque length.
    items: int
    bytes: int
    queued_items: int
    queued_bytes: int
    inflight_items: int
    inflight_bytes: int
    peak_items: int
    peak_bytes: int
    max_items: int
    max_bytes: int
    closed: bool
    waiting_putters: int
    blocked_puts: int
    put_wait_ms: float
    put_timeouts: int


@dataclass(frozen=True, slots=True)
class RetainedByteBudgetSnapshot:
    used_bytes: int
    peak_bytes: int
    capacity_bytes: int
    waiting_reservations: int
    blocked_reservations: int
    wait_ms: float
    timeouts: int


class RetainedByteLease:
    def __init__(self, budget: RetainedByteBudget, size: int) -> None:
        self._budget = budget
        self.size = size
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._released

    @property
    def budget(self) -> RetainedByteBudget:
        return self._budget

    def split(self, size: int) -> RetainedByteLease:
        """Transfer part of this already-reserved ownership to a child lease."""

        size = int(size)
        if size < 0 or size > self.size:
            raise ValueError("split size exceeds retained-byte lease")
        if self._released or self._release_task is not None:
            raise RuntimeError("cannot split a released retained-byte lease")
        self.size -= size
        return RetainedByteLease(self._budget, size)

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        task = self._release_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        await self._budget._release(self.size)
        self._released = True


class RetainedByteBudget:
    """A process-wide hard limit for bytes retained by streaming buffers."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        wait_timeout_seconds: float,
        memory_governor: AdaptiveMemoryGovernor | None = None,
        memory_category: str = "stream_buffer",
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.memory_governor = memory_governor
        self.memory_category = str(memory_category or "stream_buffer")
        self._used_bytes = 0
        self._peak_bytes = 0
        self._waiting_reservations = 0
        self._blocked_reservations = 0
        self._wait_ms = 0.0
        self._timeouts = 0
        self._condition = asyncio.Condition()

    def snapshot(self) -> RetainedByteBudgetSnapshot:
        effective_capacity = self.capacity_bytes
        if self.memory_governor is not None:
            effective_capacity = min(
                effective_capacity,
                self.memory_governor.snapshot().capacity_bytes,
            )
        return RetainedByteBudgetSnapshot(
            used_bytes=self._used_bytes,
            peak_bytes=self._peak_bytes,
            capacity_bytes=effective_capacity,
            waiting_reservations=self._waiting_reservations,
            blocked_reservations=self._blocked_reservations,
            wait_ms=self._wait_ms,
            timeouts=self._timeouts,
        )

    async def reserve(self, size: int) -> RetainedByteLease:
        size = int(size)
        if size < 0:
            raise ValueError("reservation size cannot be negative")
        if size > self.capacity_bytes:
            raise StreamQueueItemTooLarge(
                f"stream buffer reservation is {size} bytes; global limit is "
                f"{self.capacity_bytes}"
            )
        started_at = monotonic()

        async def wait_and_reserve_local() -> bool:
            async with self._condition:
                blocked = self._used_bytes + size > self.capacity_bytes
                if blocked:
                    self._blocked_reservations += 1
                    self._waiting_reservations += 1
                    try:
                        await self._condition.wait_for(
                            lambda: self._used_bytes + size <= self.capacity_bytes
                        )
                    finally:
                        self._waiting_reservations = max(
                            0, self._waiting_reservations - 1
                        )
                self._used_bytes += size
                self._peak_bytes = max(self._peak_bytes, self._used_bytes)
                if blocked:
                    self._wait_ms += (monotonic() - started_at) * 1000.0
                return blocked

        try:
            async with asyncio.timeout(self.wait_timeout_seconds):
                await wait_and_reserve_local()
                parent_reserved = False
                try:
                    if self.memory_governor is not None:
                        parent_started_at = monotonic()
                        remaining = max(
                            0.001,
                            self.wait_timeout_seconds - (monotonic() - started_at),
                        )
                        parent_reserved = await _reserve_adaptive_parent_safely(
                            self.memory_governor,
                            self.memory_category,
                            size,
                            remaining,
                        )
                        if not parent_reserved:
                            raise TimeoutError
                        self._wait_ms += max(
                            0.0,
                            (monotonic() - parent_started_at) * 1000.0,
                        )
                    return RetainedByteLease(self, size)
                except BaseException:
                    if parent_reserved and self.memory_governor is not None:
                        self.memory_governor.release(self.memory_category, size)
                    async with self._condition:
                        if size > self._used_bytes:
                            raise RuntimeError(
                                "stream retained-byte budget rollback underflow"
                            )
                        self._used_bytes -= size
                        self._condition.notify_all()
                    raise
        except TimeoutError as exc:
            self._timeouts += 1
            raise StreamBufferBudgetTimeout(
                "process-wide streaming byte budget stayed full for "
                f"{self.wait_timeout_seconds:g} seconds"
            ) from exc

    async def _release(self, size: int) -> None:
        async with self._condition:
            if size > self._used_bytes:
                raise RuntimeError("stream retained-byte budget underflow")
            self._used_bytes -= size
            self._condition.notify_all()
        if self.memory_governor is not None:
            self.memory_governor.release(self.memory_category, size)


class StreamQueueItemLease:
    """An item remains accounted until its downstream send completes."""

    def __init__(
        self,
        queue: ByteBoundedQueue,
        item: Any,
        size: int,
        budget_lease: RetainedByteLease | None,
    ) -> None:
        self._queue = queue
        self.item = item
        self.size = size
        self._budget_lease = budget_lease
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
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise

    async def _release_once(self) -> None:
        if self._released:
            return
        # Drop this owner's payload alias before returning its item/byte
        # capacity.  Callers must likewise discard aliases before release or
        # keep the lease through their downstream send.
        self.item = None
        await self._queue._release_inflight(self.size, self._budget_lease)
        self._released = True

    async def __aenter__(self) -> Any:
        return self.item

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


@dataclass(slots=True)
class ReservedStreamChunk:
    data: bytes
    reservation: RetainedByteLease
    event_type: str | None = None
    semantic_outcome: str | None = None
    terminal_timeline_event: bool = False
    sse_metadata_complete: bool = False
    usage_snapshot: StreamUsageSnapshot | None = None


@dataclass(slots=True)
class ObservedStreamFrame:
    """Protocol metadata around existing bytes without copying their payload."""

    data: bytes
    event_type: str | None = None
    semantic_outcome: str | None = None
    terminal_timeline_event: bool = False
    sse_metadata_complete: bool = False
    usage_snapshot: StreamUsageSnapshot | None = None


class ObservedStreamChunk(bytes):
    """Bytes plus bounded protocol metadata carried through ASGI unchanged.

    Starlette's ``StreamingResponse`` encodes every yielded object that is not
    already ``bytes``/``memoryview``.  Keeping the metadata on a real bytes
    subclass therefore preserves both the normal wire contract and the
    terminal-event acknowledgement while the chunk crosses
    ``BaseHTTPMiddleware``'s in-memory ASGI channel.
    """

    event_type: str | None
    semantic_outcome: str | None
    terminal_timeline_event: bool
    final_event_segment: bool
    sse_metadata_complete: bool
    usage_snapshot: StreamUsageSnapshot | None

    def __new__(
        cls,
        data: bytes | bytearray | memoryview,
        event_type: str | None = None,
        semantic_outcome: str | None = None,
        terminal_timeline_event: bool = False,
        final_event_segment: bool = True,
        sse_metadata_complete: bool = False,
        usage_snapshot: StreamUsageSnapshot | None = None,
    ) -> "ObservedStreamChunk":
        value = super().__new__(cls, data)
        value.event_type = event_type
        value.semantic_outcome = semantic_outcome
        value.terminal_timeline_event = bool(terminal_timeline_event)
        value.final_event_segment = bool(final_event_segment)
        value.sse_metadata_complete = bool(sse_metadata_complete)
        value.usage_snapshot = usage_snapshot
        return value

    @property
    def data(self) -> bytes:
        return self

    @property
    def body(self) -> bytes:
        return self


class ReservedChunkBuffer:
    """A finite precommit buffer whose bytes count against a shared budget."""

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int,
        retained_byte_budget: RetainedByteBudget | None = None,
    ) -> None:
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("chunk buffer limits must be positive")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self.retained_byte_budget = retained_byte_budget
        self._chunks: Deque[tuple[bytes, RetainedByteLease | None]] = deque()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def __bool__(self) -> bool:
        return bool(self._chunks)

    def __iter__(self):
        return (chunk for chunk, _lease in self._chunks)

    @property
    def retained_bytes(self) -> int:
        return self._bytes

    async def append(
        self,
        chunk: bytes,
        *,
        allow_over_limit: bool = False,
    ) -> None:
        chunk = bytes(chunk)
        next_items = len(self._chunks) + 1
        next_bytes = self._bytes + len(chunk)
        if not allow_over_limit and (
            next_items > self.max_items or next_bytes > self.max_bytes
        ):
            raise StreamQueueItemTooLarge(
                "stream precommit buffer limit exceeded"
            )
        lease = (
            await self.retained_byte_budget.reserve(len(chunk))
            if self.retained_byte_budget is not None
            else None
        )
        self._chunks.append((chunk, lease))
        self._bytes = next_bytes

    def popleft(self) -> tuple[bytes, RetainedByteLease | None]:
        chunk, lease = self._chunks.popleft()
        self._bytes -= len(chunk)
        return chunk, lease

    async def clear(self) -> None:
        leases: list[RetainedByteLease] = []
        while self._chunks:
            _chunk, lease = self._chunks.popleft()
            if lease is not None:
                leases.append(lease)
        self._bytes = 0
        await _release_leases_despite_cancellation(leases)


class ByteBoundedQueue:
    """A FIFO queue bounded by both item count and retained payload bytes.

    The producer waits for capacity instead of dropping data. Closing never
    enqueues a sentinel, so shutdown cannot deadlock behind a full queue.
    """

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int,
        put_timeout_seconds: float | None = None,
        retained_byte_budget: RetainedByteBudget | None = None,
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if put_timeout_seconds is not None and put_timeout_seconds <= 0:
            raise ValueError("put_timeout_seconds must be positive when set")

        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self.put_timeout_seconds = (
            float(put_timeout_seconds) if put_timeout_seconds is not None else None
        )
        self.retained_byte_budget = retained_byte_budget
        self._items: Deque[tuple[Any, int, RetainedByteLease | None]] = deque()
        self._bytes = 0
        self._inflight_items = 0
        self._inflight_bytes = 0
        self._peak_items = 0
        self._peak_bytes = 0
        self._closed = False
        self._close_error: BaseException | None = None
        self._waiting_putters = 0
        self._blocked_puts = 0
        self._put_wait_ms = 0.0
        self._put_timeouts = 0
        self._condition = asyncio.Condition()

    @staticmethod
    def item_size(item: Any) -> int:
        if isinstance(item, str):
            return len(item.encode("utf-8"))
        if isinstance(item, (bytes, bytearray, memoryview)):
            return len(item)
        body = getattr(item, "body", None)
        if isinstance(body, str):
            return len(body.encode("utf-8"))
        if isinstance(body, (bytes, bytearray, memoryview)):
            return len(body)
        return 0

    def snapshot(self) -> StreamQueueSnapshot:
        return StreamQueueSnapshot(
            items=len(self._items) + self._inflight_items,
            bytes=self._bytes,
            queued_items=len(self._items),
            queued_bytes=self._bytes - self._inflight_bytes,
            inflight_items=self._inflight_items,
            inflight_bytes=self._inflight_bytes,
            peak_items=self._peak_items,
            peak_bytes=self._peak_bytes,
            max_items=self.max_items,
            max_bytes=self.max_bytes,
            closed=self._closed,
            waiting_putters=self._waiting_putters,
            blocked_puts=self._blocked_puts,
            put_wait_ms=self._put_wait_ms,
            put_timeouts=self._put_timeouts,
        )

    def _can_put(self, size: int) -> bool:
        retained_items = len(self._items) + self._inflight_items
        return retained_items < self.max_items and self._bytes + size <= self.max_bytes

    async def put(
        self,
        item: Any,
        *,
        size: int | None = None,
        retained_byte_lease: RetainedByteLease | None = None,
    ) -> float:
        if size is not None and int(size) < 0:
            raise ValueError("item size cannot be negative")
        item_bytes = self.item_size(item) if size is None else int(size)
        if item_bytes > self.max_bytes:
            raise StreamQueueItemTooLarge(
                f"stream queue item is {item_bytes} bytes; limit is {self.max_bytes}"
            )

        started_at = monotonic()
        if retained_byte_lease is not None:
            if retained_byte_lease.size != item_bytes:
                raise ValueError("transferred retained-byte lease size mismatch")
            if self.retained_byte_budget is None:
                raise ValueError("queue has no shared retained-byte budget")
            if retained_byte_lease.budget is not self.retained_byte_budget:
                raise ValueError("transferred retained-byte lease uses another budget")
            budget_lease = retained_byte_lease
        else:
            budget_lease = (
                await self.retained_byte_budget.reserve(item_bytes)
                if self.retained_byte_budget is not None
                else None
            )

        async def wait_and_put() -> float:
            async with self._condition:
                blocked = not self._closed and not self._can_put(item_bytes)
                if blocked:
                    self._blocked_puts += 1
                    self._waiting_putters += 1
                    try:
                        await self._condition.wait_for(
                            lambda: self._closed or self._can_put(item_bytes)
                        )
                    finally:
                        self._waiting_putters = max(0, self._waiting_putters - 1)
                if self._closed:
                    raise StreamQueueClosed() from self._close_error
                self._items.append((item, item_bytes, budget_lease))
                self._bytes += item_bytes
                self._peak_items = max(
                    self._peak_items,
                    len(self._items) + self._inflight_items,
                )
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._condition.notify_all()
                waited_ms = (monotonic() - started_at) * 1000.0 if blocked else 0.0
                self._put_wait_ms += waited_ms
                return waited_ms

        try:
            if self.put_timeout_seconds is None:
                return await wait_and_put()
            try:
                async with asyncio.timeout(self.put_timeout_seconds):
                    return await wait_and_put()
            except TimeoutError as exc:
                self._put_timeouts += 1
                raise StreamQueuePutTimeout(
                    f"stream queue stayed full for {self.put_timeout_seconds:g} seconds"
                ) from exc
        except BaseException:
            if budget_lease is not None:
                await budget_lease.release()
            raise

    async def get_lease(self) -> StreamQueueItemLease:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._items) or self._closed)
            if self._items:
                item, item_bytes, budget_lease = self._items.popleft()
                self._inflight_items += 1
                self._inflight_bytes += item_bytes
                return StreamQueueItemLease(
                    self,
                    item,
                    item_bytes,
                    budget_lease,
                )
            if self._close_error is not None:
                raise self._close_error
            raise StreamQueueClosed()

    async def _release_inflight(
        self,
        item_bytes: int,
        budget_lease: RetainedByteLease | None,
    ) -> None:
        async with self._condition:
            if self._inflight_items <= 0 or item_bytes > self._inflight_bytes:
                raise RuntimeError("stream queue in-flight accounting underflow")
            self._inflight_items -= 1
            self._inflight_bytes -= item_bytes
            self._bytes -= item_bytes
            self._condition.notify_all()
        if budget_lease is not None:
            await budget_lease.release()

    async def close(
        self,
        *,
        error: BaseException | None = None,
        discard: bool = False,
    ) -> None:
        discarded: list[RetainedByteLease] = []
        async with self._condition:
            if self._closed:
                if self._close_error is None and error is not None:
                    self._close_error = error
                if discard:
                    while self._items:
                        _item, item_bytes, budget_lease = self._items.popleft()
                        self._bytes -= item_bytes
                        if budget_lease is not None:
                            discarded.append(budget_lease)
                self._condition.notify_all()
            else:
                self._closed = True
                self._close_error = error
                if discard:
                    while self._items:
                        _item, item_bytes, budget_lease = self._items.popleft()
                        self._bytes -= item_bytes
                        if budget_lease is not None:
                            discarded.append(budget_lease)
                self._condition.notify_all()
        await _release_leases_despite_cancellation(discarded)
