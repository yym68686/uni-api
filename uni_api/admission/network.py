from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from uni_api.admission.resources import (
    ephemeral_port_count,
    process_nofile_soft_limit,
    process_open_fd_count,
    tcp_socket_port_occupancy,
)


class NetworkGovernorClosed(RuntimeError):
    """The owner closed while a connection attempt was waiting for resources."""


@dataclass(frozen=True, slots=True)
class AdaptiveNetworkSnapshot:
    nofile_soft_limit: int | None
    open_fds: int | None
    fd_reserve: int | None
    fd_headroom: int | None
    ephemeral_port_count: int | None
    ephemeral_port_occupancy: int | None
    ephemeral_port_headroom: int | None
    pending_connection_attempts: int
    completed_connection_attempts_since_sample: int
    inbound_accepts_since_sample: int
    waiting_connection_attempts: int
    acquired_total: int
    blocked_total: int
    cancelled_total: int
    sample_error: str | None
    sampled_at_monotonic: float | None
    sample_age_ms: int | None


class NetworkResourceLease:
    """One provisional FD/port reservation for a connection attempt."""

    def __init__(
        self,
        governor: AdaptiveNetworkGovernor,
        *,
        wait_ms: float,
    ) -> None:
        self._governor = governor
        self.wait_ms = max(0.0, float(wait_ms))
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._released:
            return
        self._governor.release()
        self._released = True


class AdaptiveNetworkGovernor:
    """Gate connection attempts with live FD and ephemeral-port headroom.

    This governor deliberately has no request-count capacity. A short-lived
    reservation closes the race between sampling the kernel and opening a new
    socket. Once response headers establish the transport, the provisional
    reservation is released and the real socket appears in the next kernel
    sample. HTTP/2 reuse therefore does not consume one slot per request.
    """

    def __init__(
        self,
        *,
        nofile_supplier: Callable[[], int | None] = process_nofile_soft_limit,
        open_fds_supplier: Callable[[], int | None] = process_open_fd_count,
        ephemeral_ports_supplier: Callable[[], int | None] = ephemeral_port_count,
        ephemeral_occupancy_supplier: Callable[
            [], int | None
        ] = tcp_socket_port_occupancy,
        fd_reserve_min: int = 256,
        fd_reserve_ratio: float = 0.05,
        ephemeral_port_utilization: float = 0.80,
        sample_cache_seconds: float = 0.05,
        wait_poll_seconds: float = 0.10,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if fd_reserve_min < 0:
            raise ValueError("fd_reserve_min cannot be negative")
        if not 0 <= fd_reserve_ratio < 1:
            raise ValueError("fd_reserve_ratio must be below one")
        if not 0 < ephemeral_port_utilization < 1:
            raise ValueError("ephemeral_port_utilization must be between zero and one")
        if sample_cache_seconds < 0 or wait_poll_seconds <= 0:
            raise ValueError("network sampling intervals are invalid")

        self._nofile_supplier = nofile_supplier
        self._open_fds_supplier = open_fds_supplier
        self._ephemeral_ports_supplier = ephemeral_ports_supplier
        self._ephemeral_occupancy_supplier = ephemeral_occupancy_supplier
        self.fd_reserve_min = int(fd_reserve_min)
        self.fd_reserve_ratio = float(fd_reserve_ratio)
        self.ephemeral_port_utilization = float(ephemeral_port_utilization)
        self.sample_cache_seconds = float(sample_cache_seconds)
        self.wait_poll_seconds = float(wait_poll_seconds)
        self._clock = clock

        self._lock = threading.RLock()
        self._sampled_at = float("-inf")
        self._sample_error: str | None = None
        self._nofile_soft_limit: int | None = None
        self._open_fds: int | None = None
        self._ephemeral_ports: int | None = None
        self._ephemeral_occupancy: int | None = None
        self._pending_attempts = 0
        # A completed attempt may have opened a socket that is not represented
        # in the cached kernel sample yet. Preserve that provisional charge
        # until the normal refresh instead of forcing a procfs scan per request.
        self._completed_attempts_since_sample = 0
        # Accepted sockets are already kernel-owned when the protocol callback
        # runs. Count accepts inside the sampling window pessimistically so a
        # connection burst cannot spend the same cached FD headroom twice.
        self._inbound_accepts_since_sample = 0
        self._waiting_attempts = 0
        self._acquired_total = 0
        self._blocked_total = 0
        self._cancelled_total = 0
        self._waiters: set[
            tuple[asyncio.AbstractEventLoop, asyncio.Event]
        ] = set()

    def _sample_locked(self, *, force: bool = False) -> None:
        now = self._clock()
        if (
            not force
            and now - self._sampled_at < self.sample_cache_seconds
        ):
            return
        try:
            self._nofile_soft_limit = self._nofile_supplier()
            self._open_fds = self._open_fds_supplier()
            self._ephemeral_ports = self._ephemeral_ports_supplier()
            self._ephemeral_occupancy = self._ephemeral_occupancy_supplier()
            self._completed_attempts_since_sample = 0
            self._inbound_accepts_since_sample = 0
            self._sample_error = None
        except Exception as exc:
            self._sample_error = f"{type(exc).__name__}: {exc}"[:512]
        self._sampled_at = now

    def _fd_reserve_locked(self) -> int | None:
        limit = self._nofile_soft_limit
        if limit is None:
            return None
        adaptive_minimum = min(self.fd_reserve_min, max(1, limit // 2))
        return max(
            adaptive_minimum,
            math.ceil(limit * self.fd_reserve_ratio),
        )

    def _headroom_locked(self) -> tuple[int | None, int | None]:
        fd_headroom: int | None = None
        reserve = self._fd_reserve_locked()
        if (
            self._nofile_soft_limit is not None
            and self._open_fds is not None
            and reserve is not None
        ):
            fd_headroom = max(
                0,
                self._nofile_soft_limit
                - reserve
                - self._open_fds
                - self._pending_attempts
                - self._completed_attempts_since_sample
                - self._inbound_accepts_since_sample,
            )

        port_headroom: int | None = None
        if (
            self._ephemeral_ports is not None
            and self._ephemeral_occupancy is not None
        ):
            usable = math.floor(
                self._ephemeral_ports * self.ephemeral_port_utilization
            )
            port_headroom = max(
                0,
                usable
                - self._ephemeral_occupancy
                - self._pending_attempts
                - self._completed_attempts_since_sample,
            )
        return fd_headroom, port_headroom

    def _try_reserve_locked(self) -> bool:
        self._sample_locked()
        fd_headroom, port_headroom = self._headroom_locked()
        if fd_headroom is not None and fd_headroom <= 0:
            return False
        if port_headroom is not None and port_headroom <= 0:
            return False
        self._pending_attempts += 1
        self._acquired_total += 1
        return True

    def try_acquire(self) -> NetworkResourceLease | None:
        started_at = self._clock()
        with self._lock:
            if not self._try_reserve_locked():
                return None
        return NetworkResourceLease(
            self,
            wait_ms=(self._clock() - started_at) * 1000.0,
        )

    async def acquire(
        self,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> NetworkResourceLease:
        started_at = self._clock()
        immediate = self.try_acquire()
        if immediate is not None:
            immediate.wait_ms = max(
                immediate.wait_ms,
                (self._clock() - started_at) * 1000.0,
            )
            return immediate

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        waiter = (loop, event)
        with self._lock:
            self._blocked_total += 1
            self._waiting_attempts += 1
            self._waiters.add(waiter)
        try:
            while True:
                if abort is not None and abort():
                    raise NetworkGovernorClosed
                event.clear()
                with self._lock:
                    if self._try_reserve_locked():
                        return NetworkResourceLease(
                            self,
                            wait_ms=(self._clock() - started_at) * 1000.0,
                        )
                try:
                    async with asyncio.timeout(self.wait_poll_seconds):
                        await event.wait()
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            with self._lock:
                self._cancelled_total += 1
            raise
        finally:
            with self._lock:
                self._waiters.discard(waiter)
                self._waiting_attempts = max(0, self._waiting_attempts - 1)

    def release(self) -> None:
        with self._lock:
            if self._pending_attempts <= 0:
                raise RuntimeError("network connection-attempt reservation underflow")
            self._pending_attempts -= 1
            # Keep charging a possibly new socket until the regular cached
            # sample observes it. Reuse is deliberately overcharged for at
            # most one short sample window; no request triggers a forced scan.
            self._completed_attempts_since_sample += 1
            waiters = tuple(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                continue

    def allow_inbound_connection(self) -> bool:
        """Return whether one already-accepted socket preserves FD reserve."""

        with self._lock:
            self._sample_locked()
            fd_headroom, _port_headroom = self._headroom_locked()
            if fd_headroom is not None and fd_headroom <= 0:
                return False
            self._inbound_accepts_since_sample += 1
            return True

    def snapshot(self, *, force: bool = False) -> AdaptiveNetworkSnapshot:
        with self._lock:
            self._sample_locked(force=force)
            fd_headroom, port_headroom = self._headroom_locked()
            sampled_at = (
                self._sampled_at
                if self._sampled_at != float("-inf")
                else None
            )
            return AdaptiveNetworkSnapshot(
                nofile_soft_limit=self._nofile_soft_limit,
                open_fds=self._open_fds,
                fd_reserve=self._fd_reserve_locked(),
                fd_headroom=fd_headroom,
                ephemeral_port_count=self._ephemeral_ports,
                ephemeral_port_occupancy=self._ephemeral_occupancy,
                ephemeral_port_headroom=port_headroom,
                pending_connection_attempts=self._pending_attempts,
                completed_connection_attempts_since_sample=(
                    self._completed_attempts_since_sample
                ),
                inbound_accepts_since_sample=self._inbound_accepts_since_sample,
                waiting_connection_attempts=self._waiting_attempts,
                acquired_total=self._acquired_total,
                blocked_total=self._blocked_total,
                cancelled_total=self._cancelled_total,
                sample_error=self._sample_error,
                sampled_at_monotonic=sampled_at,
                sample_age_ms=(
                    max(
                        0,
                        int(round((self._clock() - sampled_at) * 1000.0)),
                    )
                    if sampled_at is not None
                    else None
                ),
            )
