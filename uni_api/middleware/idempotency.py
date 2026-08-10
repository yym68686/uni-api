from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass, field
from time import monotonic, perf_counter_ns, thread_time_ns
from typing import Any, Callable, Literal

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from uni_api.admission.memory import AdaptiveMemoryGovernor
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.idempotency_spool import (
    IdempotencySpool,
    IdempotencySpoolError,
    create_private_spool_directory,
)


IDEMPOTENCY_KEY_HEADER = b"idempotency-key"
IDEMPOTENCY_STATUS_HEADER = b"x-uni-api-idempotency-status"
IDEMPOTENCY_KEY_FINGERPRINT_STATE_KEY = (
    "uni_api_idempotency_key_fingerprint"
)
IDEMPOTENCY_ROLE_STATE_KEY = "uni_api_idempotency_role"

DEFAULT_IDEMPOTENT_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/responses/compact",
    }
)

_SAFE_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_REPLAYABLE_STATUS_MAX = 499
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 499})
_HOP_BY_HOP_HEADERS = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)


def _retryable_incomplete_stream(scope: Scope) -> bool:
    """Return whether the executed response ended after a committed stream abort.

    A transport-truncated Responses stream deliberately ends without a semantic
    terminal so compatible clients can retry it.  Treating the resulting ASGI
    body EOF as a completed idempotent response would replay the same partial
    stream forever and defeat that recovery contract.
    """

    state = scope.get("state")
    if not isinstance(state, dict):
        return False
    current_info = state.get("uni_api_request_info")
    if not isinstance(current_info, dict):
        return False
    return (
        current_info.get("success") is False
        and current_info.get("stream_outcome") == "upstream_stream_abort"
        and current_info.get("stream_error_after_response_start") is True
        and current_info.get("postcommit_stream_terminal_suppressed") is True
    )


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass(frozen=True, slots=True)
class CachedASGIResponse:
    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: IdempotencySpool

    @property
    def body_bytes(self) -> int:
        return self.body.size


@dataclass(slots=True)
class _IdempotencyEntry:
    request_hash: str
    owner_token: str
    created_at: float
    event: asyncio.Event = field(default_factory=asyncio.Event)
    response: CachedASGIResponse | None = None
    nonreplayable_reason: str | None = None
    completed_at: float | None = None
    expires_at: float | None = None
    replay_readers: int = 0

    @property
    def complete(self) -> bool:
        return self.response is not None or self.nonreplayable_reason is not None


@dataclass(frozen=True, slots=True)
class IdempotencyClaim:
    kind: Literal[
        "owner",
        "wait",
        "replay",
        "conflict",
        "unavailable",
        "nonreplayable",
    ]
    entry: _IdempotencyEntry | None = None
    owner_token: str | None = None
    response: CachedASGIResponse | None = None
    reason: str | None = None


class InMemoryIdempotencyCoordinator:
    """Bounded, process-scoped idempotency coordination.

    The production uni-api deployment currently has one process and database
    persistence disabled.  This coordinator therefore makes the actual scope
    explicit instead of pretending to provide cross-process guarantees.  A
    later persistent implementation can preserve the middleware contract.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 15 * 60,
        max_entries: int = 4096,
        max_stored_bytes: int = 256 * 1024 * 1024,
        max_response_bytes: int = 16 * 1024 * 1024,
        spool_memory_threshold_bytes: int = 256 * 1024,
        spool_parent_directory: str | None = None,
        memory_governor: AdaptiveMemoryGovernor | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_stored_bytes <= 0:
            raise ValueError("max_stored_bytes must be positive")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if spool_memory_threshold_bytes < 0:
            raise ValueError("spool_memory_threshold_bytes cannot be negative")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.max_stored_bytes = int(max_stored_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.spool_memory_threshold_bytes = int(spool_memory_threshold_bytes)
        self._memory_governor = memory_governor
        self._spool_directory = create_private_spool_directory(
            spool_parent_directory
        )
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._stored_bytes = 0
        self._spool_bytes = 0
        self._spool_peak_bytes = 0
        self._spool_memory_bytes = 0
        self._spool_memory_peak_bytes = 0
        self._spools: dict[int, list[Any]] = {}
        self._spool_bytes_by_kind: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._counters: dict[str, int] = {
            "owners": 0,
            "waits": 0,
            "replays": 0,
            "conflicts": 0,
            "failures_released": 0,
            "responses_not_cached": 0,
            "capacity_rejections": 0,
            "downstream_disconnects_detached": 0,
            "nonreplayable_completions": 0,
            "nonreplayable_rejections": 0,
            "spool_capacity_rejections": 0,
            "spool_write_failures": 0,
            "completed_evictions": 0,
        }

    def create_spool(self, kind: str) -> IdempotencySpool:
        normalized = str(kind or "unknown").strip() or "unknown"
        memory_threshold = self.spool_memory_threshold_bytes
        if normalized in {"inflight_response", "completed_response"}:
            # Every replayable response eventually becomes a completed disk
            # entry. Writing it there directly avoids copying the warm prefix
            # from bytearray to disk at stream completion.
            memory_threshold = 0
        spool = IdempotencySpool(
            directory=self._spool_directory.name,
            memory_threshold_bytes=memory_threshold,
            memory_governor=self._memory_governor,
            memory_category=f"idempotency_{normalized}",
        )
        self._spools[id(spool)] = [spool, normalized, 0, 0]
        return spool

    async def append_spool(
        self,
        spool: IdempotencySpool,
        payload: bytes | bytearray | memoryview,
    ) -> bool:
        chunk = bytes(payload)
        if not chunk:
            return True
        accounting = self._spools.get(id(spool))
        if accounting is None or accounting[0] is not spool:
            raise RuntimeError("idempotency spool is not owned by this coordinator")
        async with self._lock:
            # Expiry cleanup is needed for claims and only when quota pressure
            # exists. Scanning every cached entry for every stream chunk made
            # this O(chunks * entries) on the incident hot path.
            if self._spool_bytes + len(chunk) > self.max_stored_bytes:
                self._prune_expired_locked(monotonic())
            if not self._make_spool_room_locked(len(chunk)):
                self._counters["spool_capacity_rejections"] += 1
                return False
            self._account_spool_growth_locked(accounting, len(chunk))
        try:
            await spool.append_async(chunk)
        except IdempotencySpoolError:
            async with self._lock:
                self._discard_spool_locked(spool)
                self._counters["spool_write_failures"] += 1
            return False
        memory_before = int(accounting[3])
        memory_after = spool.memory_bytes
        accounting[3] = memory_after
        self._spool_memory_bytes = max(
            0,
            self._spool_memory_bytes + memory_after - memory_before,
        )
        self._spool_memory_peak_bytes = max(
            self._spool_memory_peak_bytes,
            self._spool_memory_bytes,
        )
        return True

    async def close_spool(self, spool: IdempotencySpool | None) -> None:
        if spool is None:
            return
        async with self._lock:
            accounting = self._spools.pop(id(spool), None)
            if accounting is not None and accounting[0] is spool:
                self._account_spool_shrink_locked(accounting, int(accounting[2]))
                self._spool_memory_bytes = max(
                    0,
                    self._spool_memory_bytes - int(accounting[3]),
                )
        await spool.close_async()

    async def claim(self, record_key: str, request_hash: str) -> IdempotencyClaim:
        now = monotonic()
        async with self._lock:
            self._prune_expired_locked(now)
            entry = self._entries.get(record_key)
            if entry is not None:
                if entry.request_hash != request_hash:
                    self._counters["conflicts"] += 1
                    return IdempotencyClaim("conflict")
                if entry.response is not None:
                    entry.replay_readers += 1
                    self._counters["replays"] += 1
                    return IdempotencyClaim(
                        "replay",
                        entry=entry,
                        response=entry.response,
                    )
                if entry.nonreplayable_reason is not None:
                    self._counters["nonreplayable_rejections"] += 1
                    return IdempotencyClaim(
                        "nonreplayable",
                        reason=entry.nonreplayable_reason,
                    )
                self._counters["waits"] += 1
                return IdempotencyClaim("wait", entry=entry)

            if not self._make_entry_room_locked():
                self._counters["capacity_rejections"] += 1
                return IdempotencyClaim("unavailable")

            owner_token = hashlib.sha256(
                f"{record_key}:{request_hash}:{now}".encode("ascii")
            ).hexdigest()
            entry = _IdempotencyEntry(
                request_hash=request_hash,
                owner_token=owner_token,
                created_at=now,
            )
            self._entries[record_key] = entry
            self._counters["owners"] += 1
            return IdempotencyClaim(
                "owner",
                entry=entry,
                owner_token=owner_token,
            )

    async def complete(
        self,
        record_key: str,
        owner_token: str,
        response: CachedASGIResponse,
    ) -> bool:
        now = monotonic()
        body_bytes = response.body_bytes
        async with self._lock:
            entry = self._entries.get(record_key)
            if entry is None or entry.owner_token != owner_token:
                return False
            if body_bytes > self.max_response_bytes:
                self._release_locked(record_key, entry, not_cached=True)
                return False
            self._prune_expired_locked(now)
            accounting = self._spools.get(id(response.body))
            if accounting is None or accounting[0] is not response.body:
                self._release_locked(record_key, entry, not_cached=True)
                return False
            memory_before = int(accounting[3])
            memory_after = response.body.memory_bytes
            accounting[3] = memory_after
            self._spool_memory_bytes = max(
                0,
                self._spool_memory_bytes + memory_after - memory_before,
            )
            entry.response = response
            entry.completed_at = now
            entry.expires_at = now + self.ttl_seconds
            self._stored_bytes += body_bytes
            self._move_spool_kind_locked(accounting, "completed_response")
            entry.event.set()
            return True

    async def complete_nonreplayable(
        self,
        record_key: str,
        owner_token: str,
        *,
        reason: str,
    ) -> bool:
        now = monotonic()
        async with self._lock:
            entry = self._entries.get(record_key)
            if entry is None or entry.owner_token != owner_token:
                return False
            self._prune_expired_locked(now)
            entry.nonreplayable_reason = str(reason or "capture_unavailable")[:120]
            entry.completed_at = now
            entry.expires_at = now + self.ttl_seconds
            self._counters["nonreplayable_completions"] += 1
            self._counters["responses_not_cached"] += 1
            entry.event.set()
            return True

    async def finish_replay(self, entry: _IdempotencyEntry) -> None:
        async with self._lock:
            if entry.replay_readers <= 0:
                raise RuntimeError("idempotency replay reader underflow")
            entry.replay_readers -= 1

    async def release_failure(
        self,
        record_key: str,
        owner_token: str,
        *,
        not_cached: bool = False,
    ) -> bool:
        async with self._lock:
            entry = self._entries.get(record_key)
            if entry is None or entry.owner_token != owner_token:
                return False
            self._release_locked(record_key, entry, not_cached=not_cached)
            return True

    def note_detached_disconnect(self) -> None:
        self._counters["downstream_disconnects_detached"] += 1

    def snapshot(self) -> dict[str, Any]:
        entries = tuple(self._entries.values())
        completed = sum(1 for entry in entries if entry.complete)
        replayable = sum(1 for entry in entries if entry.response is not None)
        nonreplayable = sum(
            1 for entry in entries if entry.nonreplayable_reason is not None
        )
        return {
            "enabled": True,
            "mode": "spool-single-process",
            "persistence": False,
            "entries": len(entries),
            "in_progress": len(entries) - completed,
            "completed": completed,
            "replayable_completed": replayable,
            "nonreplayable_completed": nonreplayable,
            "stored_response_bytes": self._stored_bytes,
            "max_entries": self.max_entries,
            "max_stored_response_bytes": self.max_stored_bytes,
            "max_response_bytes": self.max_response_bytes,
            "spool_total_bytes": self._spool_bytes,
            "spool_peak_bytes": self._spool_peak_bytes,
            "spool_memory_bytes": self._spool_memory_bytes,
            "spool_memory_peak_bytes": self._spool_memory_peak_bytes,
            "spool_bytes_by_kind": dict(self._spool_bytes_by_kind),
            "spool_memory_threshold_bytes": self.spool_memory_threshold_bytes,
            "ttl_seconds": self.ttl_seconds,
            **self._counters,
        }

    def _release_locked(
        self,
        record_key: str,
        entry: _IdempotencyEntry,
        *,
        not_cached: bool,
    ) -> None:
        if self._entries.get(record_key) is entry:
            self._entries.pop(record_key, None)
        if entry.response is not None:
            self._stored_bytes = max(
                0,
                self._stored_bytes - entry.response.body_bytes,
            )
            self._discard_spool_locked(entry.response.body)
        self._counters["failures_released"] += 1
        if not_cached:
            self._counters["responses_not_cached"] += 1
        entry.event.set()

    def _prune_expired_locked(self, now: float) -> None:
        expired = [
            (record_key, entry)
            for record_key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
            and entry.replay_readers == 0
        ]
        for record_key, entry in expired:
            self._entries.pop(record_key, None)
            if entry.response is not None:
                self._stored_bytes = max(
                    0,
                    self._stored_bytes - entry.response.body_bytes,
                )
                self._discard_spool_locked(entry.response.body)

    def _make_entry_room_locked(self) -> bool:
        if len(self._entries) < self.max_entries:
            return True
        completed = sorted(
            (
                (entry.completed_at or entry.created_at, record_key, entry)
                for record_key, entry in self._entries.items()
                if entry.complete and entry.replay_readers == 0
            ),
            key=lambda item: item[0],
        )
        while len(self._entries) >= self.max_entries and completed:
            _completed_at, record_key, entry = completed.pop(0)
            self._entries.pop(record_key, None)
            self._discard_completed_entry_locked(entry)
            self._counters["completed_evictions"] += 1
        return len(self._entries) < self.max_entries

    def _make_spool_room_locked(self, required_bytes: int) -> bool:
        if required_bytes > self.max_stored_bytes:
            return False
        if self._spool_bytes + required_bytes <= self.max_stored_bytes:
            return True
        completed = sorted(
            (
                (entry.completed_at or entry.created_at, record_key, entry)
                for record_key, entry in self._entries.items()
                if entry.complete and entry.replay_readers == 0
            ),
            key=lambda item: item[0],
        )
        while (
            self._spool_bytes + required_bytes > self.max_stored_bytes
            and completed
        ):
            _completed_at, record_key, entry = completed.pop(0)
            self._entries.pop(record_key, None)
            self._discard_completed_entry_locked(entry)
            self._counters["completed_evictions"] += 1
        return self._spool_bytes + required_bytes <= self.max_stored_bytes

    def _discard_completed_entry_locked(self, entry: _IdempotencyEntry) -> None:
        if entry.response is None:
            return
        self._stored_bytes = max(0, self._stored_bytes - entry.response.body_bytes)
        self._discard_spool_locked(entry.response.body)

    def _account_spool_growth_locked(
        self,
        accounting: list[Any],
        size: int,
    ) -> None:
        kind = str(accounting[1])
        accounting[2] = int(accounting[2]) + size
        self._spool_bytes += size
        self._spool_peak_bytes = max(self._spool_peak_bytes, self._spool_bytes)
        self._spool_bytes_by_kind[kind] = (
            self._spool_bytes_by_kind.get(kind, 0) + size
        )

    def _account_spool_shrink_locked(
        self,
        accounting: list[Any],
        size: int,
    ) -> None:
        kind = str(accounting[1])
        current = int(accounting[2])
        released = min(current, max(0, int(size)))
        accounting[2] = current - released
        self._spool_bytes = max(0, self._spool_bytes - released)
        remaining = max(0, self._spool_bytes_by_kind.get(kind, 0) - released)
        if remaining:
            self._spool_bytes_by_kind[kind] = remaining
        else:
            self._spool_bytes_by_kind.pop(kind, None)

    def _move_spool_kind_locked(
        self,
        accounting: list[Any],
        kind: str,
    ) -> None:
        old_kind = str(accounting[1])
        new_kind = str(kind or "unknown")
        if old_kind == new_kind:
            return
        size = int(accounting[2])
        old_remaining = max(
            0,
            self._spool_bytes_by_kind.get(old_kind, 0) - size,
        )
        if old_remaining:
            self._spool_bytes_by_kind[old_kind] = old_remaining
        else:
            self._spool_bytes_by_kind.pop(old_kind, None)
        self._spool_bytes_by_kind[new_kind] = (
            self._spool_bytes_by_kind.get(new_kind, 0) + size
        )
        accounting[1] = new_kind

    def _discard_spool_locked(self, spool: IdempotencySpool) -> None:
        accounting = self._spools.pop(id(spool), None)
        if accounting is not None and accounting[0] is spool:
            self._account_spool_shrink_locked(accounting, int(accounting[2]))
            self._spool_memory_bytes = max(
                0,
                self._spool_memory_bytes - int(accounting[3]),
            )
        spool.close()

    def __del__(self) -> None:
        try:
            for accounting in tuple(self._spools.values()):
                accounting[0].close()
            self._spools.clear()
            self._spool_directory.cleanup()
        except Exception:
            pass


class IdempotencyMiddleware:
    """Opt-in ASGI response coalescing and replay for logical API requests."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        coordinator: InMemoryIdempotencyCoordinator,
        enabled: bool = True,
        paths: frozenset[str] = DEFAULT_IDEMPOTENT_PATHS,
        max_request_body_bytes: int = 64 * 1024 * 1024,
        request_body_idle_timeout_seconds: float = 15.0,
        request_body_total_timeout_seconds: float = 120.0,
        wait_timeout_seconds: float = 30 * 60,
        observer: Callable[[str, dict[str, Any]], Any] | None = None,
        phase_sample_decider: Callable[[], bool] | None = None,
        phase_observer: Callable[..., Any] | None = None,
    ) -> None:
        self.app = app
        self.coordinator = coordinator
        self.enabled = bool(enabled)
        self.paths = frozenset(paths)
        self.max_request_body_bytes = int(max_request_body_bytes)
        self.request_body_idle_timeout_seconds = float(
            request_body_idle_timeout_seconds
        )
        self.request_body_total_timeout_seconds = float(
            request_body_total_timeout_seconds
        )
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.observer = observer
        self.phase_sample_decider = phase_sample_decider
        self.phase_observer = phase_observer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        raw_keys = [
            value
            for name, value in (scope.get("headers") or [])
            if name.lower() == IDEMPOTENCY_KEY_HEADER
        ]
        if not raw_keys:
            await self.app(scope, receive, send)
            return
        if len(raw_keys) != 1:
            await self._json_error(
                scope,
                receive,
                send,
                400,
                "multiple Idempotency-Key headers are not allowed",
                status="invalid-key",
            )
            return
        try:
            idempotency_key = raw_keys[0].decode("ascii")
        except UnicodeDecodeError:
            idempotency_key = ""
        if _SAFE_IDEMPOTENCY_KEY.fullmatch(idempotency_key) is None:
            await self._json_error(
                scope,
                receive,
                send,
                400,
                "Idempotency-Key must contain 1-128 safe ASCII characters",
                status="invalid-key",
            )
            return

        sample_hash = False
        if self.phase_sample_decider is not None:
            try:
                sample_hash = bool(self.phase_sample_decider())
            except Exception:
                sample_hash = False
        record_key, key_fingerprint, request_digest = _begin_request_identity(
            scope,
            idempotency_key,
        )
        body_spool: IdempotencySpool | None = None
        try:
            body_spool, body_bytes, hash_wall_ns, hash_cpu_ns = (
                await self._read_body(
                    receive,
                    request_digest=request_digest,
                    sample_hash=sample_hash,
                )
            )
        except _RequestBodyDisconnected:
            return
        except _RequestBodyTooLarge:
            await self._json_error(
                scope,
                receive,
                send,
                413,
                "request body too large",
                status="request-too-large",
            )
            return
        except _RequestBodyTimedOut:
            await self._json_error(
                scope,
                receive,
                send,
                408,
                "request body upload timed out",
                status="request-timeout",
            )
            return
        except _IdempotencySpoolCapacity:
            await self._json_error(
                scope,
                receive,
                send,
                503,
                "idempotency request spool capacity exhausted",
                status="capacity-exhausted",
                retry_after=True,
            )
            return

        request_hash = request_digest.hexdigest()
        if sample_hash and self.phase_observer is not None:
            try:
                self.phase_observer(
                    "idempotency_hash",
                    wall_ns=hash_wall_ns,
                    cpu_ns=hash_cpu_ns,
                    bytes_count=body_bytes,
                    events=1,
                )
            except Exception:
                pass
        try:
            while True:
                claim = await self.coordinator.claim(record_key, request_hash)
                self._observe(
                    claim.kind,
                    {
                        "key_fingerprint": key_fingerprint,
                        "method": str(scope.get("method") or ""),
                        "path": str(scope.get("path") or ""),
                    },
                )
                if claim.kind == "conflict":
                    await self._json_error(
                        scope,
                        receive,
                        send,
                        409,
                        "Idempotency-Key was already used for a different request",
                        status="conflict",
                        request_body_consumed=True,
                    )
                    return
                if claim.kind == "unavailable":
                    await self._json_error(
                        scope,
                        receive,
                        send,
                        503,
                        "idempotency coordinator capacity exhausted",
                        status="capacity-exhausted",
                        retry_after=True,
                        request_body_consumed=True,
                    )
                    return
                if claim.kind == "nonreplayable":
                    await self._json_error(
                        scope,
                        receive,
                        send,
                        409,
                        "the original request executed, but its response is no longer replayable",
                        status="executed-nonreplayable",
                        request_body_consumed=True,
                    )
                    return
                if claim.kind == "replay":
                    assert claim.response is not None
                    assert claim.entry is not None
                    try:
                        await _replay_response(claim.response, send)
                    finally:
                        await self.coordinator.finish_replay(claim.entry)
                    return
                if claim.kind == "wait":
                    assert claim.entry is not None
                    wait_result = await self._wait_for_owner(
                        scope,
                        claim.entry,
                    )
                    if wait_result == "disconnected":
                        return
                    if wait_result == "timeout":
                        await self._json_error(
                            scope,
                            receive,
                            send,
                            503,
                            "timed out waiting for the original idempotent request",
                            status="wait-timeout",
                            retry_after=True,
                            request_body_consumed=True,
                        )
                        return
                    continue

                assert claim.kind == "owner"
                assert claim.owner_token is not None
                assert body_spool is not None
                owner_body = body_spool
                body_spool = None
                await self._execute_owner(
                    scope,
                    receive,
                    send,
                    body=owner_body,
                    record_key=record_key,
                    owner_token=claim.owner_token,
                    key_fingerprint=key_fingerprint,
                )
                return
        finally:
            await self.coordinator.close_spool(body_spool)

    def _applies(self, scope: Scope) -> bool:
        return (
            self.enabled
            and scope.get("type") == "http"
            and str(scope.get("method") or "").upper() == "POST"
            and str(scope.get("path") or "") in self.paths
        )

    async def _read_body(
        self,
        receive: Receive,
        *,
        request_digest: Any,
        sample_hash: bool,
    ) -> tuple[IdempotencySpool, int, int, int]:
        spool = self.coordinator.create_spool("request_body")
        total = 0
        hash_wall_ns = 0
        hash_cpu_ns = 0
        deadline = monotonic() + self.request_body_total_timeout_seconds
        more_body = True
        try:
            while more_body:
                timeout = min(
                    self.request_body_idle_timeout_seconds,
                    deadline - monotonic(),
                )
                if timeout <= 0:
                    raise _RequestBodyTimedOut()
                try:
                    message = await asyncio.wait_for(receive(), timeout=timeout)
                except TimeoutError as exc:
                    raise _RequestBodyTimedOut() from exc
                if message.get("type") == "http.disconnect":
                    raise _RequestBodyDisconnected()
                if message.get("type") != "http.request":
                    continue
                chunk = bytes(message.get("body", b"") or b"")
                total += len(chunk)
                if total > self.max_request_body_bytes:
                    raise _RequestBodyTooLarge()
                if chunk:
                    if not await self.coordinator.append_spool(spool, chunk):
                        raise _IdempotencySpoolCapacity()
                    if sample_hash:
                        wall_started_ns = perf_counter_ns()
                        cpu_started_ns = thread_time_ns()
                    request_digest.update(chunk)
                    if sample_hash:
                        hash_wall_ns += max(
                            0,
                            perf_counter_ns() - wall_started_ns,
                        )
                        hash_cpu_ns += max(0, thread_time_ns() - cpu_started_ns)
                more_body = bool(message.get("more_body", False))
            await spool.seal_async(force_disk=False)
            return spool, total, hash_wall_ns, hash_cpu_ns
        except BaseException:
            await self.coordinator.close_spool(spool)
            raise

    async def _wait_for_owner(
        self,
        scope: Scope,
        entry: _IdempotencyEntry,
    ) -> Literal["ready", "disconnected", "timeout"]:
        event_task = asyncio.create_task(entry.event.wait())
        disconnect_event = _scope_disconnect_event(scope)
        disconnect_task = (
            asyncio.create_task(disconnect_event.wait())
            if disconnect_event is not None
            else None
        )
        tasks = {event_task}
        if disconnect_task is not None:
            tasks.add(disconnect_task)
        try:
            done, _pending = await asyncio.wait(
                tasks,
                timeout=self.wait_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return "timeout"
            if disconnect_task is not None and disconnect_task in done:
                return "disconnected"
            return "ready"
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_owner(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        body: IdempotencySpool,
        record_key: str,
        owner_token: str,
        key_fingerprint: str,
    ) -> None:
        original_disconnect_event = _scope_disconnect_event(scope)
        detached_scope = dict(scope)
        detached_state = dict(scope.get("state") or {})
        detached_state[DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY] = asyncio.Event()
        detached_state[IDEMPOTENCY_KEY_FINGERPRINT_STATE_KEY] = key_fingerprint
        detached_state[IDEMPOTENCY_ROLE_STATE_KEY] = "owner"
        detached_scope["state"] = detached_state
        detached_scope["headers"] = [
            (name, value)
            for name, value in (scope.get("headers") or [])
            if name.lower() != IDEMPOTENCY_KEY_HEADER
        ]

        body_spool: IdempotencySpool | None = body
        body_iterator = iter(body.iter_chunks())
        no_chunk = object()
        pending_body: bytes | object = next(body_iterator, no_chunk)
        empty_body_sent = False
        never_disconnect = asyncio.Event()

        async def detached_receive() -> Message:
            nonlocal body_spool
            nonlocal empty_body_sent
            nonlocal pending_body
            if pending_body is not no_chunk:
                current = pending_body
                pending_body = next(body_iterator, no_chunk)
                more_body = pending_body is not no_chunk
                if not more_body:
                    await self.coordinator.close_spool(body_spool)
                    body_spool = None
                return {
                    "type": "http.request",
                    "body": current,
                    "more_body": more_body,
                }
            if not empty_body_sent:
                empty_body_sent = True
                await self.coordinator.close_spool(body_spool)
                body_spool = None
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            await never_disconnect.wait()
            return {"type": "http.disconnect"}

        status_code: int | None = None
        response_headers: tuple[tuple[bytes, bytes], ...] = ()
        response_spool: IdempotencySpool | None = self.coordinator.create_spool(
            "inflight_response"
        )
        response_complete = False
        capture_enabled = True
        capture_failure_reason: str | None = None
        client_send_enabled = True

        async def capture_send(message: Message) -> None:
            nonlocal status_code
            nonlocal response_headers
            nonlocal response_complete
            nonlocal response_spool
            nonlocal capture_enabled
            nonlocal capture_failure_reason
            nonlocal client_send_enabled

            outgoing = message
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
                original_headers = tuple(message.get("headers") or ())
                response_headers = _cacheable_headers(original_headers)
                outgoing = dict(message)
                outgoing["headers"] = _with_idempotency_status(
                    original_headers,
                    b"executed",
                )
            elif message.get("type") == "http.response.body":
                chunk = bytes(message.get("body", b"") or b"")
                if capture_enabled and chunk:
                    assert response_spool is not None
                    if (
                        response_spool.size + len(chunk)
                        > self.coordinator.max_response_bytes
                    ):
                        capture_enabled = False
                        capture_failure_reason = "response_too_large"
                    elif not await self.coordinator.append_spool(
                        response_spool,
                        chunk,
                    ):
                        capture_enabled = False
                        capture_failure_reason = "spool_capacity_exhausted"
                    if not capture_enabled:
                        await self.coordinator.close_spool(response_spool)
                        response_spool = None
                if not bool(message.get("more_body", False)):
                    response_complete = True
            elif message.get("type") == "http.response.trailers":
                capture_enabled = False
                capture_failure_reason = "response_trailers_not_replayable"
                await self.coordinator.close_spool(response_spool)
                response_spool = None

            if not client_send_enabled:
                return
            try:
                await send(outgoing)
            except (BrokenPipeError, ConnectionError, OSError):
                client_send_enabled = False
            except RuntimeError:
                if (
                    original_disconnect_event is not None
                    and original_disconnect_event.is_set()
                ):
                    client_send_enabled = False
                    return
                raise

        try:
            await self.app(detached_scope, detached_receive, capture_send)
        except BaseException:
            await self.coordinator.close_spool(response_spool)
            await self.coordinator.release_failure(record_key, owner_token)
            raise
        finally:
            close_iterator = getattr(body_iterator, "close", None)
            if close_iterator is not None:
                close_iterator()
            await self.coordinator.close_spool(body_spool)
            if (
                original_disconnect_event is not None
                and original_disconnect_event.is_set()
            ):
                self.coordinator.note_detached_disconnect()

        if _retryable_incomplete_stream(detached_scope):
            await self.coordinator.close_spool(response_spool)
            await self.coordinator.release_failure(
                record_key,
                owner_token,
                not_cached=True,
            )
            return

        replayable_status = (
            response_complete
            and status_code is not None
            and status_code <= _REPLAYABLE_STATUS_MAX
            and status_code not in _TRANSIENT_STATUS_CODES
        )
        if replayable_status and not capture_enabled:
            await self.coordinator.complete_nonreplayable(
                record_key,
                owner_token,
                reason=capture_failure_reason or "capture_unavailable",
            )
            return
        if not replayable_status or response_spool is None:
            await self.coordinator.close_spool(response_spool)
            await self.coordinator.release_failure(
                record_key,
                owner_token,
                not_cached=True,
            )
            return
        try:
            await response_spool.seal_async(
                force_disk=response_spool.size > 0,
            )
        except IdempotencySpoolError:
            await self.coordinator.close_spool(response_spool)
            await self.coordinator.complete_nonreplayable(
                record_key,
                owner_token,
                reason="spool_seal_failed",
            )
            return
        cached = CachedASGIResponse(
            status_code=status_code,
            headers=response_headers,
            body=response_spool,
        )
        if not await self.coordinator.complete(record_key, owner_token, cached):
            await self.coordinator.close_spool(response_spool)

    async def _json_error(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
        *,
        status: str,
        retry_after: bool = False,
        request_body_consumed: bool = False,
    ) -> None:
        headers = {"x-uni-api-idempotency-status": status}
        if retry_after:
            headers["retry-after"] = "1"
        if (
            not request_body_consumed
            and str(scope.get("http_version") or "") in {"1.0", "1.1"}
        ):
            headers["connection"] = "close"
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": detail,
                    "type": "idempotency_error",
                    "code": status.replace("-", "_"),
                }
            },
            headers=headers,
        )
        await response(scope, receive, send)

    def _observe(self, event: str, fields: dict[str, Any]) -> None:
        if self.observer is None:
            return
        result = self.observer(event, fields)
        if hasattr(result, "__await__"):
            asyncio.create_task(result)


def build_default_idempotency_coordinator(
    *,
    memory_governor: AdaptiveMemoryGovernor | None = None,
) -> InMemoryIdempotencyCoordinator:
    return InMemoryIdempotencyCoordinator(
        ttl_seconds=_positive_float_env("IDEMPOTENCY_TTL_SECONDS", 15 * 60),
        max_entries=_positive_int_env("IDEMPOTENCY_MAX_ENTRIES", 4096),
        max_stored_bytes=_positive_int_env(
            "IDEMPOTENCY_MAX_STORED_BYTES",
            128 * 1024 * 1024,
        ),
        max_response_bytes=_positive_int_env(
            "IDEMPOTENCY_MAX_RESPONSE_BYTES",
            16 * 1024 * 1024,
        ),
        spool_memory_threshold_bytes=_positive_int_env(
            "IDEMPOTENCY_SPOOL_MEMORY_THRESHOLD_BYTES",
            256 * 1024,
        ),
        spool_parent_directory=(
            os.getenv("IDEMPOTENCY_SPOOL_DIRECTORY", "").strip() or None
        ),
        memory_governor=memory_governor,
    )


def _begin_request_identity(
    scope: Scope,
    idempotency_key: str,
) -> tuple[str, str, Any]:
    headers = _header_values(scope)
    credential = "\n".join(
        headers.get(name, "")
        for name in ("authorization", "x-api-key")
    )
    credential_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    method = str(scope.get("method") or "").upper()
    path = str(scope.get("path") or "")
    query = bytes(scope.get("query_string") or b"")
    method_bytes = method.encode("ascii", errors="replace")
    path_bytes = path.encode("utf-8")
    idempotency_key_bytes = idempotency_key.encode("ascii")
    key_scope = b"\x00".join(
        (
            method_bytes,
            path_bytes,
            query,
            credential_hash.encode("ascii"),
            idempotency_key_bytes,
        )
    )
    record_key = hashlib.sha256(key_scope).hexdigest()

    request_digest = hashlib.sha256()
    request_digest.update(method_bytes)
    for identity_part in (
        path_bytes,
        query,
        headers.get("content-type", "").encode("latin-1"),
        headers.get("content-encoding", "").encode("latin-1"),
    ):
        request_digest.update(b"\x00")
        request_digest.update(identity_part)
    # Preserve the legacy NUL separator before the body while allowing body
    # chunks to be hashed directly as they arrive.
    request_digest.update(b"\x00")
    key_fingerprint = hashlib.sha256(idempotency_key_bytes).hexdigest()[:16]
    return record_key, key_fingerprint, request_digest


def _request_identities(
    scope: Scope,
    idempotency_key: str,
    body: bytes,
) -> tuple[str, str, str]:
    record_key, key_fingerprint, request_digest = _begin_request_identity(
        scope,
        idempotency_key,
    )
    request_digest.update(body)
    return record_key, request_digest.hexdigest(), key_fingerprint


def _header_values(scope: Scope) -> dict[str, str]:
    values: dict[str, list[str]] = {}
    for name, value in (scope.get("headers") or []):
        decoded_name = name.decode("latin-1").lower()
        values.setdefault(decoded_name, []).append(value.decode("latin-1"))
    return {name: "\n".join(items) for name, items in values.items()}


def _scope_disconnect_event(scope: Scope) -> asyncio.Event | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    event = state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
    return event if isinstance(event, asyncio.Event) else None


def _cacheable_headers(
    headers: tuple[tuple[bytes, bytes], ...],
) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(
        (name, value)
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
        and name.lower() != IDEMPOTENCY_STATUS_HEADER
    )


def _with_idempotency_status(
    headers: tuple[tuple[bytes, bytes], ...],
    status: bytes,
) -> list[tuple[bytes, bytes]]:
    result = [
        (name, value)
        for name, value in headers
        if name.lower() != IDEMPOTENCY_STATUS_HEADER
    ]
    result.append((IDEMPOTENCY_STATUS_HEADER, status))
    return result


async def _replay_response(response: CachedASGIResponse, send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": response.status_code,
            "headers": _with_idempotency_status(
                response.headers,
                b"replayed",
            ),
        }
    )
    iterator = iter(response.body.iter_chunks())
    no_chunk = object()
    chunk = next(iterator, no_chunk)
    if chunk is no_chunk:
        await send(
            {"type": "http.response.body", "body": b"", "more_body": False}
        )
        return
    while chunk is not no_chunk:
        next_chunk = next(iterator, no_chunk)
        await send(
            {
                "type": "http.response.body",
                "body": chunk,
                "more_body": next_chunk is not no_chunk,
            }
        )
        chunk = next_chunk


class _RequestBodyDisconnected(Exception):
    pass


class _RequestBodyTooLarge(Exception):
    pass


class _RequestBodyTimedOut(Exception):
    pass


class _IdempotencySpoolCapacity(Exception):
    pass
