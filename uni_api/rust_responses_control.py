from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from starlette.responses import Response


class RustResponsesControlError(RuntimeError):
    """Raised when the loopback Rust/Python control protocol is violated."""


@dataclass(frozen=True, slots=True)
class RustResponsesResolvedOutcome:
    runner_result: Any
    reply: dict[str, Any] = field(default_factory=dict)


def _response_message(response: Any) -> dict[str, Any]:
    body = getattr(response, "body", b"")
    if isinstance(body, str):
        body = body.encode("utf-8")
    elif not isinstance(body, bytes):
        body = bytes(body or b"")
    headers = {
        str(name): str(value)
        for name, value in getattr(response, "headers", {}).items()
        if str(name).lower() not in {"content-length", "transfer-encoding"}
    }
    return {
        "kind": "final",
        "status_code": int(getattr(response, "status_code", 500) or 500),
        "headers": headers,
        "body_b64": base64.b64encode(body).decode("ascii"),
    }


@dataclass(slots=True)
class RustResponsesSession:
    execution: Any
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _messages: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1),
        init=False,
        repr=False,
    )
    _ready: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _finished: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _active_attempt: Any = field(default=None, init=False, repr=False)
    _active_outcome: asyncio.Future[dict[str, Any]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _final_message: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _completion_reply: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def start(self) -> None:
        if self._task is not None:
            raise RustResponsesControlError("Rust Responses session already started")
        self._task = asyncio.create_task(
            self._run(),
            name=f"rust-responses-control-{self.session_id[:12]}",
        )

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def next_message(self) -> dict[str, Any]:
        return await self._messages.get()

    async def advance(
        self,
        *,
        attempt_id: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        future = self._active_outcome
        attempt = self._active_attempt
        if future is None or attempt is None or future.done():
            raise RustResponsesControlError("No Rust Responses attempt is awaiting an outcome")
        active_attempt_id = str(getattr(attempt, "routing_attempt_id", "") or "")
        if not attempt_id or attempt_id != active_attempt_id:
            raise RustResponsesControlError("Rust Responses outcome used a stale attempt ID")
        future.set_result(dict(outcome))
        return await self.next_message()

    async def complete(
        self,
        *,
        attempt_id: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        message = await self.advance(attempt_id=attempt_id, outcome=outcome)
        task = self._task
        if task is not None:
            await task
        reply = dict(self._final_message or message)
        reply.update(self._completion_reply)
        return reply

    async def observe_commit(
        self,
        *,
        attempt_id: str,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        attempt = self._active_attempt
        if attempt is None:
            raise RustResponsesControlError("No active Rust Responses attempt")
        active_attempt_id = str(getattr(attempt, "routing_attempt_id", "") or "")
        if not attempt_id or attempt_id != active_attempt_id:
            raise RustResponsesControlError("Rust Responses commit used a stale attempt ID")
        await self.execution._observe_rust_stream_commit(attempt, observation)
        return {"kind": "ack", "attempt_id": active_attempt_id}

    async def cancel(self) -> None:
        future = self._active_outcome
        attempt = self._active_attempt
        if future is not None and attempt is not None and not future.done():
            future.set_result(
                {
                    "kind": "downstream_disconnected",
                    "status_code": 499,
                }
            )
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def control_body(self) -> AsyncIterator[bytes]:
        """Keep the original Python request lifecycle open until Rust finishes."""

        try:
            await self._finished.wait()
            if False:  # pragma: no cover - makes this an async generator
                yield b""
        finally:
            if not self._finished.is_set():
                await self.cancel()

    async def _publish(self, message: dict[str, Any]) -> None:
        await self._messages.put(message)
        self._ready.set()

    async def _execute_attempt(self, attempt: Any) -> Any:
        self._active_attempt = attempt
        loop = asyncio.get_running_loop()
        self._active_outcome = loop.create_future()
        try:
            plan = await self.execution._build_rust_stream_plan(attempt)
            await self._publish(plan)
            outcome = await self._active_outcome
            resolved = await self.execution._resolve_rust_stream_outcome(
                attempt,
                outcome,
            )
            if isinstance(resolved, RustResponsesResolvedOutcome):
                self._completion_reply = dict(resolved.reply)
                return resolved.runner_result
            return resolved
        finally:
            self._active_outcome = None
            self._active_attempt = None

    async def _run(self) -> None:
        try:
            response = await self.execution._run_attempts(
                execute_attempt=self._execute_attempt,
            )
            if not isinstance(response, Response):
                response = Response(content=str(response or ""), status_code=500)
            if not self.execution.current_info.get(
                "rust_responses_external_committed"
            ):
                self.execution.current_info["status_code"] = int(
                    response.status_code
                )
            self._final_message = _response_message(response)
            terminal = getattr(
                self.execution,
                "last_response_failed_terminal",
                None,
            )
            terminal_data = getattr(terminal, "data", None)
            if isinstance(terminal_data, bytes) and terminal_data:
                self._final_message["stream_failure_terminal_b64"] = (
                    base64.b64encode(terminal_data).decode("ascii")
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.execution.current_info["success"] = False
            self.execution.current_info["error_type"] = type(exc).__name__
            self.execution.current_info["stream_outcome"] = "rust_control_error"
            self._final_message = _response_message(
                Response(content="Rust Responses control plane failed", status_code=500)
            )
        finally:
            try:
                await self.execution._release_request_retry_payload()
            except (AttributeError, RuntimeError):
                pass
            if self._final_message is not None:
                await self._publish(self._final_message)
            self._finished.set()


class RustResponsesControlPlane:
    """Request-scoped rendezvous for the Rust streaming data plane."""

    def __init__(self) -> None:
        self._sessions: dict[str, RustResponsesSession] = {}

    async def create(self, execution: Any) -> RustResponsesSession:
        session = RustResponsesSession(execution=execution)
        self._sessions[session.session_id] = session
        execution.current_info["rust_responses_session_id"] = session.session_id
        session.start()
        await session.wait_ready()
        return session

    def get(self, session_id: str) -> RustResponsesSession:
        session = self._sessions.get(str(session_id or ""))
        if session is None:
            raise RustResponsesControlError("Unknown Rust Responses session")
        return session

    async def release(self, session: RustResponsesSession) -> None:
        self._sessions.pop(session.session_id, None)
        await session.cancel()

    def snapshot(self) -> dict[str, int]:
        return {"active_sessions": len(self._sessions)}
