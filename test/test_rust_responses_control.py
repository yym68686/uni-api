import asyncio
from types import SimpleNamespace

import pytest
from starlette.responses import Response

from uni_api.rust_responses_control import (
    RustResponsesControlError,
    RustResponsesControlPlane,
)


class _FakeExecution:
    def __init__(self):
        self.current_info = {}
        self.commits = []
        self.attempts = [
            SimpleNamespace(routing_attempt_id="attempt-1"),
            SimpleNamespace(routing_attempt_id="attempt-2"),
        ]

    async def _run_attempts(self, *, execute_attempt):
        for attempt in self.attempts:
            try:
                return await execute_attempt(attempt)
            except RuntimeError:
                continue
        return Response("exhausted", status_code=502)

    async def _build_rust_stream_plan(self, attempt):
        return {
            "kind": "plan",
            "attempt_id": attempt.routing_attempt_id,
            "url": "https://provider.example/v1/responses",
        }

    async def _resolve_rust_stream_outcome(self, _attempt, outcome):
        if outcome["kind"] == "http_error":
            raise RuntimeError("retry")
        return Response("", status_code=200)

    async def _observe_rust_stream_commit(self, attempt, observation):
        self.commits.append((attempt.routing_attempt_id, dict(observation)))


def test_control_plane_coordinates_retry_commit_and_completion():
    async def scenario():
        plane = RustResponsesControlPlane()
        execution = _FakeExecution()
        session = await plane.create(execution)

        first = await session.next_message()
        assert first["attempt_id"] == "attempt-1"
        second = await session.advance(
            attempt_id="attempt-1",
            outcome={"kind": "http_error"},
        )
        assert second["attempt_id"] == "attempt-2"

        ack = await session.observe_commit(
            attempt_id="attempt-2",
            observation={"commit_reason": "real_output"},
        )
        assert ack == {"kind": "ack", "attempt_id": "attempt-2"}
        final = await session.complete(
            attempt_id="attempt-2",
            outcome={"kind": "completed"},
        )
        assert final["kind"] == "final"
        assert final["status_code"] == 200
        assert execution.commits == [
            ("attempt-2", {"commit_reason": "real_output"})
        ]

        body = session.control_body()
        with pytest.raises(StopAsyncIteration):
            await body.__anext__()
        await plane.release(session)
        assert plane.snapshot() == {"active_sessions": 0}

    asyncio.run(scenario())


def test_control_plane_rejects_stale_attempt_outcomes():
    async def scenario():
        plane = RustResponsesControlPlane()
        session = await plane.create(_FakeExecution())
        await session.next_message()
        with pytest.raises(RustResponsesControlError, match="stale attempt"):
            await session.advance(
                attempt_id="wrong-attempt",
                outcome={"kind": "http_error"},
            )
        await plane.release(session)

    asyncio.run(scenario())
