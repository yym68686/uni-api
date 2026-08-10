import asyncio
import contextvars
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from uni_api.admission import (
    RequestAdmissionController,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.upstream.client_pool import (
    ClientPool,
    UpstreamAdmissionRejected,
    UpstreamResponseJSONEncodingUnsupported,
    UpstreamUnsupportedContentEncoding,
)
from uni_api.upstream.responses_errors import responses_failure_error
from uni_api.observability.upstream_transport import UpstreamTransportDiagnostics
from upstream import UpstreamAttemptContext, UpstreamRunner


async def _wait_until(predicate, *, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


async def _borrow(pool, url="https://upstream.example/v1"):
    async with pool.get_client(url) as client:
        return client


def _aggregate(pool):
    return pool.snapshot()["admission"]


def _only_key(pool):
    per_key = pool.snapshot()["per_key"]
    assert len(per_key) == 1
    return next(iter(per_key.values()))


def test_network_admission_is_fifo_and_bounded_by_pool_size():
    async def run():
        entered = []
        release = {
            name: asyncio.Event()
            for name in ("one", "two", "three")
        }

        async def handler(request):
            name = request.url.path.removeprefix("/")
            entered.append(name)
            await release[name].wait()
            return httpx.Response(200, content=name.encode())

        pool = ClientPool(
            pool_size=1,
            waiter_limit=2,
            wait_timeout_seconds=1,
        )
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        first = asyncio.create_task(client.post("https://upstream.example/one"))
        await _wait_until(lambda: entered == ["one"])
        second = asyncio.create_task(client.post("https://upstream.example/two"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 1)
        third = asyncio.create_task(client.post("https://upstream.example/three"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 2)

        assert entered == ["one"]
        assert _aggregate(pool)["active"] == 1

        release["one"].set()
        await _wait_until(lambda: entered == ["one", "two"])
        assert _aggregate(pool)["active"] == 1
        assert _aggregate(pool)["waiters"] == 1

        release["two"].set()
        await _wait_until(lambda: entered == ["one", "two", "three"])
        release["three"].set()
        responses = await asyncio.gather(first, second, third)

        assert [response.text for response in responses] == ["one", "two", "three"]
        assert _aggregate(pool)["active"] == 0
        assert _aggregate(pool)["waiters"] == 0
        assert _only_key(pool)["acquired_total"] == 3
        assert _only_key(pool)["wait_ms_max"] > 0
        await pool.close()

    asyncio.run(run())


def test_queue_full_and_wait_timeout_are_distinct_503_admission_errors():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_request):
            entered.set()
            await release.wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool(
            pool_size=1,
            waiter_limit=1,
            wait_timeout_seconds=0.03,
        )
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        holder = asyncio.create_task(client.get("https://upstream.example/holder"))
        await entered.wait()
        queued = asyncio.create_task(client.get("https://upstream.example/queued"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 1)

        with pytest.raises(UpstreamAdmissionRejected) as queue_full:
            await client.get("https://upstream.example/rejected")
        assert queue_full.value.status_code == 503
        assert queue_full.value.reason == "upstream_queue_full"
        assert queue_full.value.retry_after_seconds == 1
        assert "secret" not in queue_full.value.client_key_id

        with pytest.raises(UpstreamAdmissionRejected) as wait_timeout:
            await queued
        assert wait_timeout.value.reason == "upstream_wait_timeout"
        assert _aggregate(pool)["active"] == 1
        assert _aggregate(pool)["waiters"] == 0
        assert _aggregate(pool)["rejected"] == {
            "queue_full": 1,
            "wait_timeout": 1,
        }
        assert _only_key(pool)["attempts_observed"] == 3
        assert _only_key(pool)["wait_ms_max"] >= 20

        release.set()
        assert (await holder).status_code == 200
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_each_client_key_has_independent_capacity_and_a_redacted_snapshot_label():
    async def run():
        entered = []
        release = asyncio.Event()

        async def handler(request):
            entered.append(request.url.host)
            await release.wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool(pool_size=1, waiter_limit=0, wait_timeout_seconds=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        first_client = await _borrow(pool, "https://first.example/v1")
        second_client = await _borrow(pool, "https://second.example/v1")

        first = asyncio.create_task(first_client.get("https://first.example/request"))
        second = asyncio.create_task(second_client.get("https://second.example/request"))
        await _wait_until(lambda: sorted(entered) == ["first.example", "second.example"])

        assert _aggregate(pool)["active"] == 2
        labels = list(pool.snapshot()["per_key"])
        assert len(labels) == 2
        assert all("proxy=0" in label and "id=" in label for label in labels)

        with pytest.raises(UpstreamAdmissionRejected) as rejected:
            await first_client.get("https://first.example/rejected")
        assert rejected.value.reason == "upstream_queue_full"

        release.set()
        await asyncio.gather(first, second)
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_queued_and_active_cancellation_do_not_leak_capacity():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_request):
            entered.set()
            await release.wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool(pool_size=1, waiter_limit=1, wait_timeout_seconds=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        holder = asyncio.create_task(client.get("https://upstream.example/holder"))
        await entered.wait()
        queued = asyncio.create_task(client.get("https://upstream.example/queued"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 1)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert _aggregate(pool)["waiters"] == 0
        assert _aggregate(pool)["active"] == 1
        assert _aggregate(pool)["cancelled_total"] == 1

        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        await _wait_until(lambda: _aggregate(pool)["active"] == 0)

        release.set()
        response = await client.get("https://upstream.example/recovered")
        assert response.status_code == 200
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


class _ControlledStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        yield b"first"
        await self.release.wait()
        yield b"second"

    async def aclose(self):
        self.closed.set()


def test_non_stream_lease_covers_the_complete_response_body_read():
    async def run():
        stream = _ControlledStream()

        async def handler(_request):
            return httpx.Response(200, stream=stream)

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        request = asyncio.create_task(client.get("https://upstream.example/body"))
        await stream.started.wait()
        assert _aggregate(pool)["active"] == 1
        assert not request.done()

        stream.release.set()
        response = await request
        assert response.content == b"firstsecond"
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_stream_context_lease_releases_on_response_aclose_and_exit_is_idempotent():
    async def run():
        stream = _ControlledStream()

        async def handler(_request):
            return httpx.Response(200, stream=stream)

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        stream_context = client.stream("GET", "https://upstream.example/events")
        response = await stream_context.__aenter__()
        assert isinstance(response, httpx.Response)
        assert _aggregate(pool)["active"] == 1

        iterator = response.aiter_raw()
        assert await anext(iterator) == b"first"
        assert _aggregate(pool)["active"] == 1

        await asyncio.gather(response.aclose(), response.aclose())
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        assert await stream_context.__aexit__(None, None, None) is False
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_stream_consumer_cancellation_closes_response_and_releases_lease():
    async def run():
        stream = _ControlledStream()
        consumed_first = asyncio.Event()

        async def handler(_request):
            return httpx.Response(200, stream=stream)

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        async def consume():
            async with client.stream("GET", "https://upstream.example/events") as response:
                async for _chunk in response.aiter_raw():
                    consumed_first.set()

        consumer = asyncio.create_task(consume())
        await consumed_first.wait()
        assert _aggregate(pool)["active"] == 1

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_send_stream_true_binds_lease_to_the_real_httpx_response():
    async def run():
        stream = _ControlledStream()

        async def handler(_request):
            return httpx.Response(200, stream=stream)

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        request = client.build_request("GET", "https://upstream.example/send")
        assert _aggregate(pool)["active"] == 0

        response = await client.send(request, stream=True)
        assert type(response) is httpx.Response
        assert _aggregate(pool)["active"] == 1
        await asyncio.gather(response.aclose(), response.aclose())
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_stream_context_forces_identity_and_rejects_compressed_success_before_read():
    async def run():
        stream = _ControlledStream()

        async def handler(request):
            assert request.headers["accept-encoding"] == "identity"
            return httpx.Response(
                200,
                headers={"content-encoding": "gzip"},
                stream=stream,
            )

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        context = client.stream(
            "GET",
            "https://upstream.example/events",
            headers={"Accept-Encoding": "zstd"},
        )
        with pytest.raises(UpstreamUnsupportedContentEncoding) as rejected:
            await context.__aenter__()
        assert rejected.value.status_code == 502
        assert rejected.value.content_encoding == "gzip"
        assert not stream.started.is_set()
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_send_stream_true_forces_identity_and_releases_on_compressed_success():
    async def run():
        stream = _ControlledStream()

        async def handler(request):
            assert request.headers["accept-encoding"] == "identity"
            return httpx.Response(
                200,
                headers={"content-encoding": "deflate"},
                stream=stream,
            )

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        request = client.build_request(
            "GET",
            "https://upstream.example/send",
            headers={"Accept-Encoding": "gzip"},
        )

        with pytest.raises(UpstreamUnsupportedContentEncoding):
            await client.send(request, stream=True)
        assert not stream.started.is_set()
        assert stream.closed.is_set()
        assert _aggregate(pool)["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_generic_json_is_structurally_bounded_and_trusted_binary_uses_two_copies():
    async def run():
        async def handler(request):
            if request.url.path == "/json":
                return httpx.Response(200, content=b"[0,0]")
            return httpx.Response(200, content=b"binary")

        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=40,
            body_budget_bytes=40,
            max_response_bytes=40,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        try:
            # A dense JSON container is charged for its future Python object
            # graph and therefore exceeds this deliberately tiny budget.
            with pytest.raises(UpstreamResponseBudgetExhausted):
                await client.post("https://upstream.example/json")
            assert controller.snapshot()["reserved_response_bytes"] == 0

            response = await client.post_buffered_binary(
                "https://upstream.example/tts"
            )
            assert response.content == b"binary"
            # read_limited_response_body briefly retains bytearray + bytes.
            assert controller.snapshot()["reserved_response_bytes"] == 12
        finally:
            await pool.close()
            reset_request_admission_lease(token)
            await lease.release()

        assert controller.snapshot()["reserved_response_bytes"] == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [
        b"<html>maintenance</html>" + b"x" * 5000,
        b"{" + b"x" * 5000,
    ],
)
def test_non_json_upstream_error_preserves_status_without_prefix_false_positive(body):
    async def run():
        async def handler(_request):
            return httpx.Response(
                503,
                headers={"content-type": "text/plain"},
                content=body,
            )

        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        lease = await controller.acquire()
        token = bind_request_admission_lease(lease)
        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        try:
            client = await _borrow(pool)
            response = await client.get("https://upstream.example/error")
            assert response.status_code == 503
            assert response.content == body
        finally:
            await pool.close()
            reset_request_admission_lease(token)
            await lease.release()

    asyncio.run(run())


def test_utf8_bom_json_is_supported_but_utf16_json_is_rejected_before_parse():
    async def run():
        bodies = [
            b"\xef\xbb\xbf  {\"ok\":true}",
            "{\"ok\":true}".encode("utf-16"),
        ]

        async def handler(_request):
            return httpx.Response(200, content=bodies.pop(0))

        pool = ClientPool(pool_size=1)
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        try:
            assert (await client.get("https://upstream.example/utf8")).status_code == 200
            with pytest.raises(UpstreamResponseJSONEncodingUnsupported):
                await client.get("https://upstream.example/utf16")
        finally:
            await pool.close()

    asyncio.run(run())


def test_real_wait_is_reported_to_trace_and_legacy_callbacks_track_active_only():
    async def run():
        entered = []
        releases = {"one": asyncio.Event(), "two": asyncio.Event()}
        trace_context = contextvars.ContextVar("trace", default=None)
        callback_active = 0
        callback_peak = 0

        class Trace:
            def __init__(self):
                self.spans = {}

            def mark(self, stage):
                self.spans[stage] = True

            def add_ms(self, name, value):
                self.spans[name] = value

        def begin(trace):
            nonlocal callback_active, callback_peak
            callback_active += 1
            callback_peak = max(callback_peak, callback_active)
            # Match the legacy RuntimeGauges callback that used to write zero.
            if trace is not None:
                trace.add_ms("upstream_pool_wait_ms", 0)

        def end():
            nonlocal callback_active
            callback_active -= 1

        async def handler(request):
            name = request.url.path.removeprefix("/")
            entered.append(name)
            await releases[name].wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool(
            pool_size=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
            current_trace=lambda: trace_context.get(),
            begin_upstream_pool=begin,
            end_upstream_pool=end,
        )
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        traces = {"one": Trace(), "two": Trace()}

        async def make_request(name):
            token = trace_context.set(traces[name])
            try:
                return await client.get(f"https://upstream.example/{name}")
            finally:
                trace_context.reset(token)

        first = asyncio.create_task(make_request("one"))
        await _wait_until(lambda: entered == ["one"])
        second = asyncio.create_task(make_request("two"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 1)
        assert callback_active == 1

        await asyncio.sleep(0.01)
        releases["one"].set()
        await _wait_until(lambda: entered == ["one", "two"])
        releases["two"].set()
        await asyncio.gather(first, second)

        assert callback_peak == 1
        assert callback_active == 0
        assert traces["two"].spans["upstream_pool_wait_ms"] > 0
        assert traces["two"].spans["upstream_pool_wait_ms_total"] > 0
        assert traces["two"].spans["upstream_pool_admission_count"] == 1
        assert _only_key(pool)["wait_ms_max"] > 0
        await pool.close()

    asyncio.run(run())


def test_high_contention_never_exceeds_network_or_active_gauge_capacity():
    async def run():
        network_active = 0
        network_peak = 0
        callback_active = 0
        callback_peak = 0

        async def handler(_request):
            nonlocal network_active, network_peak
            network_active += 1
            network_peak = max(network_peak, network_active)
            try:
                await asyncio.sleep(0.002)
                return httpx.Response(200, content=b"ok")
            finally:
                network_active -= 1

        def begin(_trace):
            nonlocal callback_active, callback_peak
            callback_active += 1
            callback_peak = max(callback_peak, callback_active)

        def end():
            nonlocal callback_active
            callback_active -= 1

        pool = ClientPool(
            pool_size=3,
            waiter_limit=97,
            wait_timeout_seconds=2,
            begin_upstream_pool=begin,
            end_upstream_pool=end,
        )
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        responses = await asyncio.gather(
            *(
                client.get(f"https://upstream.example/request-{index}")
                for index in range(100)
            )
        )

        assert all(response.status_code == 200 for response in responses)
        assert network_peak == 3
        assert callback_peak == 3
        assert network_active == callback_active == 0
        assert _aggregate(pool)["active"] == 0
        assert _aggregate(pool)["waiters"] == 0
        assert _aggregate(pool)["acquired_total"] == 100
        await pool.close()

    asyncio.run(run())


def test_httpx_pool_timeout_is_explicit_and_not_the_connection_count(monkeypatch):
    created = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        async def aclose(self):
            return None

    monkeypatch.setattr("uni_api.upstream.client_pool.httpx.AsyncClient", FakeAsyncClient)

    async def run():
        pool = ClientPool(pool_size=7, pool_timeout_seconds=0.25)
        await pool.init({"headers": {}})
        await _borrow(pool)

        assert len(created) == 1
        assert created[0].kwargs["limits"].max_connections == 7
        assert created[0].kwargs["timeout"].pool == 0.25
        assert created[0].kwargs["timeout"].pool != pool.pool_size
        assert pool.snapshot()["pool_timeout_seconds"] == 0.25
        await pool.close()

    asyncio.run(run())


def test_resource_mode_disables_httpx_max_connections(monkeypatch):
    created = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        async def aclose(self):
            return None

    monkeypatch.setattr(
        "uni_api.upstream.client_pool.httpx.AsyncClient",
        FakeAsyncClient,
    )

    async def run():
        pool = ClientPool()
        await pool.init({"headers": {}})
        await _borrow(pool)

        assert created[0].kwargs["limits"].max_connections is None
        assert pool.snapshot()["admission"]["capacity_per_key"] is None
        await pool.close()

    asyncio.run(run())


def test_default_pool_has_no_request_count_capacity():
    pool = ClientPool()

    assert pool.pool_size is None
    assert pool.waiter_limit is None
    assert pool.resource_mode is True


def test_resource_mode_does_not_serialize_upstream_requests():
    async def run():
        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_request):
            nonlocal entered
            entered += 1
            if entered == 128:
                all_entered.set()
            await release.wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool()
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)
        requests = [
            asyncio.create_task(
                client.get(f"https://upstream.example/{index}")
            )
            for index in range(128)
        ]
        await asyncio.wait_for(all_entered.wait(), timeout=2)

        snapshot = pool.snapshot()
        assert snapshot["admission"]["mode"] == "weighted_resources"
        assert snapshot["admission"]["capacity_per_key"] is None
        assert snapshot["admission"]["active"] == 128

        release.set()
        responses = await asyncio.gather(*requests)
        assert all(response.status_code == 200 for response in responses)
        assert pool.snapshot()["admission"]["active"] == 0
        await pool.close()

    asyncio.run(run())


def test_close_racing_an_active_request_and_waiter_rejects_without_leaking():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_request):
            entered.set()
            await release.wait()
            return httpx.Response(200, content=b"ok")

        pool = ClientPool(
            pool_size=1,
            waiter_limit=1,
            wait_timeout_seconds=1,
        )
        await pool.init({"transport": httpx.MockTransport(handler)})
        client = await _borrow(pool)

        holder = asyncio.create_task(client.get("https://upstream.example/holder"))
        await entered.wait()
        waiter = asyncio.create_task(client.get("https://upstream.example/waiter"))
        await _wait_until(lambda: _aggregate(pool)["waiters"] == 1)

        await pool.close()
        closed_snapshot = pool.snapshot()
        assert closed_snapshot["closed"] is True
        assert closed_snapshot["closing"] is False
        assert closed_snapshot["client_count"] == 0

        release.set()
        assert (await holder).status_code == 200
        with pytest.raises(UpstreamAdmissionRejected) as rejected:
            await waiter
        assert rejected.value.reason == "upstream_pool_closed"
        assert _aggregate(pool)["active"] == 0
        assert _aggregate(pool)["waiters"] == 0

        with pytest.raises(UpstreamAdmissionRejected) as borrowed_after_close:
            await _borrow(pool)
        assert borrowed_after_close.value.reason == "upstream_pool_closed"

        with pytest.raises(UpstreamAdmissionRejected) as used_after_close:
            await client.get("https://upstream.example/after-close")
        assert used_after_close.value.reason == "upstream_pool_closed"

        # Idempotent repeated close must not recreate or re-close a client.
        await asyncio.gather(pool.close(), pool.close())
        assert pool.snapshot()["client_count"] == 0

    asyncio.run(run())


def test_local_admission_fails_fast_and_preserves_retry_headers(monkeypatch):
    async def run():
        attempts = [
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name="provider-a",
                original_model="model-a",
            ),
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name="provider-b",
                original_model="model-a",
            ),
        ]

        class Plan:
            auto_retry = True
            api_list = []
            status_code = 503
            error_message = "upstream_wait_timeout"
            request_model_name = "model-a"

            async def next_provider(self):
                return attempts.pop(0) if attempts else None

            def record_failure(self, status_code, error_message):
                self.status_code = status_code
                self.error_message = error_message

        failures = [
            UpstreamAdmissionRejected(
                "upstream_queue_full",
                retry_after_seconds=2,
                client_key_id="redacted-a",
            ),
            UpstreamAdmissionRejected(
                "upstream_wait_timeout",
                retry_after_seconds=7,
                client_key_id="redacted-b",
            ),
        ]

        async def execute_attempt(_attempt):
            raise failures.pop(0)

        monkeypatch.setattr("upstream.should_retry_provider", lambda *_args, **_kwargs: True)
        plan = Plan()
        response = await UpstreamRunner(plan).run(
            execute_attempt,
            build_final_response=lambda completed: httpx.Response(
                completed.status_code,
                json={"error": completed.error_message},
            ),
        )

        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert (
            response.headers["x-uni-api-admission-reason"]
            == "upstream_queue_full"
        )
        assert (
            response.headers["x-uni-api-status-origin"]
            == "ember_local_admission"
        )
        assert len(failures) == 1
        assert len(attempts) == 1

    asyncio.run(run())


def test_nonlocal_failures_do_not_gain_admission_headers(monkeypatch):
    async def run():
        attempts = [
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name="provider-a",
                original_model="model-a",
            ),
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name="provider-b",
                original_model="model-a",
            ),
        ]

        class Plan:
            auto_retry = True
            api_list = []
            status_code = 500
            error_message = "failed"
            request_model_name = "model-a"

            async def next_provider(self):
                return attempts.pop(0) if attempts else None

            def record_failure(self, status_code, error_message):
                self.status_code = status_code
                self.error_message = error_message

        failures = [RuntimeError("first upstream failure"), RuntimeError("real upstream failure")]

        async def execute_attempt(_attempt):
            raise failures.pop(0)

        monkeypatch.setattr("upstream.should_retry_provider", lambda *_args, **_kwargs: True)
        plan = Plan()
        response = await UpstreamRunner(plan).run(
            execute_attempt,
            build_final_response=lambda completed: httpx.Response(
                completed.status_code,
                json={"error": completed.error_message},
            ),
        )

        assert response.status_code == 500
        assert "retry-after" not in response.headers
        assert "x-uni-api-admission-reason" not in response.headers

    asyncio.run(run())


@pytest.mark.parametrize(
    ("wire_status", "detail", "expected_status"),
    [
        (503, "provider unavailable", 503),
        (
            400,
            "Model claude-opus-5 has not been priced by the administrator yet.",
            502,
        ),
    ],
)
def test_provider_http_status_origin_survives_status_normalization(
    wire_status,
    detail,
    expected_status,
):
    async def run():
        class Plan:
            auto_retry = False

            def record_failure(self, status_code, _error_message):
                assert status_code == expected_status

        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        diagnostics.facts["upstream_http_status_code"] = wire_status
        attempt = UpstreamAttemptContext(
            plan=Plan(),
            provider={"preferences": {}},
            provider_name="provider-a",
            original_model="model-a",
            state={"_transport_diagnostics": diagnostics},
        )
        result = await UpstreamRunner(attempt.plan)._handle_failure(
            attempt,
            HTTPException(status_code=wire_status, detail=detail),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            prepare_failure=False,
        )

        assert result.response.status_code == expected_status
        assert result.status_origin == "provider_http"
        assert result.response.headers["x-uni-api-status-origin"] == "provider_http"
        assert "x-uni-api-admission-reason" not in result.response.headers

    asyncio.run(run())


def test_local_upstream_admission_does_not_cool_or_mark_channel_failure(monkeypatch):
    async def run():
        cooldown_checks = 0
        after_failure_states = []

        class Plan:
            auto_retry = False

            def record_failure(self, status_code, error_message):
                assert status_code == 503
                assert error_message == "upstream_wait_timeout"

        async def fail_if_cooled(*_args, **_kwargs):
            raise AssertionError("local admission must not cool a provider key")

        async def should_cool(*_args):
            nonlocal cooldown_checks
            cooldown_checks += 1
            return True

        monkeypatch.setattr("upstream.maybe_cool_provider_api_key", fail_if_cooled)
        plan = Plan()
        runner = UpstreamRunner(plan)
        attempt = UpstreamAttemptContext(
            plan=plan,
            provider={"preferences": {}},
            provider_name="provider-a",
            original_model="model-a",
            state={"track_channel_stats": True},
        )
        error = UpstreamAdmissionRejected(
            "upstream_wait_timeout",
            retry_after_seconds=5,
            client_key_id="redacted",
        )

        result = await runner._handle_failure(
            attempt,
            error,
            after_failure=lambda attempt, *_args: after_failure_states.append(
                dict(attempt.state)
            ),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            should_cool_down=should_cool,
            prepare_failure=False,
        )

        assert cooldown_checks == 0
        assert after_failure_states[0]["track_channel_stats"] is False
        assert after_failure_states[0]["local_admission_rejected"] is True
        assert result.response.status_code == 503
        assert result.response.headers["retry-after"] == "5"
        assert (
            result.response.headers["x-uni-api-admission-reason"]
            == "upstream_wait_timeout"
        )
        assert (
            result.response.headers["x-uni-api-status-origin"]
            == "ember_local_admission"
        )

    asyncio.run(run())


def test_httpx_pool_timeout_is_local_overload_and_never_penalizes_provider(
    monkeypatch,
):
    async def run():
        after_failure_states = []
        current_info = {}

        class Plan:
            auto_retry = False

            def record_failure(self, status_code, error_message):
                assert status_code == 503
                assert error_message == (
                    "Local upstream connection pool timed out after "
                    "0.25 seconds"
                )

        async def fail_if_penalized(*_args, **_kwargs):
            raise AssertionError(
                "PoolTimeout must not cool a key or exclude a channel"
            )

        monkeypatch.setattr(
            "upstream.maybe_cool_provider_api_key",
            fail_if_penalized,
        )
        monkeypatch.setattr(
            "upstream.maybe_exclude_failed_channel",
            fail_if_penalized,
        )
        request = httpx.Request(
            "POST",
            "https://provider.example/v1/responses",
            extensions={"timeout": {"pool": 0.25}},
        )
        attempt = UpstreamAttemptContext(
            plan=Plan(),
            provider={
                "preferences": {"api_key_cooldown_period": 300},
            },
            provider_name="provider-a",
            original_model="model-a",
            provider_api_key_raw="provider-key",
            state={"track_channel_stats": True},
        )
        result = await UpstreamRunner(
            attempt.plan,
            observability_context=current_info,
        )._handle_failure(
            attempt,
            httpx.PoolTimeout("pool exhausted", request=request),
            after_failure=lambda item, *_args: after_failure_states.append(
                dict(item.state)
            ),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            allow_channel_exclusion=True,
            should_cool_down=lambda *_args: (_ for _ in ()).throw(
                AssertionError("PoolTimeout must not run cooldown policy")
            ),
            prepare_failure=False,
        )

        assert result.response.status_code == 503
        assert result.status_origin == "ember_local_overload"
        assert (
            result.response.headers["x-uni-api-status-origin"]
            == "ember_local_overload"
        )
        state = after_failure_states[0]
        assert state["track_channel_stats"] is False
        assert state["transport_error_kind"] == "pool_timeout"
        assert state["transport_error_owner"] == "ember_local_overload"
        assert state["transport_error_phase"] == "pool_acquire"
        assert state["provider_penalty_eligible"] is False
        assert state["local_overload"] is True
        assert current_info["transport_error_count"] == 1
        assert current_info["local_overload_count"] == 1

    asyncio.run(run())


def test_read_timeout_before_first_byte_is_visible_on_routing_attempt():
    async def run():
        class Plan:
            auto_retry = False

            def record_failure(self, status_code, _error_message):
                assert status_code == 504

        routing_entry = {}
        diagnostics = UpstreamTransportDiagnostics(routing_entry)
        responses_diagnostics = SimpleNamespace(
            facts={
                "phase": "precommit",
                "upstream_chunk_count": 0,
            }
        )
        attempt = UpstreamAttemptContext(
            plan=Plan(),
            provider={"preferences": {}},
            provider_name="provider-a",
            original_model="model-a",
            state={
                "_routing_attempt_entry": routing_entry,
                "_transport_diagnostics": diagnostics,
                "responses_stream_diagnostics_tracker": (
                    responses_diagnostics
                ),
            },
        )

        result = await UpstreamRunner(attempt.plan)._handle_failure(
            attempt,
            httpx.ReadTimeout("timed out"),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            prepare_failure=False,
        )

        assert result.response.status_code == 504
        assert routing_entry["transport_error_kind"] == "read_timeout"
        assert (
            routing_entry["transport_error_phase"]
            == "before_first_byte"
        )
        assert routing_entry["transport_error_owner"] == "upstream_transport"
        assert routing_entry["provider_penalty_eligible"] is True

    asyncio.run(run())


def test_generic_read_timeout_uses_request_first_byte_state():
    async def run():
        class Plan:
            auto_retry = False

            def record_failure(self, status_code, _error_message):
                assert status_code == 504

        routing_entry = {}
        diagnostics = UpstreamTransportDiagnostics(routing_entry)
        diagnostics.capture_response(
            httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://provider.example/v1/chat/completions",
                ),
            ),
            client=SimpleNamespace(),
        )
        attempt = UpstreamAttemptContext(
            plan=Plan(),
            provider={"preferences": {}},
            provider_name="provider-a",
            original_model="model-a",
            state={
                "_routing_attempt_entry": routing_entry,
                "_transport_diagnostics": diagnostics,
            },
        )

        result = await UpstreamRunner(
            attempt.plan,
            observability_context={"_first_byte_observed": False},
        )._handle_failure(
            attempt,
            httpx.ReadTimeout("timed out"),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            prepare_failure=False,
        )

        assert result.response.status_code == 504
        assert (
            routing_entry["transport_error_phase"]
            == "before_first_byte"
        )

    asyncio.run(run())


def test_request_scoped_semantic_error_does_not_cool_or_mark_channel_failure(
    monkeypatch,
):
    async def run():
        after_failure_states = []

        class Plan:
            auto_retry = True

            def record_failure(self, status_code, _error_message):
                assert status_code == 400

        error = responses_failure_error(
            {
                "type": "error",
                "error": {
                    "message": "context window exceeded",
                    "code": "context_length_exceeded",
                    "type": "invalid_request_error",
                },
            },
            event_type="error",
        )
        assert error is not None

        async def fail_if_cooled(*_args, **_kwargs):
            raise AssertionError("request-scoped errors must not cool a key")

        monkeypatch.setattr("upstream.maybe_cool_provider_api_key", fail_if_cooled)
        attempt = UpstreamAttemptContext(
            plan=Plan(),
            provider={
                "base_url": "https://models.inference.ai.azure.com/v1/responses",
                "preferences": {"api_key_cooldown_period": 300},
            },
            provider_name="provider-a",
            original_model="model-a",
            provider_api_key_raw="provider-key",
            state={"track_channel_stats": True},
        )
        result = await UpstreamRunner(attempt.plan)._handle_failure(
            attempt,
            error,
            after_failure=lambda item, *_args: after_failure_states.append(
                dict(item.state)
            ),
            build_error_response=lambda status, message: httpx.Response(
                status,
                json={"error": message},
            ),
            should_cool_down=lambda *_args: True,
            prepare_failure=False,
        )

        assert result.response.status_code == 400
        assert after_failure_states[0]["track_channel_stats"] is False
        assert result.should_retry is False

    asyncio.run(run())
