import asyncio
from pathlib import Path
from types import SimpleNamespace

import msgspec

from uni_api.admission.memory import (
    AdaptiveMemoryGovernor,
    ProcessMemorySample,
)
from uni_api.upstream import request_replay
from uni_api.upstream.request_replay import MessagesRequestReplayStore


def _governor() -> AdaptiveMemoryGovernor:
    return AdaptiveMemoryGovernor(
        source=lambda: ProcessMemorySample(
            current_bytes=0,
            limit_bytes=1024 * 1024 * 1024,
            source="test",
        ),
        guard_bytes=0,
        guard_ratio=0,
        sample_cache_seconds=0,
    )


async def _consume(content) -> bytes:
    if isinstance(content, bytes):
        return content
    return b"".join([chunk async for chunk in content])


def test_large_messages_payload_reopens_one_provider_spool_for_each_retry(
    tmp_path,
    monkeypatch,
):
    writes = []

    def native_write(payload, path):
        encoded = msgspec.json.encode(payload)
        Path(path).write_bytes(encoded)
        writes.append(path)
        return len(encoded)

    monkeypatch.setattr(
        request_replay,
        "_uni_api_native",
        SimpleNamespace(write_json_file=native_write),
    )

    async def scenario():
        governor = _governor()
        store = MessagesRequestReplayStore(
            4096,
            memory_governor=governor,
            threshold_bytes=1,
            chunk_bytes=17,
            spool_directory=tmp_path,
        )
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "x" * 256}],
        }

        first = await store.prepare(("provider-a", "claude-test"), payload)
        first_path = first.path
        assert first.storage == "disk"
        assert await _consume(first.content) == msgspec.json.encode(payload)
        await first.aclose()

        second = await store.prepare(("provider-a", "claude-test"), payload)
        assert second.path == first_path
        assert await _consume(second.content) == msgspec.json.encode(payload)
        await second.aclose()

        assert len(writes) == 1
        assert first_path is not None and first_path.exists()
        assert governor.snapshot().reserved_bytes == 0
        await store.aclose()
        assert not first_path.exists()

    asyncio.run(scenario())


def test_rust_spool_preserves_stdlib_json_special_float_semantics(
    tmp_path,
    monkeypatch,
):
    def native_write(payload, path):
        encoded = request_replay._serialize_compact_json_bytes(payload)
        Path(path).write_bytes(encoded)
        return len(encoded)

    monkeypatch.setattr(
        request_replay,
        "_uni_api_native",
        SimpleNamespace(write_json_file=native_write),
    )

    async def scenario():
        store = MessagesRequestReplayStore(
            4096,
            memory_governor=_governor(),
            threshold_bytes=1,
            spool_directory=tmp_path,
        )
        prepared = await store.prepare(
            "provider-a",
            {
                "model": "claude-test",
                "values": [float("nan"), float("inf"), float("-inf")],
            },
        )
        assert await _consume(prepared.content) == (
            b'{"model":"claude-test","values":[NaN,Infinity,-Infinity]}'
        )
        await prepared.aclose()
        await store.aclose()

    asyncio.run(scenario())


def test_provider_specific_spool_cache_is_finite(tmp_path, monkeypatch):
    def native_write(payload, path):
        encoded = msgspec.json.encode(payload)
        Path(path).write_bytes(encoded)
        return len(encoded)

    monkeypatch.setattr(
        request_replay,
        "_uni_api_native",
        SimpleNamespace(write_json_file=native_write),
    )

    async def scenario():
        store = MessagesRequestReplayStore(
            4096,
            memory_governor=_governor(),
            threshold_bytes=1,
            max_variants=2,
            spool_directory=tmp_path,
        )
        paths = []
        for index in range(3):
            prepared = await store.prepare(
                (f"provider-{index}", f"claude-{index}"),
                {"model": f"claude-{index}", "messages": []},
            )
            paths.append(prepared.path)
            await prepared.aclose()

        assert paths[0] is not None and not paths[0].exists()
        assert paths[1] is not None and paths[1].exists()
        assert paths[2] is not None and paths[2].exists()
        assert len(store._variants) == 2
        await store.aclose()
        assert all(path is not None and not path.exists() for path in paths)

    asyncio.run(scenario())


def test_small_payload_is_reserved_before_utf8_serialization(monkeypatch):
    async def scenario():
        governor = _governor()
        store = MessagesRequestReplayStore(
            100,
            memory_governor=governor,
            threshold_bytes=1024,
            chunk_bytes=64,
        )
        observed_reservation = []

        real_encode = request_replay._serialize_compact_json_bytes

        def encode(payload):
            observed_reservation.append(
                governor.snapshot().reservations
            )
            return real_encode(payload)

        monkeypatch.setattr(
            request_replay,
            "_serialize_compact_json_bytes",
            encode,
        )
        prepared = await store.prepare(
            "provider-a",
            {"model": "claude-test", "messages": []},
        )
        assert observed_reservation[0]["upstream_serialized_body"] >= (
            len(prepared.content) * 2
        )
        assert observed_reservation[0]["upstream_transport_buffer"] == 128
        assert prepared.storage == "memory"
        await prepared.aclose()
        await store.aclose()
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(scenario())


def test_native_spool_failure_releases_memory_and_removes_partial_file(
    tmp_path,
    monkeypatch,
):
    def failing_writer(_payload, path):
        Path(path).write_bytes(b"partial-secret")
        raise OSError("disk write failed")

    monkeypatch.setattr(
        request_replay,
        "_uni_api_native",
        SimpleNamespace(write_json_file=failing_writer),
    )

    async def scenario():
        governor = _governor()
        store = MessagesRequestReplayStore(
            4096,
            memory_governor=governor,
            threshold_bytes=1,
            spool_directory=tmp_path,
        )
        try:
            await store.prepare(
                "provider-a",
                {"model": "claude-test", "messages": []},
            )
        except OSError as exc:
            assert str(exc) == "disk write failed"
        else:
            raise AssertionError("spool failure must propagate")

        assert governor.snapshot().reserved_bytes == 0
        assert list(tmp_path.iterdir()) == []
        await store.aclose()

    asyncio.run(scenario())


def test_closing_prepared_disk_body_closes_partially_consumed_file(
    tmp_path,
    monkeypatch,
):
    def native_write(payload, path):
        encoded = msgspec.json.encode(payload)
        Path(path).write_bytes(encoded)
        return len(encoded)

    monkeypatch.setattr(
        request_replay,
        "_uni_api_native",
        SimpleNamespace(write_json_file=native_write),
    )

    async def scenario():
        governor = _governor()
        store = MessagesRequestReplayStore(
            4096,
            memory_governor=governor,
            threshold_bytes=1,
            chunk_bytes=8,
            spool_directory=tmp_path,
        )
        prepared = await store.prepare(
            "provider-a",
            {"model": "claude-test", "messages": ["x" * 128]},
        )
        iterator = prepared.content.__aiter__()
        assert await anext(iterator)
        await prepared.aclose()
        assert prepared.content._source is None
        assert governor.snapshot().reserved_bytes == 0
        await store.aclose()

    asyncio.run(scenario())
