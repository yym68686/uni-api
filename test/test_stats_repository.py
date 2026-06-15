from datetime import datetime, timezone
from types import SimpleNamespace

from uni_api.persistence.repositories import StatsRepository
from utils import get_sorted_api_keys


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows, calls):
        self.rows = rows
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, query):
        self.calls.append(query)
        return _FakeResult(self.rows)


def _fake_session_factory(rows, calls):
    return lambda: _FakeSession(rows, calls)


class _QueuedResult:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def fetchall(self):
        return self._rows

    def scalar_one(self):
        return self._scalar


class _QueuedSession:
    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, query):
        self.calls.append(query)
        return self.results.pop(0)


def _queued_session_factory(results, calls):
    return lambda: _QueuedSession(results, calls)


async def test_stats_repository_channel_key_stats_sorts_by_success_rate_then_volume():
    rows = [
        SimpleNamespace(provider_api_key="key-low", success_count=1, total_requests=2),
        SimpleNamespace(provider_api_key="key-best", success_count=9, total_requests=10),
        SimpleNamespace(provider_api_key="key-tie-more", success_count=18, total_requests=20),
        SimpleNamespace(provider_api_key="key-empty", success_count=0, total_requests=0),
    ]
    calls = []
    repository = StatsRepository(_fake_session_factory(rows, calls))

    result = await repository.query_channel_key_stats(
        provider_name="provider-a",
        start_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert [row["api_key"] for row in result] == [
        "key-tie-more",
        "key-best",
        "key-low",
        "key-empty",
    ]
    assert result[0]["success_rate"] == 0.9
    assert calls


async def test_smart_key_reorder_uses_repository_backed_channel_stats(monkeypatch):
    async def fake_query_channel_key_stats(provider_name, start_dt=None, end_dt=None):
        assert provider_name == "provider-a"
        return [
            {"api_key": "key-c", "success_rate": 0.95, "total_requests": 20},
            {"api_key": "key-a", "success_rate": 0.80, "total_requests": 100},
        ]

    monkeypatch.setattr("utils.query_channel_key_stats", fake_query_channel_key_stats)

    reordered = await get_sorted_api_keys("provider-a", ["key-a", "key-b", "key-c"], group_size=100)

    assert reordered == ["key-c", "key-a", "key-b"]


async def test_stats_repository_stats_summary_aggregates_and_sorts_rows():
    calls = []
    repository = StatsRepository(
        _queued_session_factory(
            [
                _QueuedResult(
                    [
                        SimpleNamespace(provider="p-low", model="m", total=10, success_count=1),
                        SimpleNamespace(provider="p-high", model="m", total=10, success_count=9),
                    ]
                ),
                _QueuedResult(
                    [
                        SimpleNamespace(provider="p-low", total=10, success_count=1),
                        SimpleNamespace(provider="p-high", total=10, success_count=9),
                    ]
                ),
                _QueuedResult([SimpleNamespace(model="m-a", count=3)]),
                _QueuedResult([SimpleNamespace(endpoint="/v1/chat/completions", count=4)]),
                _QueuedResult([SimpleNamespace(client_ip="127.0.0.1", count=5)]),
            ],
            calls,
        )
    )

    summary = await repository.query_stats_summary(hours=24)

    assert summary["time_range"] == "Last 24 hours"
    assert summary["channel_model_success_rates"][0]["provider"] == "p-high"
    assert summary["channel_success_rates"][0]["provider"] == "p-high"
    assert summary["model_request_counts"] == [{"model": "m-a", "count": 3}]
    assert summary["endpoint_request_counts"] == [{"endpoint": "/v1/chat/completions", "count": 4}]
    assert summary["ip_request_counts"] == [{"ip": "127.0.0.1", "count": 5}]
    assert len(calls) == 5


async def test_stats_repository_compute_total_cost_coerces_database_scalar_to_float():
    calls = []
    repository = StatsRepository(_queued_session_factory([_QueuedResult(scalar="1.25")], calls))

    assert await repository.compute_total_cost(filter_api_key="sk-test") == 1.25
    assert calls
