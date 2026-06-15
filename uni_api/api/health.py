from __future__ import annotations

from typing import Any, Callable


async def healthz_response(version: str) -> dict[str, str]:
    return {"status": "ok", "version": version}


async def observability_runtime_response(
    runtime_gauges: Any,
    client_manager: Any = None,
    *,
    stream_cleanup_snapshot: Callable[[], dict[str, Any]] | None = None,
    provider_key_pools_snapshot: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    await runtime_gauges.record_event_loop_lag()
    snapshot = runtime_gauges.snapshot()
    if client_manager is not None and hasattr(client_manager, "snapshot"):
        snapshot["upstream_http_clients"] = client_manager.snapshot()
    if stream_cleanup_snapshot is not None:
        snapshot["stream_cleanup_tasks"] = stream_cleanup_snapshot()
    if provider_key_pools_snapshot is not None:
        snapshot["provider_key_pools"] = provider_key_pools_snapshot()
    return snapshot
