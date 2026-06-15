from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from time import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from core.log_config import logger
from core.utils import get_proxy


class ClientPool:
    def __init__(
        self,
        pool_size: int = 100,
        *,
        sweep_client: Callable[[httpx.AsyncClient], Awaitable[int]] | None = None,
        current_trace: Callable[[], Any] | None = None,
        begin_upstream_pool: Callable[[Any], Any] | None = None,
        end_upstream_pool: Callable[[], Any] | None = None,
    ) -> None:
        self.pool_size = pool_size
        self.clients: dict[str, httpx.AsyncClient] = {}
        self._client_locks = defaultdict(asyncio.Lock)
        self._maintenance_task: Optional[asyncio.Task] = None
        self._last_sweep_closed_connections = 0
        self._last_sweep_error: Optional[str] = None
        self._last_sweep_at: Optional[datetime] = None
        self._sweep_client = sweep_client
        self._current_trace = current_trace or (lambda: None)
        self._begin_upstream_pool = begin_upstream_pool or (lambda trace: None)
        self._end_upstream_pool = end_upstream_pool or (lambda: None)

    async def init(self, default_config: dict[str, Any]) -> None:
        self.default_config = default_config
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            await self.sweep_idle_connections()

    async def sweep_idle_connections(self) -> int:
        closed = 0
        errors: list[str] = []
        if self._sweep_client is None:
            return 0
        for key, client in list(self.clients.items()):
            try:
                closed += await self._sweep_client(client)
            except Exception as exc:
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Failed to sweep upstream HTTP client idle connections: key=%s",
                    key,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        self._last_sweep_closed_connections = closed
        self._last_sweep_error = "; ".join(errors)[:512] if errors else None
        self._last_sweep_at = datetime.now(timezone.utc)
        return closed

    def snapshot(self) -> dict[str, Any]:
        return {
            "client_count": len(self.clients),
            "pool_size": self.pool_size,
            "last_sweep_closed_connections": self._last_sweep_closed_connections,
            "last_sweep_at": self._last_sweep_at.isoformat() if self._last_sweep_at else None,
            "last_sweep_error": self._last_sweep_error,
        }

    @asynccontextmanager
    async def get_client(self, base_url: str, proxy: str | None = None, http2: Optional[bool] = None):
        trace = self._current_trace()
        if trace is not None:
            trace.mark("client_pool_acquire_start")
        self._begin_upstream_pool(trace)
        acquire_started_at = time()
        acquired = False
        client_key = self._client_key(base_url, proxy, http2)

        if client_key not in self.clients:
            async with self._client_locks[client_key]:
                if client_key not in self.clients:
                    timeout = httpx.Timeout(
                        connect=15.0,
                        read=None,
                        write=30.0,
                        pool=self.pool_size,
                    )
                    limits = httpx.Limits(max_connections=self.pool_size)
                    client_config = {
                        **self.default_config,
                        "timeout": timeout,
                        "limits": limits,
                    }
                    client_config = get_proxy(proxy, client_config)
                    if http2 is not None:
                        client_config["http2"] = bool(http2)
                    self.clients[client_key] = httpx.AsyncClient(**client_config)

        try:
            acquired = True
            if trace is not None:
                trace.mark("client_pool_acquire_end")
                trace.add_ms("upstream_pool_wait_ms", (time() - acquire_started_at) * 1000)
            self._end_upstream_pool()
            yield self.clients[client_key]
        finally:
            if not acquired:
                self._end_upstream_pool()

    async def close(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

    @staticmethod
    def _client_key(base_url: str, proxy: str | None, http2: Optional[bool]) -> str:
        parsed_url = urlparse(base_url)
        client_key = f"{parsed_url.netloc}"
        if proxy:
            client_key += f"_{proxy.replace('socks5h://', 'socks5://')}"
        if http2 is not None:
            client_key += f"_http2_{int(bool(http2))}"
        return client_key

