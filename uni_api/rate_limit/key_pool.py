from __future__ import annotations

import asyncio
import random
from typing import Any, Callable

from fastapi import HTTPException
from contextlib import suppress

from uni_api.rate_limit.policy import DEFAULT_RATE_LIMIT, RateLimitPolicy
from uni_api.rate_limit.state import RateLimitState


class ProviderKeyPool:
    def __init__(
        self,
        items: list[str] | None = None,
        rate_limit: Any = None,
        schedule_algorithm: str = "round_robin",
        provider_name: str | None = None,
        *,
        now_func: Callable[[], float] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ):
        items = list(items or [])
        rate_limit = {"default": DEFAULT_RATE_LIMIT} if rate_limit is None else rate_limit
        self.provider_name = provider_name
        self.original_items = list(items)
        self.schedule_algorithm = self._normalize_schedule_algorithm(schedule_algorithm, on_warning)
        self.items = random.sample(items, len(items)) if self.schedule_algorithm == "random" else items
        self.index = 0
        self.lock = asyncio.Lock()
        self.state = RateLimitState(now_func=now_func)
        self.policy = RateLimitPolicy.from_config(rate_limit)
        self.rate_limits = self.policy.as_legacy_dict()
        self.reordering_task = None
        self._warn = on_warning

        if self.schedule_algorithm == "smart_round_robin":
            self._trigger_reorder()

    @property
    def requests(self):
        return self.state.requests

    @property
    def cooling_until(self):
        return self.state.cooling_until

    async def reset_items(self, new_items: list[str]):
        async with self.lock:
            if self.items != new_items:
                self.items = list(new_items)
                self.index = 0

    def _trigger_reorder(self) -> None:
        if self.provider_name and (self.reordering_task is None or self.reordering_task.done()):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._log_warning(f"No running event loop to trigger reorder for '{self.provider_name}'.")
                return
            self.reordering_task = loop.create_task(self._reorder_keys())

    async def _reorder_keys(self) -> None:
        try:
            sorted_keys = await self._load_reordered_items()
            if sorted_keys:
                await self.reset_items(sorted_keys)
        except Exception as exc:
            self._log_warning(f"Error during key reordering for provider '{self.provider_name}': {exc}")

    async def _load_reordered_items(self) -> list[str] | None:
        return None

    async def set_cooling(self, item: str, cooling_time: int = 60):
        async with self.lock:
            self.state.set_cooling(item, cooling_time)
        self._log_warning(f"API key {item} 已进入冷却状态，冷却时间 {cooling_time} 秒")

    async def is_rate_limited(self, item: str, model: str | None = None, is_check: bool = False) -> bool:
        async with self.lock:
            limited = self.state.is_rate_limited(item, model, self.policy, commit=not is_check)
        if limited and not is_check:
            self._log_warning(f"API key {item}: model: {model or 'default'} has been rate limited")
        return limited

    def rollback_rate_limit_record(self, item: str, model: str | None = None) -> None:
        self.state.rollback_last_record(item, model)

    async def next(self, model: str | None = None):
        async with self.lock:
            if not self.items:
                self._log_warning("All API keys are rate limited!")
                raise HTTPException(status_code=429, detail="Too many requests")

            if self.schedule_algorithm == "fixed_priority":
                self.index = 0

            if self.schedule_algorithm == "smart_round_robin" and self.index == len(self.items) - 1:
                self._trigger_reorder()

            start_index = self.index
            while True:
                item = self.items[self.index]
                self.index = (self.index + 1) % len(self.items)

                if not self.state.is_rate_limited(item, model, self.policy, commit=True):
                    return item

                if self.index == start_index:
                    self._log_warning("All API keys are rate limited!")
                    raise HTTPException(status_code=429, detail="Too many requests")

    async def is_tpr_exceeded(self, model: str | None = None, tokens: int = 0) -> bool:
        async with self.lock:
            return self.state.is_tpr_exceeded(model, tokens, self.policy)

    async def is_all_rate_limited(self, model: str | None = None) -> bool:
        if not self.items:
            return False

        async with self.lock:
            for item in self.items:
                if not self.state.is_rate_limited(item, model, self.policy, commit=False):
                    return False
            return True

    async def after_next_current(self):
        if not self.items:
            return None
        async with self.lock:
            return self.items[(self.index - 1) % len(self.items)]

    def get_items_count(self) -> int:
        return len(self.items)

    def snapshot(self) -> dict[str, Any]:
        task = self.reordering_task
        return {
            "provider_name": self.provider_name,
            "item_count": len(self.items),
            "schedule_algorithm": self.schedule_algorithm,
            "reordering_task_active": bool(task and not task.done()),
            "reordering_task_done": bool(task and task.done()),
        }

    async def close(self) -> None:
        task = self.reordering_task
        if task is None:
            return
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self.reordering_task = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "items": list(self.items),
            "original_items": list(self.original_items),
            "schedule_algorithm": self.schedule_algorithm,
            "index": self.index,
            "rate_limits": self.rate_limits,
            "cooling_until": dict(self.cooling_until),
        }

    @staticmethod
    def _normalize_schedule_algorithm(
        schedule_algorithm: str,
        on_warning: Callable[[str], None] | None,
    ) -> str:
        allowed = {"round_robin", "random", "fixed_priority", "smart_round_robin"}
        if schedule_algorithm in allowed:
            return schedule_algorithm
        if on_warning is not None:
            on_warning(
                f"Unknown schedule algorithm: {schedule_algorithm}, use "
                "(round_robin, random, fixed_priority, smart_round_robin) instead"
            )
        return "round_robin"

    def _log_warning(self, message: str) -> None:
        if self._warn is not None:
            self._warn(message)
