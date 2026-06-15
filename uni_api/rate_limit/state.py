from __future__ import annotations

from collections import defaultdict, deque
from time import time
from typing import Callable

from uni_api.rate_limit.policy import RateLimitPolicy, TPR_PERIOD


class RateLimitState:
    def __init__(self, now_func: Callable[[], float] | None = None):
        self._now = now_func or time
        self.requests = defaultdict(lambda: defaultdict(lambda: defaultdict(deque)))
        self.cooling_until = defaultdict(float)

    def set_cooling(self, item: str | None, cooling_time: int = 60) -> None:
        if item is None:
            return
        self.cooling_until[item] = self._now() + cooling_time

    def is_rate_limited(
        self,
        item: str,
        model: str | None,
        policy: RateLimitPolicy,
        *,
        commit: bool,
    ) -> bool:
        now = self._now()
        if now < self.cooling_until[item]:
            return True

        model_key = model or "default"
        rules = policy.rules_for(model)
        request_windows = self.requests[item][model_key]
        positive_periods = {rule.period_seconds for rule in rules if rule.period_seconds > 0}

        for period in positive_periods:
            cutoff = now - period
            window = request_windows[period]
            while window and window[0] <= cutoff:
                window.popleft()

        for rule in rules:
            if rule.period_seconds <= 0:
                continue
            if len(request_windows[rule.period_seconds]) >= rule.count:
                return True

        if commit:
            for period in positive_periods:
                request_windows[period].append(now)
        return False

    def rollback_last_record(self, item: str, model: str | None = None) -> None:
        model_key = model or "default"
        request_windows = self.requests[item][model_key]
        for window in request_windows.values():
            if window:
                window.pop()

    def is_tpr_exceeded(
        self,
        model: str | None,
        tokens: int,
        policy: RateLimitPolicy,
    ) -> bool:
        if not tokens:
            return False

        for rule in policy.rules_for(model):
            if rule.period_seconds == TPR_PERIOD and tokens > rule.count:
                return True
        return False
