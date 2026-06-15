from __future__ import annotations

from time import time

from uni_api.rate_limit import RateLimitPolicy, RateLimitState


class InMemoryRateLimiter:
    def __init__(self):
        self.state = RateLimitState(now_func=lambda: time())
        self.requests = self.state.requests

    async def is_rate_limited(self, key: str, limits) -> bool:
        policy = RateLimitPolicy.from_legacy_rules(limits or [(999999, 60)])
        return self.state.is_rate_limited(key, None, policy, commit=True)
