from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable


@dataclass(frozen=True, slots=True)
class PaidApiKeyState:
    credits: float
    created_at: datetime
    all_tokens_info: list[dict[str, Any]]
    total_cost: float
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "credits": self.credits,
            "created_at": self.created_at,
            "all_tokens_info": self.all_tokens_info,
            "total_cost": self.total_cost,
            "enabled": self.enabled,
        }


async def compute_paid_api_key_state(
    *,
    credits: float,
    created_at: datetime,
    paid_key: str,
    compute_total_cost: Callable[..., Awaitable[float]],
    get_usage_data: Callable[..., Awaitable[list[dict[str, Any]]]],
) -> tuple[PaidApiKeyState | None, float]:
    total_cost = await compute_total_cost(filter_api_key=paid_key, start_dt_obj=created_at)
    if credits == -1:
        return None, total_cost

    all_tokens_info = await get_usage_data(filter_api_key=paid_key, start_dt_obj=created_at)
    return (
        PaidApiKeyState(
            credits=credits,
            created_at=created_at,
            all_tokens_info=all_tokens_info,
            total_cost=total_cost,
            enabled=total_cost <= credits,
        ),
        total_cost,
    )

