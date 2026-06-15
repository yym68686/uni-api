from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.log_config import logger
from db import DISABLE_DATABASE, async_session
from uni_api.persistence.repositories import StatsRepository


stats_repository = StatsRepository(async_session)


async def query_channel_key_stats(
    provider_name: str,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    if DISABLE_DATABASE:
        return []
    return await stats_repository.query_channel_key_stats(
        provider_name=provider_name,
        start_dt=start_dt,
        end_dt=end_dt,
    )


async def get_sorted_api_keys(
    provider_name: str,
    all_keys_in_config: list,
    group_size: int = 100,
) -> list:
    if not all_keys_in_config:
        return []

    key_stats = {}
    try:
        start_time = datetime.now(timezone.utc) - timedelta(hours=72)
        stats_list = await query_channel_key_stats(provider_name, start_dt=start_time)
        for stat in stats_list:
            key_stats[stat["api_key"]] = {
                "success_rate": stat["success_rate"],
                "total_requests": stat["total_requests"],
            }
    except Exception as exc:
        logger.error("Error querying key stats from DB for provider %r: %s", provider_name, exc)
        return all_keys_in_config

    sorted_keys = sorted(
        all_keys_in_config,
        key=lambda key: (
            key_stats.get(key, {"success_rate": -1})["success_rate"],
            key_stats.get(key, {"total_requests": 0})["total_requests"],
        ),
        reverse=True,
    )

    num_keys = len(sorted_keys)
    if num_keys == 0:
        return []

    num_groups = (num_keys + group_size - 1) // group_size
    groups = [[] for _ in range(num_groups)]

    for index, key in enumerate(sorted_keys):
        groups[index % num_groups].append(key)

    final_sorted_list = []
    for group in groups:
        final_sorted_list.extend(group)

    logger.info(
        "Successfully sorted %s keys for provider %r using smart algorithm.",
        len(final_sorted_list),
        provider_name,
    )
    return final_sorted_list
