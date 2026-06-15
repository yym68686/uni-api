"""Compatibility facade for legacy imports.

New code should import from the package modules under ``uni_api``.
This module only re-exports the old surface area used by the current suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import time

from core.utils import *  # noqa: F401,F403
from uni_api.api.models import get_all_models, post_all_models  # noqa: F401
import uni_api.config.legacy_loader as _legacy_loader
import uni_api.persistence.key_stats as _key_stats
from core.log_config import logger
from uni_api.config.schema import validate_config_data
from uni_api.rate_limit import RateLimitPolicy, RateLimitState
from uni_api.upstream.error_handling import error_handling_wrapper  # noqa: F401


API_YAML_PATH = _legacy_loader.API_YAML_PATH
yaml_error_message = _legacy_loader.yaml_error_message


class InMemoryRateLimiter:
    def __init__(self):
        self.state = RateLimitState(now_func=lambda: time())
        self.requests = self.state.requests

    async def is_rate_limited(self, key: str, limits) -> bool:
        policy = RateLimitPolicy.from_legacy_rules(limits or [(999999, 60)])
        return self.state.is_rate_limited(key, None, policy, commit=True)


def _sync_legacy_loader_globals() -> None:
    _legacy_loader.API_YAML_PATH = API_YAML_PATH
    _legacy_loader.validate_config_data = validate_config_data


def _sync_legacy_loader_error() -> None:
    global yaml_error_message
    yaml_error_message = _legacy_loader.yaml_error_message


def save_api_yaml(config_data):
    _sync_legacy_loader_globals()
    return _legacy_loader.save_api_yaml(config_data)


async def update_config(config_data, use_config_url=False):
    _sync_legacy_loader_globals()
    result = await _legacy_loader.update_config(config_data, use_config_url=use_config_url)
    _sync_legacy_loader_error()
    return result


async def load_config(app=None):
    _sync_legacy_loader_globals()
    result = await _legacy_loader.load_config(app)
    _sync_legacy_loader_error()
    return result


async def query_channel_key_stats(provider_name, start_dt=None, end_dt=None):
    return await _key_stats.query_channel_key_stats(
        provider_name,
        start_dt=start_dt,
        end_dt=end_dt,
    )


async def get_sorted_api_keys(provider_name: str, all_keys_in_config: list, group_size: int = 100):
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
