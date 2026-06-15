from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import case, desc, func, select

from core.log_config import logger
from db import ChannelStat, RequestStat, async_session


def _clean_request_stat_fields(current_info: dict[str, Any]) -> dict[str, Any]:
    columns = [column.key for column in RequestStat.__table__.columns]
    filtered_info = {key: value for key, value in current_info.items() if key in columns}
    for key, value in filtered_info.items():
        if isinstance(value, str):
            filtered_info[key] = value.replace("\x00", "")
        elif key == "timing_spans" and isinstance(value, dict):
            filtered_info[key] = json.dumps(value, ensure_ascii=False, default=str)
    return filtered_info


class StatsRepository:
    def __init__(
        self,
        session_factory: Callable[..., Any] = async_session,
        *,
        semaphore: Any = None,
        debug: bool = False,
    ) -> None:
        self.session_factory = session_factory
        self.semaphore = semaphore
        self.debug = debug

    async def add_request_stat(self, current_info: dict[str, Any]) -> None:
        async def write() -> None:
            async with self.session_factory() as session:
                async with session.begin():
                    try:
                        session.add(RequestStat(**_clean_request_stat_fields(current_info)))
                        await session.commit()
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Error updating stats: %s", exc)
                        if self.debug:
                            import traceback

                            traceback.print_exc()

        if self.semaphore is None:
            await write()
            return
        async with self.semaphore:
            await write()

    async def add_channel_stat(
        self,
        *,
        request_id: str,
        provider: str,
        model: str,
        api_key: str,
        success: bool,
        provider_api_key: str | None = None,
    ) -> None:
        async def write() -> None:
            async with self.session_factory() as session:
                async with session.begin():
                    try:
                        session.add(
                            ChannelStat(
                                request_id=request_id,
                                provider=provider,
                                model=model,
                                api_key=api_key,
                                provider_api_key=provider_api_key,
                                success=success,
                            )
                        )
                        await session.commit()
                    except Exception as exc:
                        await session.rollback()
                        logger.error("Error updating channel stats: %s", exc)
                        if self.debug:
                            import traceback

                            traceback.print_exc()

        if self.semaphore is None:
            await write()
            return
        async with self.semaphore:
            await write()

    async def query_token_usage(
        self,
        *,
        filter_api_key: Optional[str] = None,
        filter_model: Optional[str] = None,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            query = select(
                RequestStat.api_key,
                RequestStat.model,
                func.sum(RequestStat.prompt_tokens).label("total_prompt_tokens"),
                func.sum(RequestStat.completion_tokens).label("total_completion_tokens"),
                func.sum(RequestStat.total_tokens).label("total_tokens"),
                func.count(RequestStat.id).label("request_count"),
            ).group_by(RequestStat.api_key, RequestStat.model)

            if filter_api_key:
                query = query.where(RequestStat.api_key == filter_api_key)
            if filter_model:
                query = query.where(RequestStat.model == filter_model)
            if start_dt:
                query = query.where(RequestStat.timestamp >= start_dt)
            if end_dt:
                query = query.where(RequestStat.timestamp < end_dt + timedelta(days=1))
            if not filter_model:
                query = query.where(RequestStat.model.isnot(None) & (RequestStat.model != ""))

            rows = (await session.execute(query)).mappings().all()

        processed_usage = []
        for row in rows:
            usage_dict = dict(row)
            api_key = usage_dict.get("api_key", "")
            if api_key and len(api_key) > 7:
                usage_dict["api_key_prefix"] = f"{api_key[:7]}...{api_key[-4:]}"
            else:
                usage_dict["api_key_prefix"] = api_key
            usage_dict.pop("api_key", None)
            processed_usage.append(usage_dict)
        return processed_usage

    async def query_channel_key_stats(
        self,
        *,
        provider_name: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            if not start_dt:
                start_dt = datetime.now(timezone.utc) - timedelta(hours=24)

            query = (
                select(
                    ChannelStat.provider_api_key,
                    func.count().label("total_requests"),
                    func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                )
                .where(ChannelStat.provider == provider_name)
                .where(ChannelStat.timestamp >= start_dt)
                .where(ChannelStat.provider_api_key.isnot(None))
            )
            if end_dt:
                query = query.where(ChannelStat.timestamp < end_dt)
            query = query.group_by(ChannelStat.provider_api_key)
            rows = (await session.execute(query)).mappings().all()

        key_stats = []
        for row in rows:
            total_requests = int(row.total_requests or 0)
            success_count = int(row.success_count or 0)
            key_stats.append(
                {
                    "api_key": row.provider_api_key,
                    "success_count": success_count,
                    "total_requests": total_requests,
                    "success_rate": success_count / total_requests if total_requests > 0 else 0,
                }
            )

        return sorted(
            key_stats,
            key=lambda item: (item["success_rate"], item["total_requests"]),
            reverse=True,
        )

    async def compute_total_cost(
        self,
        *,
        filter_api_key: Optional[str] = None,
        start_dt: Optional[datetime] = None,
    ) -> float:
        async with self.session_factory() as session:
            expr = (
                func.coalesce(RequestStat.prompt_tokens, 0)
                * func.coalesce(RequestStat.prompt_price, 0.3)
                + func.coalesce(RequestStat.completion_tokens, 0)
                * func.coalesce(RequestStat.completion_price, 1.0)
            ) / 1000000.0
            query = select(func.coalesce(func.sum(expr), 0.0))
            if filter_api_key:
                query = query.where(RequestStat.api_key == filter_api_key)
            if start_dt:
                query = query.where(RequestStat.timestamp >= start_dt)
            total_cost = (await session.execute(query)).scalar_one() or 0.0

        try:
            return float(total_cost)
        except Exception:
            return 0.0

    async def query_stats_summary(self, *, hours: int) -> dict[str, Any]:
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self.session_factory() as session:
            channel_model_rows = (
                await session.execute(
                    select(
                        ChannelStat.provider,
                        ChannelStat.model,
                        func.count().label("total"),
                        func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                    )
                    .where(ChannelStat.timestamp >= start_time)
                    .group_by(ChannelStat.provider, ChannelStat.model)
                )
            ).fetchall()

            channel_rows = (
                await session.execute(
                    select(
                        ChannelStat.provider,
                        func.count().label("total"),
                        func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                    )
                    .where(ChannelStat.timestamp >= start_time)
                    .group_by(ChannelStat.provider)
                )
            ).fetchall()

            model_rows = (
                await session.execute(
                    select(RequestStat.model, func.count().label("count"))
                    .where(RequestStat.timestamp >= start_time)
                    .group_by(RequestStat.model)
                    .order_by(desc("count"))
                )
            ).fetchall()

            endpoint_rows = (
                await session.execute(
                    select(RequestStat.endpoint, func.count().label("count"))
                    .where(RequestStat.timestamp >= start_time)
                    .group_by(RequestStat.endpoint)
                    .order_by(desc("count"))
                )
            ).fetchall()

            ip_rows = (
                await session.execute(
                    select(RequestStat.client_ip, func.count().label("count"))
                    .where(RequestStat.timestamp >= start_time)
                    .group_by(RequestStat.client_ip)
                    .order_by(desc("count"))
                )
            ).fetchall()

        def success_rate(row: Any) -> float:
            total = int(getattr(row, "total", 0) or 0)
            success_count = int(getattr(row, "success_count", 0) or 0)
            return success_count / total if total > 0 else 0

        return {
            "time_range": f"Last {hours} hours",
            "channel_model_success_rates": [
                {
                    "provider": row.provider,
                    "model": row.model,
                    "success_rate": success_rate(row),
                    "total_requests": row.total,
                }
                for row in sorted(channel_model_rows, key=success_rate, reverse=True)
            ],
            "channel_success_rates": [
                {
                    "provider": row.provider,
                    "success_rate": success_rate(row),
                    "total_requests": row.total,
                }
                for row in sorted(channel_rows, key=success_rate, reverse=True)
            ],
            "model_request_counts": [
                {
                    "model": row.model,
                    "count": row.count,
                }
                for row in model_rows
            ],
            "endpoint_request_counts": [
                {
                    "endpoint": row.endpoint,
                    "count": row.count,
                }
                for row in endpoint_rows
            ],
            "ip_request_counts": [
                {
                    "ip": row.client_ip,
                    "count": row.count,
                }
                for row in ip_rows
            ],
        }
