from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_serializer

from uni_api.persistence.repositories import StatsRepository


class TokenUsageEntry(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int


class QueryDetails(BaseModel):
    model_config = {"protected_namespaces": ()}

    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    api_key_filter: Optional[str] = None
    model_filter: Optional[str] = None
    credits: Optional[str] = None
    total_cost: Optional[str] = None
    balance: Optional[str] = None


class TokenUsageResponse(BaseModel):
    usage: list[TokenUsageEntry]
    query_details: QueryDetails


class ChannelKeyRanking(BaseModel):
    api_key: str
    success_count: int
    total_requests: int
    success_rate: float


class ChannelKeyRankingsResponse(BaseModel):
    rankings: list[ChannelKeyRanking]
    query_details: QueryDetails


class TokenInfo(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int


class ApiKeyState(BaseModel):
    credits: float
    created_at: datetime
    all_tokens_info: list[dict[str, Any]]
    total_cost: float
    enabled: bool

    @field_serializer("created_at")
    def serialize_dt(self, dt: datetime):
        return dt.isoformat()


class ApiKeysStatesResponse(BaseModel):
    api_keys_states: dict[str, ApiKeyState]


async def stats_summary_response(
    *,
    repository: StatsRepository,
    hours: int,
    database_disabled: bool,
) -> JSONResponse:
    if database_disabled:
        return JSONResponse(content={"stats": {}})
    return JSONResponse(content=await repository.query_stats_summary(hours=hours))


def parse_datetime_input(dt_input: str) -> datetime:
    try:
        return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
    except ValueError:
        try:
            if dt_input.endswith("Z"):
                dt_input = dt_input[:-1] + "+00:00"
            dt_obj = datetime.fromisoformat(dt_input)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        except ValueError:
            raise ValueError(
                f"Invalid datetime format: {dt_input}. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp."
            )


def resolve_time_range(
    *,
    default_days: int,
    last_n_days: Optional[int] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
) -> tuple[Optional[datetime], Optional[datetime], Optional[str], Optional[str]]:
    now = datetime.now(timezone.utc)
    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(status_code=400, detail="Cannot use last_n_days with start_datetime or end_datetime.")
        if last_n_days <= 0:
            raise HTTPException(status_code=400, detail="last_n_days must be positive.")
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now
        return (
            start_dt_obj,
            end_dt_obj,
            start_dt_obj.isoformat(timespec="seconds"),
            end_dt_obj.isoformat(timespec="seconds"),
        )

    if start_datetime or end_datetime:
        try:
            start_dt_obj = parse_datetime_input(start_datetime) if start_datetime else None
            end_dt_obj = parse_datetime_input(end_datetime) if end_datetime else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
            raise HTTPException(status_code=400, detail="end_datetime cannot be before start_datetime.")
        return (
            start_dt_obj,
            end_dt_obj,
            start_dt_obj.isoformat(timespec="seconds") if start_dt_obj else None,
            end_dt_obj.isoformat(timespec="seconds") if end_dt_obj else None,
        )

    start_dt_obj = now - timedelta(days=default_days)
    end_dt_obj = now
    return (
        start_dt_obj,
        end_dt_obj,
        start_dt_obj.isoformat(timespec="seconds"),
        end_dt_obj.isoformat(timespec="seconds"),
    )


async def token_usage_response(
    *,
    repository: StatsRepository,
    database_disabled: bool,
    config: dict[str, Any],
    admin_api_keys: list[str],
    api_index: int,
    api_key_param: Optional[str] = None,
    model: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
    update_paid_key_state: Callable[[str], Any],
) -> TokenUsageResponse:
    if database_disabled:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    requesting_token = ""
    api_keys = config.get("api_keys") if isinstance(config, dict) else None
    if isinstance(api_keys, list) and 0 <= int(api_index) < len(api_keys):
        requesting_token = str((api_keys[int(api_index)] or {}).get("api") or "")

    is_admin = requesting_token in (admin_api_keys or [])
    filter_api_key = api_key_param if is_admin and api_key_param else (None if is_admin else requesting_token)
    api_key_filter_detail = api_key_param if is_admin and api_key_param else ("all" if is_admin else "self")

    start_dt_obj, end_dt_obj, start_detail, end_detail = resolve_time_range(
        default_days=30,
        last_n_days=last_n_days,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    usage_data = await repository.query_token_usage(
        filter_api_key=filter_api_key,
        filter_model=model,
        start_dt=start_dt_obj,
        end_dt=end_dt_obj,
    )

    if filter_api_key:
        credits, total_cost = await update_paid_key_state(filter_api_key)
    else:
        credits, total_cost = None, None

    query_details = QueryDetails(
        start_datetime=start_detail,
        end_datetime=end_detail,
        api_key_filter=api_key_filter_detail,
        model_filter=model if model else "all",
        credits="$" + str(credits),
        total_cost="$" + str(total_cost),
        balance="$" + str(float(credits) - float(total_cost)) if credits and total_cost else None,
    )
    return TokenUsageResponse(
        usage=[TokenUsageEntry(**item) for item in usage_data],
        query_details=query_details,
    )


async def channel_key_rankings_response(
    *,
    repository: StatsRepository,
    database_disabled: bool,
    provider_name: str,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
) -> ChannelKeyRankingsResponse:
    if database_disabled:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    start_dt_obj, end_dt_obj, start_detail, end_detail = resolve_time_range(
        default_days=1,
        last_n_days=last_n_days,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    rankings_data = await repository.query_channel_key_stats(
        provider_name=provider_name,
        start_dt=start_dt_obj,
        end_dt=end_dt_obj,
    )
    return ChannelKeyRankingsResponse(
        rankings=[ChannelKeyRanking(**item) for item in rankings_data],
        query_details=QueryDetails(
            start_datetime=start_detail,
            end_datetime=end_detail,
            api_key_filter=provider_name,
        ),
    )


def api_keys_states_response(paid_api_keys_states: dict[str, Any]) -> ApiKeysStatesResponse:
    return ApiKeysStatesResponse(
        api_keys_states={
            key: ApiKeyState(
                credits=state["credits"],
                created_at=state["created_at"],
                all_tokens_info=state["all_tokens_info"],
                total_cost=state["total_cost"],
                enabled=state["enabled"],
            )
            for key, state in paid_api_keys_states.items()
        }
    )


def add_credits_response(*, paid_api_keys_states: dict[str, Any], paid_key: str, amount: float) -> JSONResponse:
    if paid_key not in paid_api_keys_states:
        raise HTTPException(status_code=404, detail=f"API key '{paid_key}' not found in paid API keys states.")

    paid_api_keys_states[paid_key]["credits"] += float(amount)
    current_credits = paid_api_keys_states[paid_key]["credits"]
    total_cost = paid_api_keys_states[paid_key]["total_cost"]
    paid_api_keys_states[paid_key]["enabled"] = current_credits >= total_cost

    return JSONResponse(
        content={
            "message": f"Successfully added {amount} credits to API key '{paid_key}'.",
            "paid_key": paid_key,
            "new_credits": current_credits,
            "enabled": paid_api_keys_states[paid_key]["enabled"],
        }
    )
