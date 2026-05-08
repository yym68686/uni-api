import asyncio
import inspect
import json
import random
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException

from core.log_config import logger
from core.models import RequestModel
from core.utils import (
    circular_list_encoder,
    get_model_dict,
    provider_api_circular_list,
    safe_get,
)
from utils import post_all_models


def _provider_model_dict(provider: dict) -> dict:
    return provider.get("_model_dict_cache") or get_model_dict(provider)


def _normalize_endpoint_path(endpoint: Optional[str]) -> str:
    if endpoint is None:
        return ""
    endpoint_path = str(endpoint).strip()
    if not endpoint_path:
        return ""
    endpoint_path = endpoint_path.rstrip("/")
    if not endpoint_path.startswith("/"):
        endpoint_path = f"/{endpoint_path}"
    return endpoint_path or "/"


def _endpoint_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _provider_excludes_endpoint(provider: dict, endpoint: Optional[str]) -> bool:
    normalized_endpoint = _normalize_endpoint_path(endpoint)
    if not normalized_endpoint:
        return False

    excluded_endpoints = []
    excluded_endpoints.extend(_endpoint_list(provider.get("exclude_endpoints")))
    excluded_endpoints.extend(_endpoint_list(safe_get(provider, "preferences", "exclude_endpoints")))

    return any(
        _normalize_endpoint_path(excluded_endpoint) == normalized_endpoint
        for excluded_endpoint in excluded_endpoints
    )


def weighted_round_robin(weights: dict[str, int]) -> list[str]:
    provider_names = list(weights.keys())
    current_weights = {name: 0 for name in provider_names}
    num_selections = total_weight = sum(weights.values())
    weighted_provider_list = []

    for _ in range(num_selections):
        max_ratio = -1
        selected_provider = None

        for name in provider_names:
            current_weights[name] += weights[name]
            ratio = current_weights[name] / weights[name]

            if ratio > max_ratio:
                max_ratio = ratio
                selected_provider = name

        weighted_provider_list.append(selected_provider)
        current_weights[selected_provider] -= total_weight

    return weighted_provider_list


def lottery_scheduling(weights: dict[str, int]) -> list[str]:
    total_tickets = sum(weights.values())
    selections = []
    for _ in range(total_tickets):
        ticket = random.randint(1, total_tickets)
        cumulative = 0
        for provider, weight in weights.items():
            cumulative += weight
            if ticket <= cumulative:
                selections.append(provider)
                break
    return selections


async def get_provider_rules(
    model_rule: Any,
    config: dict,
    request_model: str,
    api_list: list[str],
    models_list: dict[str, list[str]],
) -> list[str]:
    provider_rules = []
    if model_rule == "all":
        for provider in config["providers"]:
            model_dict = _provider_model_dict(provider)
            for model in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model)

    elif isinstance(model_rule, str) and "/" in model_rule:
        if model_rule.startswith("<") and model_rule.endswith(">"):
            model_rule = model_rule[1:-1]
            for provider in config["providers"]:
                model_dict = _provider_model_dict(provider)
                if model_rule in model_dict.keys():
                    provider_rules.append(provider["provider"] + "/" + model_rule)
        else:
            provider_name = model_rule.split("/")[0]
            model_name_split = "/".join(model_rule.split("/")[1:])
            available_models: list[str] = []

            if provider_name.startswith("sk-") and provider_name in api_list:
                available_models = list(models_list.get(provider_name, []))
            else:
                for provider in config["providers"]:
                    model_dict = _provider_model_dict(provider)
                    if provider["provider"] == provider_name:
                        available_models.extend(list(model_dict.keys()))

            if model_name_split == "*":
                if request_model in available_models:
                    provider_rules.append(provider_name + "/" + request_model)

                for available_model in available_models:
                    if request_model.endswith("*") and available_model.startswith(request_model.rstrip("*")):
                        provider_rules.append(provider_name + "/" + available_model)
            elif model_name_split == request_model or (
                request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*"))
            ):
                if model_name_split in available_models:
                    provider_rules.append(provider_name + "/" + model_name_split)

    else:
        for provider in config["providers"]:
            model_dict = _provider_model_dict(provider)
            if model_rule in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model_rule)

    return provider_rules


def get_provider_list(
    provider_rules: list[str],
    config: dict,
    request_model: str,
    api_list: list[str],
) -> list[dict]:
    provider_list = []
    for item in provider_rules:
        provider_name = item.split("/")[0]
        if provider_name.startswith("sk-") and provider_name in api_list:
            provider_list.append(
                {
                    "provider": provider_name,
                    "base_url": "http://127.0.0.1:8000/v1/chat/completions",
                    "model": [{request_model: request_model}],
                    "tools": True,
                    "_model_dict_cache": {request_model: request_model},
                }
            )
            continue

        for provider in config["providers"]:
            model_dict = _provider_model_dict(provider)
            if not model_dict:
                continue

            model_name_split = "/".join(item.split("/")[1:])
            if "/" not in item or provider["provider"] != provider_name or model_name_split not in model_dict.keys():
                continue

            if request_model in model_dict.keys() and model_name_split == request_model:
                new_provider = {
                    "provider": provider["provider"],
                    "base_url": provider.get("base_url", ""),
                    "api": provider.get("api", None),
                    "model": [{model_dict[model_name_split]: request_model}],
                    "preferences": provider.get("preferences", {}),
                    "tools": provider.get("tools", False),
                    "_model_dict_cache": model_dict,
                    "project_id": provider.get("project_id", None),
                    "private_key": provider.get("private_key", None),
                    "client_email": provider.get("client_email", None),
                    "cf_account_id": provider.get("cf_account_id", None),
                    "aws_access_key": provider.get("aws_access_key", None),
                    "aws_secret_key": provider.get("aws_secret_key", None),
                    "engine": provider.get("engine", None),
                    "exclude_endpoints": provider.get("exclude_endpoints", []),
                }
                provider_list.append(new_provider)
            elif request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*")):
                new_provider = {
                    "provider": provider["provider"],
                    "base_url": provider.get("base_url", ""),
                    "api": provider.get("api", None),
                    "model": [{model_dict[model_name_split]: request_model}],
                    "preferences": provider.get("preferences", {}),
                    "tools": provider.get("tools", False),
                    "_model_dict_cache": model_dict,
                    "project_id": provider.get("project_id", None),
                    "private_key": provider.get("private_key", None),
                    "client_email": provider.get("client_email", None),
                    "cf_account_id": provider.get("cf_account_id", None),
                    "aws_access_key": provider.get("aws_access_key", None),
                    "aws_secret_key": provider.get("aws_secret_key", None),
                    "engine": provider.get("engine", None),
                    "exclude_endpoints": provider.get("exclude_endpoints", []),
                }
                provider_list.append(new_provider)

    return provider_list


async def get_matching_providers(
    request_model: str,
    config: dict,
    api_index: int,
    api_list: list[str],
    models_list: dict[str, list[str]],
) -> list[dict]:
    provider_rules = []
    for model_rule in config["api_keys"][api_index]["model"]:
        provider_rules.extend(await get_provider_rules(model_rule, config, request_model, api_list, models_list))

    return get_provider_list(provider_rules, config, request_model, api_list)


async def get_right_order_providers(
    request_model: str,
    config: dict,
    api_index: int,
    scheduling_algorithm: str,
    api_list: list[str],
    models_list: dict[str, list[str]],
    *,
    endpoint: Optional[str] = None,
    channel_manager=None,
    request_total_tokens: Optional[int] = None,
    debug: bool = False,
) -> list[dict]:
    matching_providers = await get_matching_providers(request_model, config, api_index, api_list, models_list)
    if endpoint:
        matching_providers = [
            provider
            for provider in matching_providers
            if not _provider_excludes_endpoint(provider, endpoint)
        ]

    if request_total_tokens and matching_providers:
        available_providers = []
        for provider in matching_providers:
            model_dict = _provider_model_dict(provider)
            original_model = model_dict[request_model]
            provider_name = provider["provider"]
            if provider_name.startswith("sk-") and provider_name in api_list:
                available_providers.append(provider)
                continue

            is_tpr_exceeded = await provider_api_circular_list[provider_name].is_tpr_exceeded(
                original_model,
                tokens=request_total_tokens,
            )
            if is_tpr_exceeded:
                continue
            available_providers.append(provider)

        matching_providers = available_providers

        if not matching_providers:
            raise HTTPException(
                status_code=413,
                detail=f"The request body is too long, No available providers at the moment: {request_model}",
            )

    if not matching_providers:
        raise HTTPException(status_code=404, detail=f"No available providers at the moment: {request_model}")

    num_matching_providers = len(matching_providers)
    if channel_manager and channel_manager.cooldown_period > 0 and num_matching_providers > 1:
        matching_providers = await channel_manager.get_available_providers(matching_providers)
        num_matching_providers = len(matching_providers)
        if not matching_providers:
            raise HTTPException(status_code=503, detail="No available providers at the moment")

    if scheduling_algorithm == "random":
        matching_providers = random.sample(matching_providers, num_matching_providers)

    weights = safe_get(config, "api_keys", api_index, "weights")
    if weights:
        intersection = None
        all_providers = set(provider["provider"] + "/" + request_model for provider in matching_providers)
        if all_providers:
            weight_keys = set(weights.keys())
            provider_rules = []
            for model_rule in weight_keys:
                provider_rules.extend(
                    await get_provider_rules(model_rule, config, request_model, api_list, models_list)
                )
            provider_list = get_provider_list(provider_rules, config, request_model, api_list)
            weight_keys = set(provider["provider"] + "/" + request_model for provider in provider_list)
            intersection = all_providers.intersection(weight_keys)
            if len(intersection) == 1:
                intersection = None

        if intersection:
            filtered_weights = {
                key.split("/")[0]: value
                for key, value in weights.items()
                if key.split("/")[0] + "/" + request_model in intersection
            }

            if scheduling_algorithm == "weighted_round_robin":
                weighted_provider_name_list = weighted_round_robin(filtered_weights)
            elif scheduling_algorithm == "lottery":
                weighted_provider_name_list = lottery_scheduling(filtered_weights)
            else:
                weighted_provider_name_list = list(filtered_weights.keys())

            new_matching_providers = []
            for provider_name in weighted_provider_name_list:
                for provider in matching_providers:
                    if provider["provider"] == provider_name:
                        new_matching_providers.append(provider)
            matching_providers = new_matching_providers

    if debug:
        for provider in matching_providers:
            logger.info(
                "available provider: %s",
                json.dumps(provider, indent=4, ensure_ascii=False, default=circular_list_encoder),
            )

    return matching_providers


def estimate_request_total_tokens(request_data: Any) -> int:
    request_total_tokens = 0
    if request_data and isinstance(request_data, RequestModel):
        for message in request_data.messages:
            if message.content and isinstance(message.content, str):
                request_total_tokens += len(message.content)
    return int(request_total_tokens / 4)


def build_api_key_models_map(config: dict, api_list: list[str]) -> dict[str, list[str]]:
    api_key_to_index = {api_key: index for index, api_key in enumerate(api_list)}
    resolved: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def resolve(api_key: str) -> list[str]:
        if api_key in resolved:
            return resolved[api_key]
        if api_key in visiting:
            logger.warning("Detected recursive api key model dependency for '%s'", api_key[:8])
            resolved[api_key] = []
            return resolved[api_key]

        api_index = api_key_to_index.get(api_key)
        if api_index is None:
            return []

        visiting.add(api_key)
        model_rules = safe_get(config, "api_keys", api_index, "model", default=[]) or []
        for model_rule in model_rules:
            provider_name = None
            if isinstance(model_rule, dict):
                provider_name = next(iter(model_rule.keys()), "").split("/", 1)[0]
            elif isinstance(model_rule, str):
                if "/" in model_rule:
                    provider_name = model_rule.split("/", 1)[0]
                elif model_rule.startswith("sk-"):
                    provider_name = model_rule

            if provider_name and provider_name.startswith("sk-") and provider_name in api_key_to_index and provider_name != api_key:
                resolve(provider_name)

        resolved[api_key] = [
            model_info["id"]
            for model_info in post_all_models(api_index, config, api_list, resolved)
        ]
        visiting.remove(api_key)
        return resolved[api_key]

    for api_key in api_list:
        resolve(api_key)

    return resolved


async def select_provider_api_key_raw(
    provider: dict,
    original_model: str,
    api_list: list[str],
) -> Optional[str]:
    provider_name = provider["provider"]
    if provider_name.startswith("sk-") and provider_name in api_list:
        return provider_name
    if provider.get("api"):
        return await provider_api_circular_list[provider_name].next(original_model)
    return None


async def compute_start_index(
    last_provider_indices: dict,
    locks: dict,
    request_model_name: str,
    scheduling_algorithm: str,
    num_matching_providers: int,
) -> int:
    start_index = 0
    if scheduling_algorithm != "fixed_priority" and num_matching_providers > 1:
        async with locks[request_model_name]:
            last_provider_indices[request_model_name] = (
                last_provider_indices[request_model_name] + 1
            ) % num_matching_providers
            start_index = last_provider_indices[request_model_name]
    return start_index


async def _call_provider_resolver(
    resolver,
    request_model_name: str,
    config: dict,
    api_index: int,
    scheduling_algorithm: str,
    *,
    api_list: list[str],
    models_list: dict[str, list[str]],
    endpoint: Optional[str] = None,
    channel_manager=None,
    request_total_tokens: int = 0,
    debug: bool = False,
) -> list[dict]:
    params = inspect.signature(resolver).parameters
    if "api_list" in params:
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        resolver_kwargs = {}
        if "endpoint" in params or accepts_kwargs:
            resolver_kwargs["endpoint"] = endpoint
        if "channel_manager" in params or accepts_kwargs:
            resolver_kwargs["channel_manager"] = channel_manager
        if "request_total_tokens" in params or accepts_kwargs:
            resolver_kwargs["request_total_tokens"] = request_total_tokens
        if "debug" in params or accepts_kwargs:
            resolver_kwargs["debug"] = debug

        return await resolver(
            request_model_name,
            config,
            api_index,
            scheduling_algorithm,
            api_list,
            models_list,
            **resolver_kwargs,
        )
    return await resolver(request_model_name, config, api_index, scheduling_algorithm)


def compute_retry_count(matching_providers: list[dict]) -> int:
    num_matching_providers = len(matching_providers)
    if num_matching_providers <= 0:
        return 0

    if num_matching_providers == 1:
        provider_name = safe_get(matching_providers, 0, "provider", default=None)
        if provider_name:
            count = provider_api_circular_list[provider_name].get_items_count()
            if count > 1:
                return count

    tmp_retry_count = sum(
        provider_api_circular_list[provider["provider"]].get_items_count()
        for provider in matching_providers
    ) * 2
    return tmp_retry_count if tmp_retry_count < 10 else 10


@dataclass
class ProviderAttempt:
    provider: dict
    provider_name: str
    original_model: str


@dataclass
class RoutingPlan:
    app: Any
    request_model_name: str
    api_index: int
    request_total_tokens: int
    scheduling_algorithm: str
    auto_retry: Any
    role: str
    matching_providers: list[dict]
    num_matching_providers: int
    start_index: int
    retry_count: int
    provider_resolver: Any
    index: int = 0
    status_code: int = 500
    error_message: Optional[str] = None

    @classmethod
    async def create(
        cls,
        app: Any,
        request_model_name: str,
        api_index: int,
        last_provider_indices: dict,
        locks: dict,
        *,
        endpoint: Optional[str] = None,
        request_total_tokens: int = 0,
        debug: bool = False,
        provider_resolver=None,
    ) -> "RoutingPlan":
        config = app.state.config
        api_list = getattr(app.state, "api_list", None) or [
            item.get("api")
            for item in config.get("api_keys", [])
            if item.get("api")
        ]
        models_list = getattr(app.state, "models_list", None) or build_api_key_models_map(config, api_list)
        api_key_preferences = safe_get(
            config,
            "api_keys",
            api_index,
            "preferences",
            default={},
        ) or {}
        scheduling_algorithm = safe_get(
            config,
            "api_keys",
            api_index,
            "preferences",
            "SCHEDULING_ALGORITHM",
            default="fixed_priority",
        )
        resolver = provider_resolver or get_right_order_providers
        matching_providers = await _call_provider_resolver(
            resolver,
            request_model_name,
            config,
            api_index,
            scheduling_algorithm,
            api_list=api_list,
            models_list=models_list,
            endpoint=endpoint,
            channel_manager=getattr(app.state, "channel_manager", None),
            request_total_tokens=request_total_tokens,
            debug=debug,
        )
        start_index = await compute_start_index(
            last_provider_indices,
            locks,
            request_model_name,
            scheduling_algorithm,
            len(matching_providers),
        )
        return cls(
            app=app,
            request_model_name=request_model_name,
            api_index=api_index,
            request_total_tokens=request_total_tokens,
            scheduling_algorithm=scheduling_algorithm,
            auto_retry=api_key_preferences["AUTO_RETRY"]
            if isinstance(api_key_preferences, dict) and "AUTO_RETRY" in api_key_preferences
            else True,
            role=safe_get(
                config,
                "api_keys",
                api_index,
                "role",
                default=safe_get(config, "api_keys", api_index, "api", default="None")[:8],
            ),
            matching_providers=matching_providers,
            num_matching_providers=len(matching_providers),
            start_index=start_index,
            retry_count=compute_retry_count(matching_providers),
            provider_resolver=resolver,
        )

    async def next_provider(self) -> Optional[ProviderAttempt]:
        while self.index <= self.num_matching_providers + self.retry_count:
            current_index = (self.start_index + self.index) % self.num_matching_providers
            self.index += 1
            provider = self.matching_providers[current_index]
            provider_name = provider["provider"]
            original_model = provider["_model_dict_cache"][self.request_model_name]

            if await provider_api_circular_list[provider_name].is_all_rate_limited(original_model):
                self.status_code = 429
                self.error_message = "All API keys are rate limited and stop auto retry!"
                if self.num_matching_providers == 1:
                    return None
                continue

            return ProviderAttempt(
                provider=provider,
                provider_name=provider_name,
                original_model=original_model,
            )

        return None

    async def refresh_matching_providers(self, *, debug: bool = False) -> None:
        config = self.app.state.config
        api_list = getattr(self.app.state, "api_list", None) or [
            item.get("api")
            for item in config.get("api_keys", [])
            if item.get("api")
        ]
        models_list = getattr(self.app.state, "models_list", None) or build_api_key_models_map(config, api_list)
        matching_providers = await _call_provider_resolver(
            self.provider_resolver,
            self.request_model_name,
            config,
            self.api_index,
            self.scheduling_algorithm,
            api_list=api_list,
            models_list=models_list,
            channel_manager=getattr(self.app.state, "channel_manager", None),
            request_total_tokens=self.request_total_tokens,
            debug=debug,
        )
        last_num_matching_providers = self.num_matching_providers
        self.matching_providers = matching_providers
        self.num_matching_providers = len(matching_providers)
        self.retry_count = compute_retry_count(matching_providers)
        if self.num_matching_providers != last_num_matching_providers:
            self.index = 0
