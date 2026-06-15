from __future__ import annotations

from typing import Any, Callable

from core.utils import get_model_dict


MODEL_INFO_CREATED = 1720524448858


def get_all_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    all_models: list[dict[str, Any]] = []
    unique_models: set[str] = set()

    for provider in config.get("providers", []) or []:
        model_dict = provider.get("_model_dict_cache") or get_model_dict(provider)
        for model in model_dict.keys():
            if model not in unique_models:
                unique_models.add(model)
                all_models.append(
                    {
                        "id": model,
                        "object": "model",
                        "created": MODEL_INFO_CREATED,
                        "owned_by": "uni-api",
                    }
                )

    return all_models


def post_all_models(api_index: int, config: dict[str, Any], api_list: list[str], models_list: dict[str, list[str]]) -> list[dict[str, Any]]:
    all_models: list[dict[str, Any]] = []
    unique_models: set[str] = set()

    if config["api_keys"][api_index]["model"]:
        for model in config["api_keys"][api_index]["model"]:
            if model == "all":
                return get_all_models(config)
            if "/" in model:
                provider = model.split("/")[0]
                model = model.split("/")[1]
                if model == "*":
                    if provider.startswith("sk-") and provider in api_list:
                        for model_item in models_list[provider]:
                            if model_item not in unique_models:
                                unique_models.add(model_item)
                                all_models.append(
                                    {
                                        "id": model_item,
                                        "object": "model",
                                        "created": MODEL_INFO_CREATED,
                                        "owned_by": "uni-api",
                                    }
                                )
                    else:
                        for provider_item in config["providers"]:
                            if provider_item["provider"] != provider:
                                continue
                            model_dict = provider_item.get("_model_dict_cache") or get_model_dict(provider_item)
                            for model_item in model_dict.keys():
                                if model_item not in unique_models:
                                    unique_models.add(model_item)
                                    all_models.append(
                                        {
                                            "id": model_item,
                                            "object": "model",
                                            "created": MODEL_INFO_CREATED,
                                            "owned_by": "uni-api",
                                        }
                                    )
                else:
                    if provider.startswith("sk-") and provider in api_list:
                        if model in models_list[provider] and model not in unique_models:
                            unique_models.add(model)
                            all_models.append(
                                {
                                    "id": model,
                                    "object": "model",
                                    "created": MODEL_INFO_CREATED,
                                    "owned_by": "uni-api",
                                }
                            )
                    else:
                        for provider_item in config["providers"]:
                            if provider_item["provider"] != provider:
                                continue
                            model_dict = provider_item.get("_model_dict_cache") or get_model_dict(provider_item)
                            for model_item in model_dict.keys():
                                if model_item not in unique_models and model_item == model:
                                    unique_models.add(model_item)
                                    all_models.append(
                                        {
                                            "id": model_item,
                                            "object": "model",
                                            "created": MODEL_INFO_CREATED,
                                            "owned_by": "uni-api",
                                        }
                                    )
                continue

            if model.startswith("sk-") and model in api_list:
                continue

            if model not in unique_models:
                unique_models.add(model)
                all_models.append(
                    {
                        "id": model,
                        "object": "model",
                        "created": MODEL_INFO_CREATED,
                        "owned_by": "uni-api",
                    }
                )

    return all_models


def list_models_payload(
    *,
    api_index: int,
    api_list: list[str],
    model_response_cache: dict[str, list[dict]],
    config: dict[str, Any],
    models_list: dict[str, list[str]],
    build_models: Callable[[int, dict, list[str], dict[str, list[str]]], list[dict]],
) -> dict[str, Any]:
    api_key = api_list[api_index] if 0 <= api_index < len(api_list) else None
    models = model_response_cache.get(api_key)
    if models is None:
        models = build_models(api_index, config, api_list, models_list)
    return {"object": "list", "data": models}
