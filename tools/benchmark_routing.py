from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.routing.core import (
    build_api_key_models_map,
    build_routing_index,
    get_right_order_providers,
    weighted_round_robin,
)


def build_config(provider_count: int, models_per_provider: int) -> tuple[dict, list[str], dict[str, list[str]]]:
    providers = []
    for provider_index in range(provider_count):
        provider_name = f"provider-{provider_index}"
        providers.append(
            {
                "provider": provider_name,
                "base_url": "https://example.com/v1/chat/completions",
                "api": f"upstream-{provider_index}",
                "model": [f"model-{model_index}" for model_index in range(models_per_provider)],
                "exclude_endpoints": ["/v1/responses/compact"] if provider_index % 8 == 0 else [],
            }
        )
    config = {
        "providers": providers,
        "api_keys": [
            {
                "api": "sk-bench",
                "model": [f"provider-{index}/*" for index in range(provider_count)],
                "weights": {
                    f"provider-{index}/model-0": (index % 5) + 1
                    for index in range(provider_count)
                },
            }
        ],
    }
    api_list = ["sk-bench"]
    models_list = build_api_key_models_map(config, api_list)
    return config, api_list, models_list


def time_call(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - start


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", type=int, default=100)
    parser.add_argument("--models-per-provider", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    config, api_list, models_list = build_config(args.providers, args.models_per_provider)
    routing_index = build_routing_index(config, api_list)
    weight_items = {
        f"provider-{index}": (index % 5) + 1
        for index in range(args.providers)
    }

    route_timings = []
    weighted_timings = []
    for _ in range(args.rounds):
        route_start = time.perf_counter()
        for _ in range(args.iterations):
            providers = await get_right_order_providers(
                "model-0",
                config,
                0,
                "fixed_priority",
                api_list,
                models_list,
                routing_index=routing_index,
                endpoint="/v1/chat/completions",
            )
            assert providers
        route_timings.append(time.perf_counter() - route_start)
        weighted_timings.append(time_call(lambda: weighted_round_robin(weight_items), args.iterations))

    print(
        json.dumps(
            {
                "providers": args.providers,
                "models_per_provider": args.models_per_provider,
                "iterations": args.iterations,
                "rounds": args.rounds,
                "routing_seconds_best": min(route_timings),
                "routing_seconds_mean": statistics.mean(route_timings),
                "weighted_seconds_best": min(weighted_timings),
                "weighted_seconds_mean": statistics.mean(weighted_timings),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

