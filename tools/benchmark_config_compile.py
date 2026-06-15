from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.config.compiler import compile_runtime_config


def build_config(provider_count: int, models_per_provider: int, api_key_count: int) -> tuple[dict, list[str]]:
    providers = []
    model_rules = []
    for provider_index in range(provider_count):
        provider_name = f"provider-{provider_index}"
        models = [
            f"model-{provider_index}-{model_index}"
            for model_index in range(models_per_provider)
        ]
        providers.append(
            {
                "provider": provider_name,
                "base_url": "https://example.com/v1/chat/completions",
                "api": f"upstream-{provider_index}",
                "model": models,
                "exclude_endpoints": ["v1/responses/compact"] if provider_index % 10 == 0 else [],
                "preferences": {
                    "model_timeout": {"default": 30},
                    "api_key_rate_limit": {"default": "999999/min"},
                },
            }
        )
        model_rules.append(f"{provider_name}/*")

    api_keys = [
        {
            "api": f"sk-{api_key_index}",
            "model": list(model_rules),
            "weights": {f"provider-{api_key_index % provider_count}/model-{api_key_index % provider_count}-0": 2},
        }
        for api_key_index in range(api_key_count)
    ]

    return {"providers": providers, "api_keys": api_keys}, [item["api"] for item in api_keys]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", type=int, default=100)
    parser.add_argument("--models-per-provider", type=int, default=100)
    parser.add_argument("--api-keys", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    config, api_list = build_config(args.providers, args.models_per_provider, args.api_keys)
    timings = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        runtime = compile_runtime_config(config, api_list)
        elapsed = time.perf_counter() - start
        assert len(runtime.provider_by_name) == args.providers
        assert len(runtime.api_key_by_token) == args.api_keys
        timings.append(elapsed)

    print(
        json.dumps(
            {
                "providers": args.providers,
                "models_per_provider": args.models_per_provider,
                "api_keys": args.api_keys,
                "rounds": args.rounds,
                "seconds_best": min(timings),
                "seconds_mean": statistics.mean(timings),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
