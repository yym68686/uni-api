from __future__ import annotations

import argparse
import json
import random
import statistics
import time


def _time_call(fn, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-keys", type=int, default=10000)
    parser.add_argument("--lookups", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_keys = [f"sk-{index:08d}" for index in range(args.api_keys)]
    random.seed(args.seed)
    lookup_keys = [api_keys[random.randrange(args.api_keys)] for _ in range(args.lookups)]
    api_key_index = {api_key: index for index, api_key in enumerate(api_keys)}

    def list_index_lookup() -> None:
        for api_key in lookup_keys:
            api_keys.index(api_key)

    def dict_lookup() -> None:
        for api_key in lookup_keys:
            api_key_index[api_key]

    list_timings = [_time_call(list_index_lookup, 1) for _ in range(args.rounds)]
    dict_timings = [_time_call(dict_lookup, 1) for _ in range(args.rounds)]
    list_best = min(list_timings)
    dict_best = min(dict_timings)

    print(
        json.dumps(
            {
                "api_keys": args.api_keys,
                "lookups": args.lookups,
                "rounds": args.rounds,
                "list_index_seconds_best": list_best,
                "list_index_seconds_mean": statistics.mean(list_timings),
                "dict_lookup_seconds_best": dict_best,
                "dict_lookup_seconds_mean": statistics.mean(dict_timings),
                "best_speedup": list_best / dict_best if dict_best else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
