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

from uni_api.upstream.client_pool import ClientPool


async def run_once(concurrency: int, acquisitions: int) -> float:
    pool = ClientPool(pool_size=100)
    await pool.init({"headers": {}, "http2": True, "verify": True, "follow_redirects": True})

    async def worker(worker_id: int) -> None:
        for index in range(acquisitions):
            http2 = (index + worker_id) % 2 == 0
            proxy = None if index % 3 else "socks5h://127.0.0.1:1080"
            async with pool.get_client("https://example.com/v1/chat/completions", proxy=proxy, http2=http2) as client:
                assert client is not None

    started = time.perf_counter()
    await asyncio.gather(*(worker(worker_id) for worker_id in range(concurrency)))
    elapsed = time.perf_counter() - started
    assert pool.snapshot()["client_count"] <= 4
    await pool.close()
    return elapsed


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--acquisitions", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    timings = [
        await run_once(args.concurrency, args.acquisitions)
        for _ in range(args.rounds)
    ]
    print(
        json.dumps(
            {
                "concurrency": args.concurrency,
                "acquisitions_per_worker": args.acquisitions,
                "rounds": args.rounds,
                "seconds_best": min(timings),
                "seconds_mean": statistics.mean(timings),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

