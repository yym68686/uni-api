#!/usr/bin/env python3
"""Run the gateway's isolated HTTP regressions from any working directory."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def stop_processes(process: subprocess.Popen) -> None:
    """Stop a timed-out suite together with its mock servers and gateway child."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "binary", nargs="?", type=Path, default=root / "target/debug/uni-api-front"
    )
    parser.add_argument("--timeout", type=int, default=180, help="Seconds per script")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Gateway binary does not exist: {binary}; run cargo build --locked")

    tests = sorted((root / "tests/http").glob("verify_*.py"))
    if not tests:
        parser.error("No HTTP regression scripts found")
    failures = []
    for script in tests:
        print(f"\nRunning {script.name}", flush=True)
        process = subprocess.Popen(
            [sys.executable, str(script), str(binary)],
            cwd=root,
            start_new_session=os.name == "posix",
        )
        try:
            if process.wait(timeout=args.timeout):
                failures.append(script.name)
        except subprocess.TimeoutExpired:
            stop_processes(process)
            failures.append(script.name)
            print(f"Timed out after {args.timeout}s: {script.name}", file=sys.stderr)
        except KeyboardInterrupt:
            stop_processes(process)
            return 130
    print(f"\nHTTP regressions: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("Failed: " + ", ".join(failures), file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
