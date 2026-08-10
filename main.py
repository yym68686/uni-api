"""Compatibility entry point for the refactored application."""

from __future__ import annotations

import os as _os
import sys as _sys
import types as _types

import uni_api.runtime as _runtime

for _name in dir(_runtime):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_runtime, _name)

__all__ = tuple(name for name in globals() if not name.startswith("__"))


class _MainModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__") and hasattr(_runtime, name):
            setattr(_runtime, name, value)


_sys.modules[__name__].__class__ = _MainModule


def _run_uvicorn_cli(argv: list[str] | None = None) -> None:
    """Run Uvicorn with dynamic safe defaults and compatible CLI overrides."""

    import click
    import uvicorn

    arguments = [
        "uni_api.runtime:app",
        "--host",
        str(_os.getenv("HOST", "0.0.0.0")),
        "--port",
        str(_os.getenv("PORT", "8000")),
        "--backlog",
        str(_runtime.UVICORN_BACKLOG),
        *(list(argv) if argv is not None else _sys.argv[1:]),
    ]
    try:
        with uvicorn.main.make_context("uvicorn", arguments) as context:
            parameters = dict(context.params)
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None

    workers = parameters.get("workers")
    if workers not in (None, 1):
        raise SystemExit(
            "uni-api uses one process-scoped memory/admission governor; "
            "--workers must remain 1"
        )
    if parameters.get("reload"):
        raise SystemExit(
            "main.py does not support --reload with the process-scoped "
            "production admission envelope"
        )

    requested_connection_limit = parameters.get("limit_concurrency")
    connection_limit = _runtime.UVICORN_CONNECTION_LIMIT
    if requested_connection_limit is not None:
        if _runtime.RESOURCE_ADMISSION_MODE != "legacy":
            raise SystemExit(
                "--limit-concurrency is available only when "
                "RESOURCE_ADMISSION_MODE=legacy"
            )
        requested_connection_limit = int(requested_connection_limit)
        if requested_connection_limit <= 0:
            raise SystemExit(
                "--limit-concurrency must be positive when explicitly set"
            )
        if (
            connection_limit is not None
            and requested_connection_limit > connection_limit
        ):
            raise SystemExit(
                "--limit-concurrency cannot exceed the configured legacy "
                "connection limit"
            )
        connection_limit = requested_connection_limit
    protocol, protocol_stats = _runtime.build_bounded_h11_protocol(
        connection_limit=connection_limit,
        header_timeout_seconds=_runtime.UVICORN_HEADER_TIMEOUT_SECONDS,
        connection_admission=_runtime.INBOUND_CONNECTION_ADMISSION,
    )
    _runtime.UVICORN_CONNECTION_LIMIT = connection_limit
    _runtime.UVICORN_HTTP_PROTOCOL = protocol
    _runtime.BOUNDED_HTTP_PROTOCOL_STATS = protocol_stats
    parameters["http"] = protocol
    parameters["limit_concurrency"] = None
    parameters["workers"] = 1
    uvicorn.main.callback(**parameters)


if __name__ == "__main__":
    _run_uvicorn_cli()
