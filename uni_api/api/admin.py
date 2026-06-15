from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


async def api_config_response(config: dict[str, Any]) -> JSONResponse:
    return JSONResponse(content={"api_config": jsonable_encoder(config)})


async def api_config_update_response(
    *,
    app: Any,
    config_patch: dict[str, Any],
    update_config: Callable[..., Awaitable[tuple[dict[str, Any], list[dict[str, Any]], list[str]]]],
    refresh_runtime_state: Callable[[Any], Awaitable[None]],
) -> JSONResponse:
    if "providers" in config_patch:
        app.state.config["providers"] = config_patch["providers"]
        app.state.config, app.state.api_keys_db, app.state.api_list = await update_config(
            app.state.config,
            use_config_url=False,
        )
        await refresh_runtime_state(app)
    return JSONResponse(content={"message": "API config updated"})

