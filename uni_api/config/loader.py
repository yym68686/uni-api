from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from ruamel.yaml import YAML

from uni_api.config.env import expand_config_environment
from uni_api.config.schema import validate_config_data


yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)


def load_local_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config_data = yaml.load(file)
    return validate_config_data(expand_config_environment(config_data))


async def load_remote_config(url: str, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=100, write=30.0, pool=200))
    try:
        response = await client.get(url)
        response.raise_for_status()
        config_data = yaml.load(response.text)
        return validate_config_data(expand_config_environment(config_data))
    finally:
        if close_client:
            await client.aclose()
