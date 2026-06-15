from __future__ import annotations

import os
import re
from typing import Any, Mapping


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_config_environment(value: Any, env: Mapping[str, str] | None = None) -> Any:
    env = os.environ if env is None else env
    if isinstance(value, dict):
        return {
            key: expand_config_environment(item, env)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_config_environment(item, env) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_environment(item, env) for item in value)
    if isinstance(value, str):
        return _expand_string(value, env)
    return value


def _expand_string(value: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in env:
            return str(env[name])
        if default is not None:
            return default
        return ""

    return _ENV_PATTERN.sub(replace, value)
