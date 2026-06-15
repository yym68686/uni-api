from __future__ import annotations

import asyncio
import json
from typing import Any

JSONDecodeError = json.JSONDecodeError


def dumps_text(value: Any, *, ensure_ascii: bool = False, separators: tuple[str, str] | None = None) -> str:
    kwargs: dict[str, Any] = {"ensure_ascii": ensure_ascii}
    if separators is not None:
        kwargs["separators"] = separators
    return json.dumps(value, **kwargs)


def dumps_bytes(value: Any, *, ensure_ascii: bool = False, separators: tuple[str, str] = (",", ":")) -> bytes:
    return dumps_text(value, ensure_ascii=ensure_ascii, separators=separators).encode("utf-8")


def loads_data(value: str | bytes | bytearray) -> Any:
    return json.loads(value)


async def dumps_text_async(value: Any, *, ensure_ascii: bool = False, separators: tuple[str, str] | None = None) -> str:
    return await asyncio.to_thread(dumps_text, value, ensure_ascii=ensure_ascii, separators=separators)


async def loads_data_async(value: str | bytes | bytearray) -> Any:
    return await asyncio.to_thread(loads_data, value)


def sse_data(payload: Any, *, ensure_ascii: bool = False) -> str:
    from core.utils import end_of_line

    return "data: " + dumps_text(payload, ensure_ascii=ensure_ascii) + end_of_line


def dumps(value: Any, *, ensure_ascii: bool = False, separators: tuple[str, str] | None = None) -> str:
    return dumps_text(value, ensure_ascii=ensure_ascii, separators=separators)


def loads(value: str | bytes | bytearray) -> Any:
    return loads_data(value)


async def dumps_async(value: Any, *, ensure_ascii: bool = False, separators: tuple[str, str] | None = None) -> str:
    return await dumps_text_async(value, ensure_ascii=ensure_ascii, separators=separators)


async def loads_async(value: str | bytes | bytearray) -> Any:
    return await loads_data_async(value)
