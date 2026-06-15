from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from fastapi import HTTPException


def extract_api_key_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    token = headers.get("x-api-key")
    if token:
        return token

    authorization = headers.get("Authorization") or headers.get("authorization")
    if not authorization:
        return None

    parts = authorization.split(" ", 1)
    if len(parts) == 2:
        return parts[1].strip() or None
    return None


def resolve_api_key_index(api_list: Sequence[str], token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    try:
        return list(api_list).index(token)
    except ValueError:
        return None


def require_api_key_index(api_list: Sequence[str], token: Optional[str]) -> int:
    api_index = resolve_api_key_index(api_list, token)
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_index


def require_admin_api_key(
    api_keys_db: Sequence[Mapping[str, Any]],
    api_list: Sequence[str],
    token: Optional[str],
) -> str:
    api_index = require_api_key_index(api_list, token)
    if len(api_list) == 1:
        return str(token)
    role = api_keys_db[api_index].get("role", "") if api_index < len(api_keys_db) else ""
    if "admin" not in str(role):
        raise HTTPException(status_code=403, detail="Permission denied")
    return str(token)

