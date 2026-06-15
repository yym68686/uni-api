from __future__ import annotations

import asyncio
import json
import os
from time import time
from typing import Any, Callable, Optional

from fastapi import HTTPException


def split_codex_api_key(raw_api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if raw_api_key is None:
        return None, None
    raw = str(raw_api_key).strip()
    if not raw:
        return None, None
    if "," not in raw:
        return None, raw
    account_id, token = raw.split(",", 1)
    account_id = account_id.strip() or None
    token = token.strip()
    if not token:
        raise ValueError("Invalid Codex API key format: expected 'account_id,refresh_token' (refresh_token missing)")
    return account_id, token


class CodexRefreshTokenStore:
    def __init__(self, path: str, *, logger: Any = None) -> None:
        self.path = path
        self.logger = logger
        self.tokens: dict[str, str] = {}
        self.loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        if self.loaded:
            return
        async with self._lock:
            if self.loaded:
                return
            self._load_unlocked(action="load")
            self.loaded = True

    async def reload(self) -> None:
        async with self._lock:
            self.tokens.clear()
            self._load_unlocked(action="reload")
            self.loaded = True

    async def get(self, provider_api_key_raw: Optional[str], *, force_reload: bool = False) -> Optional[str]:
        key = self._normalize_key(provider_api_key_raw)
        if key is None:
            return None
        if force_reload:
            await self.reload()
        else:
            await self.ensure_loaded()
        token = self.tokens.get(key)
        return str(token) if token else None

    async def persist(self, provider_api_key_raw: Optional[str], refresh_token: Optional[str]) -> None:
        key = self._normalize_key(provider_api_key_raw)
        token = str(refresh_token or "").strip()
        if key is None or not token:
            return
        await self.ensure_loaded()

        async with self._lock:
            if self.tokens.get(key) == token:
                return
            self.tokens[key] = token
            self._write_atomic_unlocked()

    @staticmethod
    def _normalize_key(provider_api_key_raw: Optional[str]) -> Optional[str]:
        if provider_api_key_raw is None:
            return None
        key = str(provider_api_key_raw).strip()
        return key or None

    def _load_unlocked(self, *, action: str) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for raw_key, raw_value in payload.items():
                    key = str(raw_key).strip()
                    value = str(raw_value).strip()
                    if key and value:
                        self.tokens[key] = value
        except FileNotFoundError:
            pass
        except Exception as exc:
            self._warn("Failed to %s Codex refresh token store '%s': %s", action, self.path, exc)

    def _write_atomic_unlocked(self) -> None:
        try:
            store_dir = os.path.dirname(self.path)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            tmp_path = f"{self.path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.tokens, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        except Exception as exc:
            self._warn("Failed to persist Codex refresh token store '%s': %s", self.path, exc)

    def _warn(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.warning(message, *args)


class CodexOAuthTokenManager:
    def __init__(
        self,
        *,
        refresh_token_store: CodexRefreshTokenStore,
        client_getter: Callable[[str, Optional[str]], Any],
        token_url: str,
        client_id: str,
        refresh_skew_seconds: int = 30,
    ) -> None:
        self.refresh_token_store = refresh_token_store
        self.client_getter = client_getter
        self.token_url = token_url
        self.client_id = client_id
        self.refresh_skew_seconds = refresh_skew_seconds
        self.cache: dict[str, dict[str, Any]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def resolve_upstream_auth(
        self,
        provider_name: str,
        provider_api_key_raw: Optional[str],
        proxy: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if provider_api_key_raw is None:
            return None, None

        raw = str(provider_api_key_raw).strip()
        if not raw:
            return None, None

        if "," not in raw:
            return raw, None

        codex_account_id, _ = split_codex_api_key(raw)
        api_key = await self.get_access_token(provider_name, raw, proxy)
        return api_key, codex_account_id

    async def get_access_token(self, provider_name: str, provider_api_key_raw: str, proxy: Optional[str]) -> str:
        _account_id, refresh_token_from_config = split_codex_api_key(provider_api_key_raw)
        if not refresh_token_from_config:
            raise HTTPException(status_code=401, detail="Codex refresh_token missing")

        persisted_refresh_token = await self.refresh_token_store.get(provider_api_key_raw)
        if persisted_refresh_token:
            refresh_token_from_config = persisted_refresh_token

        lock = self._lock_for(provider_api_key_raw)
        async with lock:
            entry = self.cache.get(provider_api_key_raw) or {}
            if self.access_token_is_valid(entry):
                return str(entry["access_token"])

            old_refresh_token = str(entry.get("refresh_token") or refresh_token_from_config).strip()
            try:
                refreshed = await self.refresh_access_token(old_refresh_token, proxy)
            except HTTPException as exc:
                detail = str(getattr(exc, "detail", "") or "")
                if "refresh_token_reused" in detail:
                    latest = await self.refresh_token_store.get(provider_api_key_raw, force_reload=True)
                    if latest and latest != old_refresh_token:
                        refreshed = await self.refresh_access_token(latest, proxy)
                        old_refresh_token = latest
                    else:
                        raise
                else:
                    raise

            updated_refresh_token = refreshed.get("refresh_token") or old_refresh_token
            self.cache[provider_api_key_raw] = {
                "access_token": refreshed["access_token"],
                "refresh_token": updated_refresh_token,
                "expires_at": refreshed.get("expires_at"),
            }
            await self.refresh_token_store.persist(provider_api_key_raw, updated_refresh_token)
            return str(refreshed["access_token"])

    async def refresh_access_token(self, refresh_token: str, proxy: Optional[str]) -> dict[str, Any]:
        token = str(refresh_token or "").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Codex refresh_token missing")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "client_id": self.client_id,
            "grant_type": "refresh_token",
            "refresh_token": token,
            "scope": "openid profile email",
        }

        try:
            async with self.client_getter(self.token_url, proxy) as client:
                resp = await client.post(self.token_url, data=data, headers=headers, timeout=30)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Codex token refresh request failed: {type(exc).__name__}: {exc}",
            )

        if resp.status_code != 200:
            body = (resp.text or "").strip()
            raise HTTPException(status_code=401, detail=f"Codex token refresh failed: status {resp.status_code}: {body}")

        try:
            payload = resp.json()
        except Exception:
            payload = {}

        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(status_code=401, detail=f"Codex token refresh returned empty access_token: {resp.text}")

        new_refresh_token = str(payload.get("refresh_token") or "").strip() or None
        expires_at = None
        try:
            expires_in_int = int(payload.get("expires_in"))
            if expires_in_int > 0:
                expires_at = time() + expires_in_int
        except Exception:
            expires_at = None

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "expires_at": expires_at,
        }

    def clear(self, provider_api_key_raw: Optional[str]) -> None:
        if provider_api_key_raw is not None:
            self.cache.pop(provider_api_key_raw, None)

    def access_token_is_valid(self, entry: dict[str, Any]) -> bool:
        token = entry.get("access_token")
        if not token:
            return False
        expires_at = entry.get("expires_at")
        if expires_at is None:
            return True
        try:
            return time() < float(expires_at) - self.refresh_skew_seconds
        except Exception:
            return True

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self.locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[key] = lock
        return lock

