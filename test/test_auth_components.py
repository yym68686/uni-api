import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from uni_api.auth.api_keys import (
    extract_api_key_from_headers,
    require_admin_api_key,
    require_api_key_index,
    resolve_api_key_index,
)
from uni_api.auth.codex_oauth import (
    CodexOAuthTokenManager,
    CodexRefreshTokenStore,
    split_codex_api_key,
)


def test_extract_api_key_from_headers_supports_x_api_key_and_bearer():
    assert extract_api_key_from_headers({"x-api-key": "sk-header"}) == "sk-header"
    assert extract_api_key_from_headers({"Authorization": "Bearer sk-bearer"}) == "sk-bearer"
    assert extract_api_key_from_headers({}) is None


def test_require_api_key_index_and_admin_permissions():
    assert resolve_api_key_index(["sk-a", "sk-b"], "sk-b") == 1
    assert require_api_key_index(["sk-a"], "sk-a") == 0

    with pytest.raises(HTTPException) as invalid:
        require_api_key_index(["sk-a"], "missing")
    assert invalid.value.status_code == 403

    api_keys_db = [{"api": "sk-a", "role": "user"}, {"api": "sk-admin", "role": "admin"}]
    assert require_admin_api_key(api_keys_db, ["sk-a", "sk-admin"], "sk-admin") == "sk-admin"

    with pytest.raises(HTTPException) as denied:
        require_admin_api_key(api_keys_db, ["sk-a", "sk-admin"], "sk-a")
    assert denied.value.status_code == 403


def test_single_api_key_is_admin_compatible():
    assert require_admin_api_key([{"api": "sk-only", "role": "user"}], ["sk-only"], "sk-only") == "sk-only"


def test_split_codex_api_key_plain_and_account_refresh():
    assert split_codex_api_key("plain-bearer") == (None, "plain-bearer")
    assert split_codex_api_key("account-1, refresh-token") == ("account-1", "refresh-token")

    with pytest.raises(ValueError):
        split_codex_api_key("account-1,")


async def test_codex_refresh_token_store_persists_with_atomic_replace(tmp_path: Path):
    path = tmp_path / "codex_refresh_tokens.json"
    store = CodexRefreshTokenStore(str(path))

    await store.persist("account-1,refresh-1", "refresh-2")

    assert json.loads(path.read_text()) == {"account-1,refresh-1": "refresh-2"}
    assert list(tmp_path.glob("*.tmp.*")) == []

    reloaded = CodexRefreshTokenStore(str(path))
    assert await reloaded.get("account-1,refresh-1") == "refresh-2"


class _RefreshClient:
    def __init__(self, manager):
        self.manager = manager

    async def post(self, url, data=None, headers=None, timeout=None):
        self.manager.calls += 1
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "access_token": f"access-{self.manager.calls}",
                "refresh_token": "refresh-new",
                "expires_in": 3600,
            },
        )


class _RefreshClientManager:
    def __init__(self):
        self.calls = 0

    @asynccontextmanager
    async def get_client(self, url, proxy=None):
        yield _RefreshClient(self)


async def test_codex_oauth_refresh_is_single_flight_per_provider_key(tmp_path: Path):
    client_manager = _RefreshClientManager()
    store = CodexRefreshTokenStore(str(tmp_path / "tokens.json"))
    manager = CodexOAuthTokenManager(
        refresh_token_store=store,
        client_getter=client_manager.get_client,
        token_url="https://auth.example.test/oauth/token",
        client_id="client-id",
    )

    results = await asyncio.gather(
        manager.get_access_token("codex", "account-1,refresh-1", None),
        manager.get_access_token("codex", "account-1,refresh-1", None),
        manager.get_access_token("codex", "account-1,refresh-1", None),
    )

    assert results == ["access-1", "access-1", "access-1"]
    assert client_manager.calls == 1
    assert await store.get("account-1,refresh-1") == "refresh-new"


async def test_codex_oauth_access_token_ttl_controls_refresh(tmp_path: Path):
    client_manager = _RefreshClientManager()
    manager = CodexOAuthTokenManager(
        refresh_token_store=CodexRefreshTokenStore(str(tmp_path / "tokens.json")),
        client_getter=client_manager.get_client,
        token_url="https://auth.example.test/oauth/token",
        client_id="client-id",
        refresh_skew_seconds=30,
    )

    manager.cache["account-1,refresh-1"] = {
        "access_token": "cached",
        "refresh_token": "refresh-1",
        "expires_at": 9999999999,
    }
    assert await manager.get_access_token("codex", "account-1,refresh-1", None) == "cached"
    assert client_manager.calls == 0

    manager.cache["account-1,refresh-1"]["expires_at"] = 1
    assert await manager.get_access_token("codex", "account-1,refresh-1", None) == "access-1"
    assert client_manager.calls == 1

