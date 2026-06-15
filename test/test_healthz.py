import asyncio
import os
import sys

import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def test_healthz_does_not_require_api_key():
    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/healthz")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": main.VERSION}


def test_trace_headers_are_returned_on_authenticated_request(monkeypatch):
    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    main.app.state.config = {
        "api_keys": [{"api": "sk-test", "model": ["all"]}],
        "preferences": {"rate_limit": "999999/min"},
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.user_api_keys_rate_limit = main._build_user_api_keys_rate_limit(
        main.app.state.config,
        main.app.state.api_list,
    )

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(
                "/v1/models",
                headers={
                    "Authorization": "Bearer sk-test",
                    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                    "x-request-id": "legacy-request",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_database_disabled_core_v1_path_does_not_require_paid_state(monkeypatch):
    monkeypatch.setattr(main, "DISABLE_DATABASE", True)
    if hasattr(main.app.state, "paid_api_keys_states"):
        delattr(main.app.state, "paid_api_keys_states")
    main.app.state.config = {
        "api_keys": [{"api": "sk-test", "model": ["all"]}],
        "preferences": {"rate_limit": "999999/min"},
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.user_api_keys_rate_limit = main._build_user_api_keys_rate_limit(
        main.app.state.config,
        main.app.state.api_list,
    )

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})

    response = asyncio.run(run_request())

    assert response.status_code == 200
