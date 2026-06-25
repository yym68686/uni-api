import asyncio
import json

import httpx
import zstandard as zstd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import main
from uni_api.middleware.request_decompression import RequestBodyDecompressionMiddleware


def _zstd_compress(body: bytes) -> bytes:
    return zstd.ZstdCompressor(level=3).compress(body)


def test_zstd_middleware_decodes_body_and_strips_encoding_headers():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo(request: Request):
        return {
            "body": (await request.body()).decode("utf-8"),
            "content_encoding": request.headers.get("content-encoding"),
            "content_length": request.headers.get("content-length"),
        }

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=_zstd_compress(b'{"ok":true}'),
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                    "Content-Length": "999",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {
        "body": '{"ok":true}',
        "content_encoding": None,
        "content_length": None,
    }


def test_zstd_middleware_rejects_invalid_zstd_body():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=b"not-zstd",
                headers={"Content-Encoding": "zstd"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid zstd body"}


def test_zstd_middleware_rejects_unsupported_content_encoding():
    app = FastAPI()
    app.add_middleware(RequestBodyDecompressionMiddleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/echo",
                content=b"body",
                headers={"Content-Encoding": "gzip"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 415
    assert response.json() == {"detail": "unsupported content encoding: gzip"}


def test_main_app_accepts_zstd_chat_completion_request(monkeypatch):
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

    async def fake_request_model(request, api_index, background_tasks, endpoint=None, current_info=None, http_request=None):
        _ = http_request
        assert api_index == 0
        assert endpoint is None
        assert current_info["model"] == "gpt-5.5"
        return JSONResponse({"model": request.model, "message": request.messages[0].content})

    monkeypatch.setattr(main.model_handler, "request_model", fake_request_model)
    payload = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "zstd request"}],
        "stream": False,
    }

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/v1/chat/completions",
                content=_zstd_compress(json.dumps(payload).encode("utf-8")),
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"model": "gpt-5.5", "message": "zstd request"}


def test_main_app_accepts_zstd_responses_request(monkeypatch):
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

    async def fake_request_responses(
        http_request,
        request,
        api_index,
        background_tasks,
        endpoint="/v1/responses",
    ):
        assert api_index == 0
        assert endpoint == "/v1/responses"
        assert http_request.headers.get("content-encoding") is None
        assert request.model == "gpt-5.5"
        assert request.input == "zstd responses request"
        return JSONResponse({"model": request.model, "input": request.input})

    monkeypatch.setattr(main.responses_handler, "request_responses", fake_request_responses)
    payload = {
        "model": "gpt-5.5",
        "input": "zstd responses request",
        "stream": False,
    }

    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/v1/responses",
                content=_zstd_compress(json.dumps(payload).encode("utf-8")),
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                    "Content-Encoding": "zstd",
                },
            )

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"model": "gpt-5.5", "input": "zstd responses request"}
