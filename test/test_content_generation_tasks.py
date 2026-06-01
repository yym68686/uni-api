import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
from fastapi import BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class DummyCircularList:
    def __init__(self, items):
        self.items = list(items)
        self.next_calls = []
        self.cooling_calls = []

    async def is_all_rate_limited(self, model):
        return False

    async def next(self, model):
        item = self.items[len(self.next_calls) % len(self.items)]
        self.next_calls.append((model, item))
        return item

    def get_items_count(self):
        return len(self.items)

    async def set_cooling(self, item, cooling_time):
        self.cooling_calls.append((item, cooling_time))


class DummyClient:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    def _pick_response(self, url):
        if isinstance(self.responses, dict):
            return self.responses[url]
        return self.responses

    async def post(self, url, headers=None, content=None, timeout=None):
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        return self._pick_response(url)

    async def get(self, url, headers=None, timeout=None):
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._pick_response(url)

    async def delete(self, url, headers=None, timeout=None):
        self.calls.append(
            {
                "method": "DELETE",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._pick_response(url)

    async def put(self, url, headers=None, content=None, timeout=None):
        self.calls.append(
            {
                "method": "PUT",
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        return self._pick_response(url)


class DummyClientManager:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        _ = base_url, proxy, http2
        yield DummyClient(self.responses, self.calls)


def _set_content_generation_state(monkeypatch, responses):
    provider_name = "deyunai"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["ark-key"]))
    main.app.state.config = {
        "providers": [
            {
                "provider": provider_name,
                "base_url": "https://api.deyunai.com/api/v3",
                "api": ["ark-key"],
                "model": [
                    {"doubao-seedance-1-0-pro-250528": "seedance-video"},
                ],
                "preferences": {},
            }
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["seedance-video"],
                "preferences": {"AUTO_RETRY": True},
            }
        ],
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.models_list = {"sk-test": ["seedance-video"]}
    main.app.state.routing_index = None
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.channel_manager = None
    main.app.state.client_manager = DummyClientManager(responses)
    return provider_name


def _set_lingjing_state(monkeypatch, responses):
    provider_name = "lingjing"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["lingjing-primary"]))
    main.app.state.config = {
        "providers": [
            {
                "provider": provider_name,
                "engine": "lingjing",
                "base_url": "https://api-llm.lingjingai.cn",
                "api": ["lingjing-primary"],
                "model": [
                    {"sd_2_0": "seedance-2-0"},
                ],
                "preferences": {
                    "access_key": "ak-test",
                    "secret_key": "sk-test-upstream",
                },
            }
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["seedance-2-0"],
                "preferences": {"AUTO_RETRY": True},
            }
        ],
    }
    main.app.state.api_list = ["sk-test"]
    main.app.state.models_list = {"sk-test": ["seedance-2-0"]}
    main.app.state.routing_index = None
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.channel_manager = None
    main.app.state.client_manager = DummyClientManager(responses)
    return provider_name


def _request_with_query(query=""):
    return SimpleNamespace(
        headers={},
        query_params={},
        url=SimpleNamespace(query=query),
    )


def _run_with_request_info(coro):
    token = main.request_info.set(
        {
            "request_id": "req-test",
            "api_key": "sk-test",
            "disconnect_event": None,
            "first_response_time": None,
            "success": False,
            "provider": None,
            "model": None,
        }
    )
    try:
        return asyncio.run(coro)
    finally:
        main.request_info.reset(token)


def test_content_generation_create_maps_model_and_remembers_task_route(monkeypatch):
    _set_content_generation_state(
        monkeypatch,
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.deyunai.com/api/v3/contents/generations/tasks"),
            json={"id": "cgt-test"},
        ),
    )
    handler = main.ContentGenerationTaskHandler()
    body = {
        "model": "seedance-video",
        "content": [{"type": "text", "text": "小猫对着镜头打哈欠"}],
        "duration": 5,
        "ratio": "16:9",
    }

    response = _run_with_request_info(
        handler.create_task(
            SimpleNamespace(headers={}),
            body,
            0,
            BackgroundTasks(),
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"id": "cgt-test"}
    assert len(main.app.state.client_manager.calls) == 1
    call = main.app.state.client_manager.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.deyunai.com/api/v3/contents/generations/tasks"
    assert call["headers"]["Authorization"] == "Bearer ark-key"
    assert json.loads(call["content"]) == {
        **body,
        "model": "doubao-seedance-1-0-pro-250528",
    }
    assert handler.task_routes["cgt-test"]["provider_api_key_raw"] == "ark-key"


def test_content_generation_get_uses_remembered_provider_and_key(monkeypatch):
    responses = {
        "https://api.deyunai.com/api/v3/contents/generations/tasks": httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.deyunai.com/api/v3/contents/generations/tasks"),
            json={"id": "cgt-test"},
        ),
        "https://api.deyunai.com/api/v3/contents/generations/tasks/cgt-test": httpx.Response(
            200,
            request=httpx.Request("GET", "https://api.deyunai.com/api/v3/contents/generations/tasks/cgt-test"),
            json={"id": "cgt-test", "status": "succeeded", "content": {"video_url": "https://example.com/out.mp4"}},
        ),
    }
    _set_content_generation_state(monkeypatch, responses)
    handler = main.ContentGenerationTaskHandler()

    _run_with_request_info(
        handler.create_task(
            SimpleNamespace(headers={}),
            {
                "model": "seedance-video",
                "content": [{"type": "text", "text": "生成一个短片"}],
            },
            0,
            BackgroundTasks(),
        )
    )
    response = _run_with_request_info(
        handler.get_or_delete_task(
            SimpleNamespace(headers={}),
            "cgt-test",
            0,
            BackgroundTasks(),
            method="GET",
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "succeeded"
    get_call = main.app.state.client_manager.calls[-1]
    assert get_call["method"] == "GET"
    assert get_call["url"] == "https://api.deyunai.com/api/v3/contents/generations/tasks/cgt-test"
    assert get_call["headers"]["Authorization"] == "Bearer ark-key"


def test_content_generation_delete_removes_remembered_route(monkeypatch):
    responses = {
        "https://api.deyunai.com/api/v3/contents/generations/tasks": httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.deyunai.com/api/v3/contents/generations/tasks"),
            json={"id": "cgt-test"},
        ),
        "https://api.deyunai.com/api/v3/contents/generations/tasks/cgt-test": httpx.Response(
            200,
            request=httpx.Request("DELETE", "https://api.deyunai.com/api/v3/contents/generations/tasks/cgt-test"),
            json={"id": "cgt-test", "deleted": True},
        ),
    }
    _set_content_generation_state(monkeypatch, responses)
    handler = main.ContentGenerationTaskHandler()

    _run_with_request_info(
        handler.create_task(
            SimpleNamespace(headers={}),
            {
                "model": "seedance-video",
                "content": [{"type": "text", "text": "生成一个短片"}],
            },
            0,
            BackgroundTasks(),
        )
    )
    assert "cgt-test" in handler.task_routes

    response = _run_with_request_info(
        handler.get_or_delete_task(
            SimpleNamespace(headers={}),
            "cgt-test",
            0,
            BackgroundTasks(),
            method="DELETE",
        )
    )

    assert response.status_code == 200
    assert "cgt-test" not in handler.task_routes
    assert main.app.state.client_manager.calls[-1]["method"] == "DELETE"


def test_lingjing_content_generation_uses_x_keys_and_normalizes_responses(monkeypatch):
    responses = {
        "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/submit": httpx.Response(
            200,
            request=httpx.Request("POST", "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/submit"),
            json={"code": 200, "msg": "OK", "data": {"taskId": "task-lj"}},
        ),
        "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/query?taskId=task-lj": httpx.Response(
            200,
            request=httpx.Request(
                "GET",
                "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/query?taskId=task-lj",
            ),
            json={
                "code": 200,
                "msg": "OK",
                "data": {
                    "task_id": "task-lj",
                    "status": "SUCCESS",
                    "result": [{"type": "video", "url": "https://example.com/out.mp4"}],
                },
            },
        ),
    }
    _set_lingjing_state(monkeypatch, responses)
    handler = main.ContentGenerationTaskHandler()

    create_response = _run_with_request_info(
        handler.create_task(
            SimpleNamespace(headers={}),
            {
                "model": "seedance-2-0",
                "content": [
                    {"type": "text", "text": "一只橘猫在窗边晒太阳"},
                    {"type": "image_url", "image_url": {"url": "asset://Asset-test"}, "role": "first_frame"},
                ],
                "ratio": "16:9",
                "duration": 5,
                "resolution": "720p",
                "generate_audio": False,
            },
            0,
            BackgroundTasks(),
        )
    )

    assert create_response.status_code == 200
    assert json.loads(create_response.body)["id"] == "task-lj"
    submit_call = main.app.state.client_manager.calls[0]
    assert submit_call["url"] == "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/submit"
    assert submit_call["headers"]["X-Access-Key"] == "ak-test"
    assert submit_call["headers"]["X-Secret-Key"] == "sk-test-upstream"
    assert "Authorization" not in submit_call["headers"]
    submit_body = json.loads(submit_call["content"])
    assert submit_body["modelCode"] == "sd_2_0"
    assert submit_body["taskParams"]["input"]["quality"] == "720"
    assert submit_body["taskParams"]["input"]["resources"][0]["source"] == {"kind": "asset_id", "value": "Asset-test"}
    assert handler.task_routes["task-lj"]["provider_api_key_raw"] == "lingjing-primary"

    query_response = _run_with_request_info(
        handler.get_or_delete_task(
            SimpleNamespace(headers={}),
            "task-lj",
            0,
            BackgroundTasks(),
            method="GET",
        )
    )

    query_body = json.loads(query_response.body)
    assert query_response.status_code == 200
    assert query_body["status"] == "succeeded"
    assert query_body["content"]["video_url"] == "https://example.com/out.mp4"
    query_call = main.app.state.client_manager.calls[-1]
    assert query_call["url"] == "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/query?taskId=task-lj"
    assert query_call["headers"]["X-Access-Key"] == "ak-test"
    assert "Authorization" not in query_call["headers"]


def test_lingjing_openapi_material_endpoint_proxies_raw_response(monkeypatch):
    upstream_url = "https://api-llm.lingjingai.cn/api/entrance/openapi/material/asset-groups"
    _set_lingjing_state(
        monkeypatch,
        httpx.Response(
            200,
            request=httpx.Request("POST", upstream_url),
            json={"code": 200, "data": {"id": "Group-test"}},
        ),
    )
    handler = main.LingjingOpenapiHandler()
    body = {"platform": "BYTEPLUS", "name": "codex-test"}

    response = _run_with_request_info(
        handler.request_openapi(
            _request_with_query(),
            body,
            0,
            BackgroundTasks(),
            method="POST",
            openapi_path="/material/asset-groups",
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"code": 200, "data": {"id": "Group-test"}}
    call = main.app.state.client_manager.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == upstream_url
    assert call["headers"]["X-Access-Key"] == "ak-test"
    assert call["headers"]["X-Secret-Key"] == "sk-test-upstream"
    assert "Authorization" not in call["headers"]
    assert json.loads(call["content"]) == body


def test_lingjing_openapi_draw_submit_accepts_request_model_alias(monkeypatch):
    upstream_url = "https://api-llm.lingjingai.cn/api/entrance/openapi/draw/task/submit"
    _set_lingjing_state(
        monkeypatch,
        httpx.Response(
            200,
            request=httpx.Request("POST", upstream_url),
            json={"code": 200, "data": {"taskId": "task-lj"}},
        ),
    )
    handler = main.LingjingOpenapiHandler()

    response = _run_with_request_info(
        handler.request_openapi(
            _request_with_query(),
            {
                "model": "seedance-2-0",
                "taskParams": {"input": {"prompt": "测试视频", "quality": "480", "duration": 4, "ratio": "16:9"}},
            },
            0,
            BackgroundTasks(),
            method="POST",
            openapi_path="/draw/task/submit",
        )
    )

    assert response.status_code == 200
    submit_body = json.loads(main.app.state.client_manager.calls[0]["content"])
    assert submit_body["modelCode"] == "sd_2_0"
    assert "model" not in submit_body


def test_lingjing_upstream_query_strips_routing_only_model():
    assert (
        main._lingjing_upstream_query("platform=BYTEPLUS&model=seedance-2-0&request_model=seedance-2-0&taskId=abc")
        == "platform=BYTEPLUS&taskId=abc"
    )
