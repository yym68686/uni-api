from uni_api.upstream.urls import (
    lingjing_upstream_query,
    normalize_content_generation_tasks_upstream_url,
    normalize_lingjing_draw_task_upstream_url,
    normalize_lingjing_openapi_upstream_url,
    normalize_messages_upstream_url,
    normalize_responses_compact_upstream_url,
    normalize_responses_upstream_url,
)


def test_responses_url_normalization_preserves_gpt_and_appends_codex_endpoint():
    assert normalize_responses_upstream_url("https://example.com/v1/responses", "gpt") == "https://example.com/v1/responses"
    assert normalize_responses_upstream_url("https://chatgpt.com/backend-api/codex", "codex") == "https://chatgpt.com/backend-api/codex/responses"


def test_responses_compact_url_normalization_handles_codex_and_existing_paths():
    assert normalize_responses_compact_upstream_url("https://example.com/v1/responses", "gpt") == "https://example.com/v1/responses/compact"
    assert normalize_responses_compact_upstream_url("https://chatgpt.com/backend-api/codex", "codex") == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert normalize_responses_compact_upstream_url("https://example.com/responses/compact", "gpt") == "https://example.com/responses/compact"


def test_messages_url_normalization_appends_messages_endpoint_once():
    assert normalize_messages_upstream_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/messages"
    assert normalize_messages_upstream_url("https://api.anthropic.com/v1/messages/") == "https://api.anthropic.com/v1/messages"


def test_content_generation_tasks_url_normalization_and_task_id_encoding():
    assert (
        normalize_content_generation_tasks_upstream_url("https://ark.example.com", "task/id")
        == "https://ark.example.com/api/v3/contents/generations/tasks/task%2Fid"
    )
    assert (
        normalize_content_generation_tasks_upstream_url("https://ark.example.com/api/v3", None)
        == "https://ark.example.com/api/v3/contents/generations/tasks"
    )


def test_lingjing_openapi_url_normalization_and_query_filtering():
    assert (
        normalize_lingjing_openapi_upstream_url(
            "https://api-llm.lingjingai.cn",
            "/material/assets/create",
            "model=a&request_model=b&keep=1",
        )
        == "https://api-llm.lingjingai.cn/api/entrance/openapi/material/assets/create?model=a&request_model=b&keep=1"
    )
    assert lingjing_upstream_query("platform=x&model=a&request_model=b&keep=1") == "platform=x&keep=1"


def test_lingjing_draw_task_url_normalization():
    assert normalize_lingjing_draw_task_upstream_url("https://api-llm.lingjingai.cn", method="POST").endswith(
        "/api/entrance/openapi/draw/task/submit"
    )
    assert normalize_lingjing_draw_task_upstream_url(
        "https://api-llm.lingjingai.cn",
        method="GET",
        task_id="task/id",
    ).endswith("/api/entrance/openapi/draw/task/query?taskId=task%2Fid")

