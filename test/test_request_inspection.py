from uni_api.observability.request_inspection import inspect_request_body


def test_stats_middleware_does_not_reintroduce_full_pydantic_validation():
    import inspect
    from uni_api.observability.middleware import StatsMiddleware

    source = inspect.getsource(StatsMiddleware.dispatch)
    assert "UnifiedRequest.model_validate" not in source


def test_chat_route_reuses_fastapi_validated_request_object():
    import inspect
    import main
    from uni_api.api.chat import chat_completions_response

    source = inspect.getsource(main.chat_completions_route)
    handler_source = inspect.getsource(chat_completions_response)
    assert "RequestModel.model_validate" not in source
    assert "model_validate" not in source + handler_source
    assert "RequestModel(" not in source
    assert "model_handler.request_model(request, api_index, background_tasks)" in handler_source


def test_upstream_retry_loop_does_not_rebuild_full_request_models():
    import inspect
    import upstream

    source = inspect.getsource(upstream.UpstreamRunner)
    assert "RequestModel(" not in source
    assert "ResponsesRequest(" not in source
    assert "model_validate" not in source


def test_streaming_modules_do_not_convert_chunks_to_pydantic_models():
    import inspect
    import uni_api.streaming.responses_events as responses_events
    import uni_api.streaming.sse as sse

    source = inspect.getsource(responses_events) + inspect.getsource(sse)
    assert "BaseModel" not in source
    assert "model_validate" not in source


def test_inspect_chat_request_body_uses_last_text_message():
    inspection = inspect_request_body(
        {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "first"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                        {"type": "text", "text": "last text"},
                    ],
                },
            ],
        }
    )

    assert inspection.model == "gpt-4.1"
    assert inspection.request_type == "chat"
    assert inspection.moderated_content == "last text"


def test_inspect_responses_request_body_uses_last_user_input_text():
    inspection = inspect_request_body(
        {
            "model": "gpt-4.1",
            "input": [
                {"role": "user", "content": "old"},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "new"}],
                },
            ],
        }
    )

    assert inspection.model == "gpt-4.1"
    assert inspection.request_type == "chat"
    assert inspection.moderated_content == "new"


def test_inspect_image_tts_embedding_and_moderation_bodies():
    image = inspect_request_body({"model": "dall-e-3", "prompt": "draw"})
    tts = inspect_request_body({"model": "tts-1", "input": "speak"})
    embedding = inspect_request_body({"model": "text-embedding-3-small", "input": ["a", "b"]})
    moderation = inspect_request_body({"input": "moderate me"})

    assert image.request_type == "image"
    assert image.moderated_content == "draw"
    assert tts.request_type == "tts"
    assert tts.moderated_content == "speak"
    assert embedding.request_type == "embedding"
    assert embedding.moderated_content == "a\nb"
    assert moderation.request_type == "moderation"
    assert moderation.model is None
    assert moderation.moderated_content is None
