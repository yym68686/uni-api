import httpx
import pytest
from fastapi import HTTPException

from uni_api.upstream.policies import CooldownPolicy, ProviderErrorClassifier, RetryPolicy
from uni_api.upstream.responses_errors import responses_failure_error
from uni_api.upstream.transport_errors import classify_httpx_transport_error


def _safe_get(data, *keys, default=None):
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _get_engine(provider, endpoint=None, original_model=None):
    _ = endpoint, original_model
    return provider.get("engine", "gpt"), None


def test_provider_error_classifier_normalizes_http_and_network_errors():
    classifier = ProviderErrorClassifier(_safe_get)

    assert classifier.normalize_exception(HTTPException(status_code=418, detail="teapot")) == (418, "teapot")
    assert classifier.normalize_exception(httpx.ConnectError("no route")) == (503, "Unable to connect to service")
    assert classifier.remap_status_code(500, "string_above_max_length") == 413


@pytest.mark.parametrize(
    ("error_type", "kind", "owner", "phase", "status_code", "penalty"),
    [
        (
            httpx.ConnectTimeout,
            "connect_timeout",
            "upstream_transport",
            "connect",
            504,
            True,
        ),
        (
            httpx.ConnectError,
            "connect_error",
            "upstream_transport",
            "connect",
            503,
            True,
        ),
        (
            httpx.PoolTimeout,
            "pool_timeout",
            "ember_local_overload",
            "pool_acquire",
            503,
            False,
        ),
        (
            httpx.ReadTimeout,
            "read_timeout",
            "upstream_transport",
            "unknown",
            504,
            True,
        ),
        (
            httpx.WriteTimeout,
            "write_timeout",
            "upstream_transport",
            "send_body",
            504,
            True,
        ),
        (
            httpx.ReadError,
            "read_error",
            "upstream_transport",
            "read_body",
            502,
            True,
        ),
        (
            httpx.WriteError,
            "write_error",
            "upstream_transport",
            "send_body",
            502,
            True,
        ),
        (
            httpx.CloseError,
            "close_error",
            "upstream_transport",
            "close",
            502,
            True,
        ),
        (
            httpx.ProxyError,
            "proxy_error",
            "upstream_transport",
            "proxy_connect",
            502,
            True,
        ),
        (
            httpx.UnsupportedProtocol,
            "unsupported_protocol",
            "ember_local_configuration",
            "configuration",
            500,
            True,
        ),
        (
            httpx.LocalProtocolError,
            "local_protocol_error",
            "upstream_transport",
            "protocol",
            502,
            True,
        ),
        (
            httpx.RemoteProtocolError,
            "remote_protocol_error",
            "upstream_transport",
            "protocol",
            502,
            True,
        ),
    ],
)
def test_httpx_transport_error_classification_is_complete(
    error_type,
    kind,
    owner,
    phase,
    status_code,
    penalty,
):
    classification = classify_httpx_transport_error(error_type("failure"))

    assert classification is not None
    assert classification.kind == kind
    assert classification.owner == owner
    assert classification.phase == phase
    assert classification.status_code == status_code
    assert classification.provider_penalty_eligible is penalty


def test_read_timeout_classification_distinguishes_first_byte_phase():
    before = classify_httpx_transport_error(
        httpx.ReadTimeout("timed out"),
        failure_stage="precommit",
        first_byte_observed=False,
    )
    after = classify_httpx_transport_error(
        httpx.ReadTimeout("timed out"),
        failure_stage="postcommit",
        first_byte_observed=True,
    )

    assert before is not None
    assert before.kind == "read_timeout"
    assert before.phase == "before_first_byte"
    assert after is not None
    assert after.kind == "read_timeout"
    assert after.phase == "after_first_byte"


@pytest.mark.parametrize(
    ("error_type", "timeout_key", "timeout_value", "expected"),
    [
        (
            httpx.ConnectTimeout,
            "connect",
            15,
            (504, "Connection timed out after 15 seconds"),
        ),
        (
            httpx.PoolTimeout,
            "pool",
            0.25,
            (503, "Local upstream connection pool timed out after 0.25 seconds"),
        ),
        (
            httpx.WriteTimeout,
            "write",
            30,
            (504, "Request write timed out after 30 seconds"),
        ),
    ],
)
def test_provider_error_classifier_reports_transport_timeout_phase_setting(
    error_type,
    timeout_key,
    timeout_value,
    expected,
):
    classifier = ProviderErrorClassifier(_safe_get)
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/responses",
        extensions={"timeout": {timeout_key: timeout_value}},
    )

    assert classifier.normalize_exception(
        error_type("timed out", request=request)
    ) == expected


@pytest.mark.parametrize(
    "details",
    [
        (
            "模型 claude-opus-5 的价格尚未由管理员配置，暂时无法使用，请联系站点管理员开启该模型；"
            "Model claude-opus-5 has not been priced by the administrator yet."
        ),
        {
            "error": {
                "type": "upstream_error",
                "code": "model_price_not_configured",
                "message": "model pricing is not available for this route",
            }
        },
        {
            "detail": {
                "code": "model_not_priced",
                "message": "The model has not been priced by administrator yet",
            }
        },
    ],
)
def test_provider_error_classifier_remaps_model_pricing_400_for_failover(details):
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)

    assert classifier.is_model_pricing_unconfigured_error(400, details) is True
    assert classifier.remap_status_code(400, str(details)) == 502
    assert retry_policy.should_retry(
        True,
        400,
        {"base_url": "https://provider.example/v1/messages"},
        error_message=str(details),
    ) is True


def test_provider_error_classifier_keeps_real_bad_request_non_retryable():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    details = {
        "error": {
            "type": "invalid_request_error",
            "message": "messages: field required",
        }
    }

    assert classifier.is_model_pricing_unconfigured_error(400, details) is False
    assert classifier.remap_status_code(400, str(details)) == 400
    assert retry_policy.should_retry(
        True,
        400,
        {"base_url": "https://provider.example/v1/messages"},
        error_message=str(details),
    ) is False


def test_provider_error_classifier_does_not_remap_pricing_text_on_non400_status():
    classifier = ProviderErrorClassifier(_safe_get)
    details = "model has not been priced by the administrator yet"

    assert classifier.is_model_pricing_unconfigured_error(503, details) is False
    assert classifier.remap_status_code(503, details) == 503


@pytest.mark.parametrize(
    ("read_timeout", "expected"),
    [
        (20, "Request timed out after 20 seconds"),
        (20.5, "Request timed out after 20.5 seconds"),
        ("30", "Request timed out after 30 seconds"),
    ],
)
def test_provider_error_classifier_reports_configured_read_timeout(read_timeout, expected):
    classifier = ProviderErrorClassifier(_safe_get)
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/responses",
        extensions={"timeout": {"read": read_timeout}},
    )

    assert classifier.normalize_exception(httpx.ReadTimeout("timed out", request=request)) == (
        504,
        expected,
    )


@pytest.mark.parametrize(
    "extensions",
    [
        {},
        {"timeout": {"read": None}},
        {"timeout": {"read": -1}},
        {"timeout": {"read": float("inf")}},
        {"timeout": {"read": "invalid"}},
    ],
)
def test_provider_error_classifier_never_reports_unknown_or_invalid_timeout(extensions):
    classifier = ProviderErrorClassifier(_safe_get)
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/responses",
        extensions=extensions,
    )

    assert classifier.normalize_exception(httpx.ReadTimeout("timed out", request=request)) == (
        504,
        "Request timed out",
    )


def test_provider_error_classifier_preserves_local_upstream_admission_503():
    classifier = ProviderErrorClassifier(_safe_get)

    class LocalAdmissionError(Exception):
        status_code = 503
        reason = "upstream_wait_timeout"
        local_admission_rejection = True

    assert classifier.normalize_exception(LocalAdmissionError()) == (
        503,
        "upstream_wait_timeout",
    )


def test_provider_error_classifier_preserves_responses_semantic_400():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    error = responses_failure_error(
        {
            "error": {
                "code": "oaix_gateway_error",
                "message": "Your input exceeds the context window of this model.",
                "status": 400,
                "type": "gateway_error",
            }
        },
        event_type="error",
    )

    assert error is not None
    status_code, detail = classifier.normalize_exception(error)
    assert status_code == 400
    assert '"code":"oaix_gateway_error"' in detail
    assert retry_policy.should_retry(
        True,
        status_code,
        {"base_url": "https://example.com/v1/responses"},
        error_message=detail,
        endpoint="/v1/chat/completions",
        original_model="gpt-5.5",
    ) is False


def test_responses_failure_error_fast_returns_for_ordinary_delta():
    class DeltaPayload(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError("ordinary delta must not inspect payload values")

    payload = DeltaPayload(delta="hello", sequence_number=1)

    assert responses_failure_error(
        payload,
        event_type="response.output_text.delta",
        wire_status_code=200,
        validated_provider_sse=True,
    ) is None


def test_responses_semantic_error_bounds_attacker_sized_message():
    error = responses_failure_error(
        {
            "type": "error",
            "error": {
                "code": "server_error",
                "message": "x" * (1024 * 1024),
            },
        },
        event_type="error",
    )

    assert error is not None
    assert len(error.message.encode("utf-8")) <= 4096
    assert len(error.detail_json.encode("utf-8")) < 8192
    assert error.message.endswith(" [truncated]")
    assert error.passthrough_error_body is None


def test_response_failed_has_detached_bounded_responses_terminal():
    error = responses_failure_error(
        {
            "type": "response.failed",
            "sequence_number": 7,
            "response": {
                "id": "resp_ctx",
                "object": "response",
                "model": "gpt-test",
                "status": "failed",
                "error": {
                    "code": " Context_Length_Exceeded ",
                    "type": " Invalid_Request_Error ",
                    "message": "x" * (1024 * 1024),
                    "param": "input",
                    "ignored": {"large": "y" * (1024 * 1024)},
                },
                "ignored": ["z" * (1024 * 1024)],
            },
        },
        event_type="response.failed",
        wire_status_code=200,
    )

    assert error is not None
    assert error.sse_payload["type"] == "error"
    assert error.responses_sse_event_type == "response.failed"
    assert error.responses_sse_payload == {
        "type": "response.failed",
        "sequence_number": 7,
        "response": {
            "id": "resp_ctx",
            "object": "response",
            "model": "gpt-test",
            "status": "failed",
            "error": {
                "code": "context_length_exceeded",
                "type": "invalid_request_error",
                "message": error.message,
                "param": "input",
            },
        },
    }
    assert len(str(error.responses_sse_payload).encode("utf-8")) < 8192


def test_preserved_response_failed_http_body_does_not_retain_large_graph():
    ignored = "y" * (7 * 1024 * 1024)
    error = responses_failure_error(
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "context_length_exceeded",
                    "message": "input is too long",
                    "ignored": {"attacker_owned": ignored},
                },
            },
        },
        event_type="response.failed",
        preserve_error_body=True,
    )

    assert error is not None
    assert error.passthrough_error_body == {
        "error": {
            "code": "context_length_exceeded",
            "message": "input is too long",
        }
    }
    assert ignored not in str(error.passthrough_error_body)
    assert len(str(error.passthrough_error_body).encode("utf-8")) < 8192


def test_generic_error_event_is_not_promoted_to_response_failed():
    error = responses_failure_error(
        {
            "type": "error",
            "error": {
                "code": "context_length_exceeded",
                "message": "input is too long",
            },
        },
        event_type="error",
    )

    assert error is not None
    assert error.responses_sse_event_type == "error"
    assert error.responses_sse_payload is error.sse_payload


def test_validated_provider_error_event_has_detached_response_failed_terminal():
    ignored = "y" * (7 * 1024 * 1024)
    error = responses_failure_error(
        {
            "type": "error",
            "sequence_number": 2,
            "error": {
                "code": " Context_Length_Exceeded ",
                "type": " Invalid_Request_Error ",
                "message": "input is too long",
                "param": "input",
                "ignored": {"attacker_owned": ignored},
            },
            "ignored": [ignored],
        },
        event_type="error",
        wire_status_code=200,
        validated_provider_sse=True,
    )

    assert error is not None
    assert error.event_type == "error"
    assert error.status_code == 400
    assert error.responses_sse_event_type == "response.failed"
    assert error.responses_sse_payload == {
        "type": "response.failed",
        "sequence_number": 2,
        "response": {
            "status": "failed",
            "error": {
                "code": "context_length_exceeded",
                "type": "invalid_request_error",
                "message": "input is too long",
                "param": "input",
            },
        },
    }
    assert ignored not in str(error.responses_sse_payload)
    assert len(str(error.responses_sse_payload).encode("utf-8")) < 8192


def test_validated_provider_error_without_message_is_not_promoted():
    error = responses_failure_error(
        {
            "type": "error",
            "error": {"code": "context_length_exceeded"},
        },
        event_type="error",
        validated_provider_sse=True,
    )

    assert error is not None
    assert error.responses_sse_event_type == "error"
    assert error.responses_sse_payload is error.sse_payload


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "missing top-level type"}},
        {
            "type": "response.failed",
            "error": {"message": "conflicting top-level type"},
        },
        {"type": 1, "error": {"message": "non-string top-level type"}},
        {"type": "error", "error": "scalar error"},
        {
            "type": "error",
            "error": {"message": " ", "code": "", "type": "\t"},
        },
        {"type": "error", "error": {"message": 1, "code": True}},
    ],
)
def test_validated_provider_error_rejects_ambiguous_canonicalization(payload):
    error = responses_failure_error(
        payload,
        event_type="error",
        validated_provider_sse=True,
    )

    assert error is not None
    assert error.event_type == "error"
    assert error.responses_sse_event_type == "error"
    assert error.responses_sse_payload is error.sse_payload


def test_response_failed_rejects_non_string_status_without_stringifying():
    class ExplosiveStatus(list):
        def __str__(self):
            raise AssertionError("protocol status must not be stringified")

    error = responses_failure_error(
        {
            "type": "response.failed",
            "response": {
                "status": ExplosiveStatus(["x" * (1024 * 1024)]),
                "error": {
                    "code": "context_length_exceeded",
                    "message": "input is too long",
                },
            },
        },
        event_type="response.failed",
    )

    assert error is None


def test_retry_policy_does_not_retry_missing_persisted_response_item():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    error = {
        "error": {
            "message": "Item with id 'rs_1' not found. Items are not persisted when `store` is set to false.",
            "type": "invalid_request_error",
        }
    }

    assert retry_policy.should_retry(
        True,
        404,
        {"base_url": "https://example.com/v1/responses"},
        error_message=str(error),
        endpoint="/v1/responses",
        original_model="gpt-5.4",
    ) is False


def test_retry_policy_retries_codex_chatgpt_model_unsupported():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)

    assert retry_policy.should_retry(
        True,
        400,
        {"base_url": "https://chatgpt.com/backend-api/codex", "engine": "codex"},
        error_message='{"error":{"message":"model is not supported when using codex with a ChatGPT account"}}',
        endpoint="/v1/responses",
        original_model="gpt-5.5",
    ) is True


def test_cooldown_policy_uses_retry_after_and_configured_minimum():
    classifier = ProviderErrorClassifier(_safe_get)
    cooldown_policy = CooldownPolicy(classifier, _get_engine)
    details = (
        '{"error":{"code":"rate_limit_exceeded",'
        '"message":"Rate limit reached. Please try again in 2500ms."}}'
    )

    assert cooldown_policy.rate_limit_cooling_time(
        {"preferences": {"api_key_rate_limit_cooldown_period": 1}},
        429,
        details,
    ) == 3


def test_cooldown_policy_identifies_quota_and_codex_auth_cooldowns():
    classifier = ProviderErrorClassifier(_safe_get)
    retry_policy = RetryPolicy(classifier, _get_engine)
    cooldown_policy = CooldownPolicy(classifier, _get_engine)

    assert cooldown_policy.should_use_quota_cooldown(
        {"engine": "gpt"},
        429,
        "insufficient_quota",
        endpoint="/v1/responses",
        original_model="gpt-5.4",
        retry_policy=retry_policy,
    ) is True

    assert cooldown_policy.should_use_quota_cooldown(
        {"engine": "codex"},
        403,
        '{"error":{"code":"account_deactivated","message":"account has been deactivated"}}',
        endpoint="/v1/responses",
        original_model="gpt-5.4",
        retry_policy=retry_policy,
    ) is True
