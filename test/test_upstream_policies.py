import httpx
from fastapi import HTTPException

from uni_api.upstream.policies import CooldownPolicy, ProviderErrorClassifier, RetryPolicy


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
