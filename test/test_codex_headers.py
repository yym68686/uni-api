import asyncio
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import (
    force_codex_client_headers,
    get_codex_payload,
    strip_unsupported_codex_payload_fields,
)


def test_codex_payload_uses_current_cli_version_headers():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "say test"}],
        stream=True,
    )
    provider = {
        "provider": "codex",
        "base_url": "https://chatgpt.com/backend-api/codex/responses",
        "model": ["gpt-5.4"],
    }

    _, headers, _ = asyncio.run(get_codex_payload(request, "codex", provider, api_key="access-token"))

    assert headers["Authorization"] == "Bearer access-token"
    assert headers["Version"] == "0.125.0"
    assert headers["User-Agent"] == "codex_cli_rs/0.125.0"


def test_force_codex_client_headers_removes_stale_case_variants():
    headers = {
        "version": "0.21.0",
        "User-Agent": "yaak",
        "X-Test": "kept",
    }

    force_codex_client_headers(headers)

    assert headers == {
        "X-Test": "kept",
        "Version": "0.125.0",
        "User-Agent": "codex_cli_rs/0.125.0",
    }


def test_codex_payload_strips_unsupported_truncation_field():
    payload = {
        "model": "gpt-5.5",
        "input": [{"role": "user", "content": "hello"}],
        "truncation": "auto",
    }

    strip_unsupported_codex_payload_fields(payload)

    assert "truncation" not in payload


def test_responses_route_overrides_stale_client_codex_version_header():
    main_source = (Path(__file__).resolve().parents[1] / "uni_api" / "runtime.py").read_text()

    assert 'headers.setdefault("Version", CODEX_CLI_VERSION)' in main_source
    assert 'headers.setdefault("Version", http_request.headers.get("Version") or CODEX_CLI_VERSION)' not in main_source
    assert "force_codex_client_headers(headers)" in main_source
