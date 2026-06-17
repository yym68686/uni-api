#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_PORT = 1455
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_SUB2API_OUTPUT = "sub2api-data.json"
DEFAULT_PRIVACY_MODE = "training_set_cf_blocked"
DEFAULT_CONCURRENCY = 10
DEFAULT_PRIORITY = 1
DEFAULT_RATE_MULTIPLIER = 1
DEFAULT_AUTO_PAUSE_ON_EXPIRED = True


def _b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _gen_pkce() -> tuple[str, str]:
    # Match CLIProxyAPI:
    # - verifier: 96 random bytes -> base64url (no padding) => 128 chars
    # - challenge: base64url(sha256(verifier)) no padding
    verifier = _b64url_no_pad(secrets.token_bytes(96))
    challenge = _b64url_no_pad(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _gen_state() -> str:
    # Match CLIProxyAPI: 16 random bytes hex-encoded.
    return secrets.token_hex(16)


def _read_response_text(response: Any) -> str:
    chunks: list[bytes] = []
    while True:
        raw = response.read(65536)
        if not raw:
            break
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        chunks.append(bytes(raw))
    return b"".join(chunks).decode("utf-8", errors="replace")


def _redact_token_response_text(text: str, *, limit: int = 1000) -> str:
    redacted = re.sub(
        r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]*',
        r"\1<redacted>",
        text,
    )
    if len(redacted) > limit:
        return redacted[:limit] + f"... <truncated {len(redacted) - limit} chars>"
    return redacted


def _post_token_form_with_curl(body: bytes, timeout: int = 30) -> tuple[int, str]:
    if not shutil.which("curl"):
        raise FileNotFoundError("curl not found")

    status_marker = "\n__CODEX_OAUTH_HTTP_STATUS__:"
    proc = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            str(timeout),
            "--request",
            "POST",
            "--header",
            "Content-Type: application/x-www-form-urlencoded",
            "--header",
            "Accept: application/json",
            "--data-binary",
            "@-",
            "--write-out",
            f"{status_marker}%{{http_code}}",
            OPENAI_TOKEN_URL,
        ],
        input=body,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    output = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if status_marker not in output:
        detail = stderr or _redact_token_response_text(output)
        raise RuntimeError(f"token exchange failed: curl exit {proc.returncode}: {detail}")

    raw, status_text = output.rsplit(status_marker, 1)
    try:
        status = int(status_text.strip())
    except ValueError as e:
        raise RuntimeError(f"token exchange failed: invalid curl HTTP status: {status_text!r}") from e

    if proc.returncode != 0 and not raw:
        raise RuntimeError(f"token exchange failed: curl exit {proc.returncode}: {stderr}")
    return status, raw


def _post_token_form_with_urllib(body: bytes, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(
        OPENAI_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _read_response_text(resp)
    except urllib.error.HTTPError as e:
        raw = _read_response_text(e) if hasattr(e, "read") else str(e)
        return int(getattr(e, "code", 0) or 0), raw


def _jwt_claims_no_verify(id_token: str) -> dict:
    parts = (id_token or "").split(".")
    if len(parts) != 3:
        raise ValueError("invalid id_token: expected 3 JWT parts")
    payload_raw = _b64url_decode(parts[1])
    return json.loads(payload_raw.decode("utf-8"))


def _nested_sections(section_name: str, *claims_list: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for claims in claims_list:
        section = claims.get(section_name)
        if isinstance(section, dict):
            sections.append(section)
    return sections


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _nested_value(section_name: str, key: str, *claims_list: dict[str, Any]) -> Any:
    for section in _nested_sections(section_name, *claims_list):
        value = _first_non_empty(section.get(key))
        if value is not None:
            return value
    return None


def _coerce_timestamp(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_expires_at(source: dict[str, Any], access_claims: dict[str, Any], exported_at: datetime) -> int | None:
    access_exp = _coerce_timestamp(access_claims.get("exp"))
    if access_exp is not None:
        return access_exp

    expires_in = _coerce_timestamp(source.get("expires_in"))
    if expires_in is not None:
        return int(exported_at.timestamp()) + max(expires_in, 0)
    return None


def _resolve_organization_id(auth_claims: dict[str, Any]) -> str | None:
    organizations = auth_claims.get("organizations")
    if not isinstance(organizations, list):
        return None
    for item in organizations:
        if not isinstance(item, dict):
            continue
        org_id = _first_non_empty(item.get("id"))
        if org_id is not None:
            return str(org_id)
    return None


def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_chatgpt_account_id(claims: dict) -> Optional[str]:
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        account_id = auth.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            return account_id.strip()
    return None


def build_sub2api_payload(
    source: dict[str, Any],
    *,
    exported_at: datetime | None = None,
    local_tz=None,
    privacy_mode: str = DEFAULT_PRIVACY_MODE,
    concurrency: int = DEFAULT_CONCURRENCY,
    priority: int = DEFAULT_PRIORITY,
    rate_multiplier: int | float = DEFAULT_RATE_MULTIPLIER,
    auto_pause_on_expired: bool = DEFAULT_AUTO_PAUSE_ON_EXPIRED,
) -> dict[str, Any]:
    if local_tz is None:
        local_tz = datetime.now().astimezone().tzinfo

    if exported_at is None:
        exported_at = datetime.now(timezone.utc).replace(microsecond=0)
        privacy_checked_at = datetime.now(local_tz)
    else:
        if exported_at.tzinfo is None:
            exported_at = exported_at.replace(tzinfo=timezone.utc)
        privacy_checked_at = exported_at.astimezone(local_tz)

    access_token = str(source.get("access_token") or "")
    id_token = str(source.get("id_token") or "")
    refresh_token = str(source.get("refresh_token") or "")

    access_claims = _jwt_claims_no_verify(access_token)
    id_claims = _jwt_claims_no_verify(id_token)
    auth_claims = _nested_sections("https://api.openai.com/auth", id_claims, access_claims)

    email = _first_non_empty(
        id_claims.get("email"),
        _nested_value("https://api.openai.com/profile", "email", id_claims, access_claims),
    )
    account_id = _nested_value("https://api.openai.com/auth", "chatgpt_account_id", id_claims, access_claims)
    user_id = _nested_value("https://api.openai.com/auth", "chatgpt_user_id", id_claims, access_claims)
    plan_type = _nested_value("https://api.openai.com/auth", "chatgpt_plan_type", id_claims, access_claims)

    organization_id = None
    for auth_claim in auth_claims:
        organization_id = _resolve_organization_id(auth_claim)
        if organization_id is not None:
            break

    last_refresh_ts = _coerce_timestamp(_first_non_empty(access_claims.get("iat"), id_claims.get("iat")))
    last_refresh = None
    if last_refresh_ts is not None:
        last_refresh = datetime.fromtimestamp(last_refresh_ts, tz=timezone.utc).astimezone(local_tz).isoformat()

    expires_at = _resolve_expires_at(source, access_claims, exported_at)
    if expires_at is not None:
        expires_in = max(expires_at - int(exported_at.timestamp()), 0)
    else:
        expires_in = _coerce_timestamp(source.get("expires_in"))

    display_name = _first_non_empty(email, account_id, user_id, "openai-oauth-account")
    email_key = _first_non_empty(email, display_name)

    account: dict[str, Any] = {
        "name": display_name,
        "platform": "openai",
        "type": "oauth",
        "credentials": {
            "access_token": access_token,
            "email": email,
            "chatgpt_account_id": account_id,
            "chatgpt_user_id": user_id,
            "expires_at": expires_at,
            "expires_in": expires_in,
            "id_token": id_token,
            "organization_id": organization_id,
            "refresh_token": refresh_token,
            "plan_type": plan_type,
        },
        "extra": {
            "email": email,
            "email_key": email_key,
            "last_refresh": last_refresh,
            "privacy_mode": privacy_mode,
            "privacy_checked_at": privacy_checked_at.isoformat(),
        },
        "concurrency": concurrency,
        "priority": priority,
        "rate_multiplier": rate_multiplier,
        "expires_at": expires_at,
        "auto_pause_on_expired": auto_pause_on_expired,
    }

    return {
        "type": "sub2api-data",
        "version": 1,
        "exported_at": _to_utc_z(exported_at),
        "proxies": [],
        "accounts": [account],
    }


@dataclass
class OAuthCallback:
    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


class _OAuthHandler(BaseHTTPRequestHandler):
    expected_state: str = ""
    callback: OAuthCallback = OAuthCallback()
    done: threading.Event = threading.Event()

    def log_message(self, *_args, **_kwargs) -> None:
        # Keep output clean (tokens may appear in URL query in some cases).
        return

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/success":
            self._send_html(
                200,
                "<html><body><h3>Codex OAuth success</h3><p>You can close this tab.</p></body></html>",
            )
            return

        if parsed.path != "/auth/callback":
            self._send_html(404, "<html><body>Not Found</body></html>")
            return

        query = urllib.parse.parse_qs(parsed.query)
        code = (query.get("code") or [None])[0]
        state = (query.get("state") or [None])[0]
        err = (query.get("error") or [None])[0]
        err_desc = (query.get("error_description") or [None])[0]

        if err:
            _OAuthHandler.callback = OAuthCallback(error=err, error_description=err_desc)
            _OAuthHandler.done.set()
            self._send_html(400, f"<html><body><h3>OAuth error</h3><pre>{err}</pre></body></html>")
            return

        if not code or not state:
            _OAuthHandler.callback = OAuthCallback(error="missing_code_or_state")
            _OAuthHandler.done.set()
            self._send_html(400, "<html><body><h3>OAuth error</h3><pre>missing code/state</pre></body></html>")
            return

        if state != _OAuthHandler.expected_state:
            _OAuthHandler.callback = OAuthCallback(error="state_mismatch", state=state, code=code)
            _OAuthHandler.done.set()
            self._send_html(400, "<html><body><h3>OAuth error</h3><pre>state mismatch</pre></body></html>")
            return

        _OAuthHandler.callback = OAuthCallback(code=code, state=state)
        _OAuthHandler.done.set()

        self.send_response(302)
        self.send_header("Location", "/success")
        self.end_headers()


def _exchange_code_for_tokens(code: str, code_verifier: str, redirect_uri: str) -> dict:
    data = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    body = urllib.parse.urlencode(data).encode("utf-8")

    try:
        status, raw = _post_token_form_with_curl(body)
    except (FileNotFoundError, OSError):
        status, raw = _post_token_form_with_urllib(body)

    if status != 200:
        raise RuntimeError(f"token exchange failed: status {status}: {_redact_token_response_text(raw)}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        detail = _redact_token_response_text(raw)
        raise RuntimeError(f"token exchange returned incomplete/non-JSON response ({len(raw)} bytes): {detail}") from e


def _render_login_outputs(account_id: str, refresh_token: str, sub2api_payload: dict) -> str:
    return (
        f"{account_id},{refresh_token}\n\n"
        f"{json.dumps(sub2api_payload, ensure_ascii=False, indent=2)}"
    )


def _write_sub2api_output(sub2api_payload: dict, output_path: str) -> Path:
    path = Path(output_path).expanduser()
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sub2api_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Codex OAuth login helper (outputs: <chatgpt_account_id>,<refresh_token> and sub2api JSON)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="callback port (default: 1455)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="wait timeout seconds (default: 300)")
    parser.add_argument("--no-browser", action="store_true", help="do not open browser automatically")
    parser.add_argument(
        "--sub2api-output",
        default=DEFAULT_SUB2API_OUTPUT,
        help=f"write sub2api JSON to this file (default: {DEFAULT_SUB2API_OUTPUT})",
    )
    args = parser.parse_args()

    redirect_uri = f"http://localhost:{args.port}/auth/callback"
    state = _gen_state()
    code_verifier, code_challenge = _gen_pkce()

    params = {
        "client_id": OPENAI_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid email profile offline_access",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    auth_url = f"{OPENAI_AUTH_URL}?{urllib.parse.urlencode(params)}"

    print(f"[1/3] Starting callback server on {redirect_uri}", file=sys.stderr)
    _OAuthHandler.expected_state = state
    _OAuthHandler.done = threading.Event()
    _OAuthHandler.callback = OAuthCallback()

    httpd = HTTPServer(("127.0.0.1", args.port), _OAuthHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        print("[2/3] Open the browser to continue authentication:", file=sys.stderr)
        print(auth_url, file=sys.stderr)
        if not args.no_browser:
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass

        if not _OAuthHandler.done.wait(timeout=args.timeout):
            raise RuntimeError("timeout waiting for OAuth callback")

        cb = _OAuthHandler.callback
        if cb.error:
            desc = f": {cb.error_description}" if cb.error_description else ""
            raise RuntimeError(f"oauth callback error: {cb.error}{desc}")
        if not cb.code:
            raise RuntimeError("oauth callback missing code")

        print("[3/3] Exchanging code for tokens...", file=sys.stderr)
        token_resp = _exchange_code_for_tokens(cb.code, code_verifier, redirect_uri)

        refresh_token = str(token_resp.get("refresh_token") or "").strip()
        if not refresh_token:
            raise RuntimeError(f"missing refresh_token in token response: {json.dumps(token_resp, ensure_ascii=False)}")

        id_token = str(token_resp.get("id_token") or "").strip()
        if not id_token:
            raise RuntimeError("missing id_token in token response (cannot extract chatgpt_account_id)")

        claims = _jwt_claims_no_verify(id_token)
        account_id = _extract_chatgpt_account_id(claims)
        print("account_id", account_id, file=sys.stderr)
        print("token_resp", token_resp, file=sys.stderr)
        if not account_id:
            raise RuntimeError(
                "missing chatgpt_account_id in id_token claims (expected claims['https://api.openai.com/auth']['chatgpt_account_id'])"
            )

        sub2api_payload = build_sub2api_payload(token_resp)
        saved_path = _write_sub2api_output(sub2api_payload, args.sub2api_output)
        print(f"sub2api saved to {saved_path}", file=sys.stderr)

        # Output for uni-api config: "account_id,refresh_token"
        print(_render_login_outputs(account_id, refresh_token, sub2api_payload))
        return 0
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
