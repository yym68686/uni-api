#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import secrets
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_PORT = 1455
DEFAULT_TIMEOUT_SECONDS = 300


def _b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


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


def _jwt_claims_no_verify(id_token: str) -> dict:
    parts = (id_token or "").split(".")
    if len(parts) != 3:
        raise ValueError("invalid id_token: expected 3 JWT parts")
    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload_raw = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
    return json.loads(payload_raw.decode("utf-8"))


def _extract_chatgpt_account_id(claims: dict) -> Optional[str]:
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        account_id = auth.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            return account_id.strip()
    return None


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
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"token exchange failed: status {resp.status}: {raw}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"token exchange failed: status {getattr(e, 'code', 'unknown')}: {raw}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"token exchange returned non-JSON: {raw}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Codex OAuth login helper (outputs: <chatgpt_account_id>,<refresh_token>)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="callback port (default: 1455)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="wait timeout seconds (default: 300)")
    parser.add_argument("--no-browser", action="store_true", help="do not open browser automatically")
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
        if not account_id:
            raise RuntimeError(
                "missing chatgpt_account_id in id_token claims (expected claims['https://api.openai.com/auth']['chatgpt_account_id'])"
            )

        # Output for uni-api config: "account_id,refresh_token"
        print(f"{account_id},{refresh_token}")
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
