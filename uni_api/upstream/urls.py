from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse


LINGJING_UPSTREAM_OPENAPI_PREFIX = "/api/entrance/openapi"


def normalize_responses_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    if engine != "codex":
        return base
    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return base
    return f"{base}/responses"


def normalize_responses_compact_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")

    if base.endswith("/v1/responses/compact") or base.endswith("/responses/compact"):
        return base

    if engine == "codex":
        base = normalize_responses_upstream_url(base, engine)

    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return f"{base}/compact"
    if base.endswith("/compact"):
        return base
    return f"{base}/compact"


def normalize_messages_upstream_url(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    if base.endswith("/v1/messages") or base.endswith("/messages"):
        return base
    return f"{base}/messages"


def normalize_content_generation_tasks_upstream_url(base_url: str, task_id: Optional[str] = None) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")

    if path.endswith("/contents/generations/tasks"):
        tasks_url = base
    elif path in ("", "/"):
        tasks_url = f"{base}/api/v3/contents/generations/tasks"
    else:
        tasks_url = f"{base}/contents/generations/tasks"

    if task_id is not None:
        tasks_url = f"{tasks_url}/{quote(str(task_id), safe='')}"
    return tasks_url


def normalize_lingjing_openapi_upstream_url(base_url: str, openapi_path: str, query: str = "") -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return base

    path = "/" + str(openapi_path or "").strip("/")
    if not path.startswith("/openapi/"):
        path = "/openapi" + path

    parsed = urlparse(base)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/api/entrance/openapi"):
        upstream_path = base_path + path[len("/openapi") :]
    elif base_path.endswith("/api/entrance"):
        upstream_path = base_path + path
    else:
        upstream_path = base_path + LINGJING_UPSTREAM_OPENAPI_PREFIX + path[len("/openapi") :]

    url = urlunparse(parsed[:2] + (upstream_path,) + ("",) * 3)
    return f"{url}?{query}" if query else url


def lingjing_upstream_query(raw_query: str) -> str:
    pairs = parse_qsl(raw_query or "", keep_blank_values=True)
    filtered = [(key, value) for key, value in pairs if key not in {"model", "request_model"}]
    return urlencode(filtered, doseq=True)


def normalize_lingjing_draw_task_upstream_url(
    base_url: str,
    *,
    method: str,
    task_id: Optional[str] = None,
) -> str:
    if method.upper() == "POST":
        return normalize_lingjing_openapi_upstream_url(base_url, "/draw/task/submit")
    if method.upper() == "GET":
        if not task_id:
            return normalize_lingjing_openapi_upstream_url(base_url, "/draw/task/query")
        return normalize_lingjing_openapi_upstream_url(
            base_url,
            "/draw/task/query",
            query=f"taskId={quote(str(task_id), safe='')}",
        )
    return ""

