import pytest
import uvicorn

import main


def test_main_cli_preserves_common_uvicorn_overrides_with_safe_protocol(monkeypatch):
    captured = {}
    monkeypatch.setattr(main._runtime, "RESOURCE_ADMISSION_MODE", "legacy")
    for name in (
        "UVICORN_CONNECTION_LIMIT",
        "UVICORN_HTTP_PROTOCOL",
        "BOUNDED_HTTP_PROTOCOL_STATS",
    ):
        monkeypatch.setattr(main._runtime, name, getattr(main._runtime, name))
    monkeypatch.setattr(
        uvicorn.main,
        "callback",
        lambda **parameters: captured.update(parameters),
    )

    main._run_uvicorn_cli(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9123",
            "--log-level",
            "warning",
            "--limit-concurrency",
            "10",
        ]
    )

    assert captured["app"] == "uni_api.runtime:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9123
    assert captured["log_level"] == "warning"
    assert captured["limit_concurrency"] is None
    assert captured["workers"] == 1
    assert captured["http"].__name__ == "BoundedH11Protocol"


def test_main_cli_rejects_multiple_workers_and_unsafe_connection_override(
    monkeypatch,
):
    with pytest.raises(SystemExit, match="--workers must remain 1"):
        main._run_uvicorn_cli(["--workers", "2"])

    monkeypatch.setattr(main._runtime, "RESOURCE_ADMISSION_MODE", "legacy")
    monkeypatch.setattr(main._runtime, "UVICORN_CONNECTION_LIMIT", 10)
    with pytest.raises(SystemExit, match="cannot exceed"):
        main._run_uvicorn_cli(
            [
                "--limit-concurrency",
                "11",
            ]
        )


def test_main_cli_rejects_request_count_limit_in_weighted_mode(monkeypatch):
    monkeypatch.setattr(main._runtime, "RESOURCE_ADMISSION_MODE", "weighted")
    with pytest.raises(SystemExit, match="available only"):
        main._run_uvicorn_cli(["--limit-concurrency", "10"])


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_main_cli_eager_options_exit_cleanly(argument, capsys):
    with pytest.raises(SystemExit) as exited:
        main._run_uvicorn_cli([argument])
    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert "Traceback" not in output
