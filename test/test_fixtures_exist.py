from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_adapter_sse_and_config_fixtures_exist():
    assert (FIXTURE_ROOT / "adapters" / "openai_chat_request.json").is_file()
    assert (FIXTURE_ROOT / "sse" / "responses_output_text_delta.sse").is_file()
    assert (FIXTURE_ROOT / "config" / "minimal.yaml").is_file()


def test_autouse_state_isolation_fixture_is_defined():
    import inspect
    import test.conftest as conftest

    source = inspect.getsource(conftest.isolated_main_app_state)
    assert "main.app.state" in source
    assert "current.clear()" in source
    assert "current.update(before)" in source
