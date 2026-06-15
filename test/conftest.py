import asyncio
import inspect
import os
import sys

import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


collect_ignore = [
    "test_dict.py",
    "test_httpx.py",
    "test_json.py",
    "test_nostream.py",
    "test_ruamel_yaml.py",
    "test_siliconflow.py",
    "test_weights.py",
]


RUNTIME_DERIVED_STATE_ATTRS = (
    "runtime_snapshot",
    "runtime_config",
    "runtime_config_source_id",
    "models_list",
    "routing_index",
    "model_response_cache",
    "api_key_index",
)


def _clear_runtime_derived_state(state):
    for attr in RUNTIME_DERIVED_STATE_ATTRS:
        if hasattr(state, attr):
            delattr(state, attr)


@pytest.fixture(autouse=True)
def isolated_main_app_state():
    try:
        import main
    except Exception:
        yield
        return

    state = main.app.state
    before = dict(getattr(state, "_state", {}))
    _clear_runtime_derived_state(state)
    try:
        yield
    finally:
        current = getattr(state, "_state", None)
        if isinstance(current, dict):
            current.clear()
            current.update(before)


def pytest_pyfunc_call(pyfuncitem):
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None

    test_args = {
        arg: pyfuncitem.funcargs[arg]
        for arg in pyfuncitem._fixtureinfo.argnames
    }
    _ensure_main_runtime_config_for_test()
    asyncio.run(pyfuncitem.obj(**test_args))
    return True


def _ensure_main_runtime_config_for_test():
    try:
        import main
    except Exception:
        return

    state = main.app.state
    if hasattr(state, "runtime_config") or not hasattr(state, "config"):
        return
    api_list = getattr(state, "api_list", None)
    if api_list is None:
        api_keys = (getattr(state, "config", {}) or {}).get("api_keys", [])
        api_list = [item.get("api") for item in api_keys if item.get("api")]
        state.api_list = api_list
    models_list = getattr(state, "models_list", None)
    try:
        state.runtime_config = main.compile_runtime_config(
            state.config,
            list(api_list or []),
            models_list=models_list,
            default_timeout=main.DEFAULT_TIMEOUT,
        )
        state.runtime_config_source_id = id(state.config)
        state.models_list = state.runtime_config.api_key_allowed_models
        state.routing_index = state.runtime_config.routing_index
        state.model_response_cache = state.runtime_config.api_key_model_response_cache
        state.api_key_index = state.runtime_config.api_key_index_by_token
    except Exception:
        return
