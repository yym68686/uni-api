"""Compatibility entry point for the refactored application."""

from __future__ import annotations

import os as _os
import sys as _sys
import types as _types

import uni_api.runtime as _runtime

for _name in dir(_runtime):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_runtime, _name)

__all__ = tuple(name for name in globals() if not name.startswith("__"))


class _MainModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__") and hasattr(_runtime, name):
            setattr(_runtime, name, value)


_sys.modules[__name__].__class__ = _MainModule


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "uni_api.runtime:app",
        host="0.0.0.0",
        port=int(_os.getenv("PORT", "8000")),
    )
