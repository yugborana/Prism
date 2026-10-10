"""Shared pytest configuration for the Prism backend test suite."""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Make `agents`, `services`, ... importable when running `pytest` from backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# The production logger is structlog-backed. When running the pure-logic unit
# tests outside Docker (structlog not installed), fall back to a no-op logger
# so these tests don't need the full service dependency stack.
try:  # pragma: no cover - environment dependent
    import structlog  # noqa: F401
except ImportError:  # pragma: no cover
    class _NoopLogger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    _obs = types.ModuleType("observability")
    _obs.__path__ = []  # mark as package
    _log = types.ModuleType("observability.logging")
    _log.get_logger = lambda *a, **k: _NoopLogger()
    sys.modules.setdefault("observability", _obs)
    sys.modules.setdefault("observability.logging", _log)
