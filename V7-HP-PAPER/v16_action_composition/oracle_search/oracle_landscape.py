"""Import shim for the numbered oracle analysis entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_PATH = Path(__file__).with_name("02_oracle_action_landscape.py")
_SPEC = importlib.util.spec_from_file_location("v16_oracle_landscape_impl", _PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

summarize_query = _MODULE.summarize_query
aggregate = _MODULE.aggregate
decision = _MODULE.decision
