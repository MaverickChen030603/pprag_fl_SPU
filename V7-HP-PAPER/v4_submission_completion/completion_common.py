#!/usr/bin/env python3
"""Shared paths and helpers for the V4 submission-completion experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("FEDE4RAG_ROOT", HERE.parents[1])).expanduser().resolve()
V4_ROOT = PROJECT_ROOT / "V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
OUTPUTS = HERE / "outputs"
REPORTS = HERE / "reports"
TABLES = HERE / "tables"
PAPER = HERE / "paper"
EXTERNAL = OUTPUTS / "external_2wiki_frozen"
FAITHFUL = OUTPUTS / "faithful_baseline"
ABLATION = OUTPUTS / "generator_ablation"


def ensure_layout() -> None:
    for path in (OUTPUTS, REPORTS, TABLES, PAPER, EXTERNAL, FAITHFUL, ABLATION):
        path.mkdir(parents=True, exist_ok=True)


def add_v4_import_path() -> None:
    value = str(V4_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def query_fingerprint(query_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in query_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_fold(query_id: str, folds: int = 5) -> int:
    return int(hashlib.md5(str(query_id).encode("utf-8")).hexdigest(), 16) % folds

