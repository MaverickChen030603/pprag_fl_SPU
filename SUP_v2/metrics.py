from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_encrypted_bytes(
    plain_parameter_count: int,
    bytes_per_parameter: int = 4,
    encryption_expansion: float = 8.0,
) -> float:
    return plain_parameter_count * bytes_per_parameter * encryption_expansion


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, record: Mapping) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, records: Iterable[Mapping]) -> None:
    path = Path(path)
    records = list(records)
    ensure_dir(path.parent)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

