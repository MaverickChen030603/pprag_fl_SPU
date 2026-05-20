from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


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


def utility_per_payload(total_utility: float, payload_ratio: float) -> float:
    return float(total_utility) / max(float(payload_ratio), 1e-12)


def selection_entropy(upload_blocks: Sequence[str]) -> float:
    counts = {}
    total = 0
    for block in upload_blocks:
        if block == "__ALL__":
            continue
        counts[block] = counts.get(block, 0) + 1
        total += 1
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log(prob + 1e-12)
    return entropy


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
