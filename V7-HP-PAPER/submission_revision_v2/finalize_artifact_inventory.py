#!/usr/bin/env python3
"""Append local submission documents and scripts to the result inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "artifact_inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8")) if INVENTORY.exists() else {"artifacts": []}
    external_prefixes = ("V7-HP4/", "V7-HP-PAPER/selector_v2_3/")
    external = [
        row
        for row in payload.get("artifacts", [])
        if row.get("path", "").startswith(external_prefixes)
    ]
    local = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == INVENTORY or "__pycache__" in path.parts:
            continue
        local.append(
            {
                "path": str(path.relative_to(HERE)),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output = {
        "generated_by": Path(__file__).name,
        "scope": "External source artifacts plus every submission-v2 file except this self-referential inventory and bytecode caches.",
        "artifact_count": len(external) + len(local),
        "artifacts": external + local,
    }
    INVENTORY.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {INVENTORY} with {output['artifact_count']} entries")


if __name__ == "__main__":
    main()
