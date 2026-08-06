#!/usr/bin/env python3
"""Write the required stage-level Recovery split manifest after both freezes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    payload = {"stage": "R2-A.6_REMP", "final_test_accessed": False, "reader_started": False, "datasets": {}}
    for dataset in ("2wikimultihopqa", "musique"):
        path = args.stage_root / "protocol" / dataset / "recovery_split_manifest.json"
        payload["datasets"][dataset] = json.loads(path.read_text(encoding="utf-8"))
    (args.stage_root / "protocol" / "recovery_split_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": payload["stage"], "datasets": list(payload["datasets"]), "final_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
