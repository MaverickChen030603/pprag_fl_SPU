#!/usr/bin/env python3
"""Combine per-dataset R2-A.6 no-leak audits into the required stage artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    audits = {}
    for dataset in ("2wikimultihopqa", "musique"):
        path = args.stage_root / "protocol" / dataset / "no_leak_audit.json"
        audits[dataset] = json.loads(path.read_text(encoding="utf-8"))
    payload = {"stage": "R2-A.6_REMP", "status": "pass" if all(audit["status"] == "pass" for audit in audits.values()) else "fail", "datasets": audits, "reader_start_decision": "blocked_before_reader"}
    (args.stage_root / "protocol" / "no_leak_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
