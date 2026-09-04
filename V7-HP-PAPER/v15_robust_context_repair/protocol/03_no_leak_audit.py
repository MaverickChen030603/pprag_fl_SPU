#!/usr/bin/env python3
"""Fail when development/training code reads sealed labels or forbidden features."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("retrieval", "action_generation", "action_scorer", "risk_gate", "multi_reader", "cascade", "baselines")
FORBIDDEN = ("data/sealed", "final_test_labels", "is_support", "answer_presence", "gold_support_count")
ALLOW = {
    ("action_scorer/scorer_common.py", "is_support"),
    ("action_scorer/scorer_common.py", "answer_presence"),
}


def main() -> None:
    findings = []
    for directory in SCAN_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                relative = str(path.relative_to(ROOT))
                if token in text and (relative, token) not in ALLOW:
                    findings.append({"path": relative, "token": token})
    payload = {"status": "pass" if not findings else "fail", "scanned_directories": list(SCAN_DIRS), "forbidden_tokens": list(FORBIDDEN), "findings": findings}
    output = Path(__file__).resolve().parent / "no_leak_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
