#!/usr/bin/env python3
"""Static no-leak audit for V16 inference-time modules."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN = ("retrieval", "action_atoms", "composer", "scorer", "risk_gate", "baselines", "efficiency")
FORBIDDEN = ("sealed_label", "final_test_labels", "is_supporting", "is_support", "gold_answer", "answer_presence", "supporting_facts", "supporting_titles")


def main() -> None:
    findings = []
    for directory in SCAN:
        for path in (ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                if token in text:
                    findings.append({"path": str(path.relative_to(ROOT)), "token": token})
    payload = {"status": "pass" if not findings else "fail", "scanned": list(SCAN), "forbidden": list(FORBIDDEN), "findings": findings}
    Path(__file__).with_name("no_leak_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
