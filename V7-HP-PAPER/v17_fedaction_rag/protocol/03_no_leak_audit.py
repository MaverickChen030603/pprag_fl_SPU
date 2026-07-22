#!/usr/bin/env python3
"""Static no-leak scan for V17 inference and federated-training code."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNED = (
    "retrieval", "routing", "centralized_selector", "federated_training",
    "personalization", "risk_gate", "communication", "baselines", "evaluation",
)
FORBIDDEN = (
    "sealed_label", "final_test_labels", "is_supporting", "is_support",
    "gold_answer", "answer_presence", "supporting_facts", "supporting_titles",
    "reader_generated_answer", "local_training_contexts",
)


def main() -> None:
    findings = []
    for directory in SCANNED:
        root = ROOT / directory
        for path in root.rglob("*.py"):
            lowered = path.read_text(encoding="utf-8").lower()
            for token in FORBIDDEN:
                if token in lowered:
                    findings.append({"path": str(path.relative_to(ROOT)), "token": token})
    status = "fail" if findings else "pass"
    payload = {"status": status, "scanned": list(SCANNED), "forbidden": list(FORBIDDEN), "findings": findings}
    output = Path(__file__).with_name("no_leak_audit.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
