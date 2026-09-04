from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path("V7-HP-PAPER/outputs/selector_v1_100")
    cases_path = root / "failure_cases.jsonl"
    rows = []
    if cases_path.exists():
        rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "n_failure_cases": len(rows),
        "label_counts": dict(Counter(row.get("failure_label", "unknown") for row in rows)),
        "avg_answer_delta": sum(float(row.get("answer_f1_delta", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_joint_delta": sum(float(row.get("joint_f1_delta", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_support_recall_delta": sum(float(row.get("support_recall_delta", 0.0)) for row in rows) / max(len(rows), 1),
    }
    (root / "failure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
