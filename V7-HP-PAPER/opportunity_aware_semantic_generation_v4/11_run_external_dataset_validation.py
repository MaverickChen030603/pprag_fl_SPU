#!/usr/bin/env python3
"""Apply the pre-registered 300-query external opportunity gate after Hotpot success."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v4_common import OUTPUTS, REPORTS, ensure_layout, read_json, write_json


def main() -> None:
    ensure_layout()
    scaleup = read_json(OUTPUTS / "scaleup/scaleup_summary.json")
    if scaleup.get("status") != "complete":
        payload = {"status": "skipped_by_upstream_gate", "reason": "External validation is allowed only after the HotpotQA main result and frozen scale-up complete."}
    else:
        summary_path = Path(os.environ.get("V4_EXTERNAL_300_SUMMARY", ""))
        if not summary_path.is_file():
            payload = {
                "status": "blocked_missing_external_300_summary",
                "required_environment": "V4_EXTERNAL_300_SUMMARY",
                "allowed_datasets": ["2WikiMultiHopQA", "MuSiQue"],
            }
        else:
            external = read_json(summary_path)
            overall = float(external["overall_positive_query_coverage"])
            conditional = float(external["non_ceiling_positive_query_coverage"])
            payload = {
                "status": "complete" if overall >= 0.30 or conditional >= 0.45 else "stopped_by_external_opportunity_gate",
                "n_queries": int(external["n_queries"]),
                "overall_positive_query_coverage": overall,
                "non_ceiling_positive_query_coverage": conditional,
                "gate_passed": overall >= 0.30 or conditional >= 0.45,
                "post_hoc_tuning": False,
            }
    write_json(OUTPUTS / "external_dataset/external_validation_summary.json", payload)
    (REPORTS / "external_dataset_report.md").write_text(f"# External Dataset Validation\n\nStatus: **{payload['status']}**.\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
