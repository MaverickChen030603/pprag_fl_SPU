#!/usr/bin/env python3
"""Gate-controlled scale-up entrypoint for a frozen 3,000+ query action set."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v4_common import OUTPUTS, REPORTS, ensure_layout, read_json, read_jsonl, write_json


def main() -> None:
    ensure_layout()
    multi_reader = read_json(OUTPUTS / "multi_reader/multi_reader_summary.json")
    if multi_reader.get("status") != "complete" or multi_reader.get("systematic_answer_degradation"):
        reason = "Scale-up is allowed only after successful 1,000-query selector, official, and multi-reader stages."
        payload = {"status": "skipped_by_upstream_gate", "reason": reason}
    else:
        action_path = Path(os.environ.get("V4_FROZEN_SCALEUP_ACTIONS", ""))
        outcome_path = Path(os.environ.get("V4_FROZEN_SCALEUP_OUTCOMES", ""))
        if not action_path.is_file() or not outcome_path.is_file():
            payload = {
                "status": "blocked_missing_frozen_scaleup_inputs",
                "reason": "A pre-generated action table and reader outcomes for at least 3,000 untouched queries are required; the 1,000-query thresholds may not be retuned.",
                "required_environment": ["V4_FROZEN_SCALEUP_ACTIONS", "V4_FROZEN_SCALEUP_OUTCOMES"],
            }
        else:
            actions, outcomes = read_jsonl(action_path), read_jsonl(outcome_path)
            query_count = len({str(row["query_id"]) for row in outcomes})
            if query_count < 3000:
                raise AssertionError(f"Scale-up set has only {query_count} queries")
            payload = {
                "status": "complete",
                "n_queries": query_count,
                "frozen_action_rows": len(actions),
                "reader_outcome_rows": len(outcomes),
                "thresholds_retuned": False,
                "note": "Metric aggregation must be supplied by the frozen scale-up producer manifest.",
            }
    write_json(OUTPUTS / "scaleup/scaleup_summary.json", payload)
    (REPORTS / "scaleup_report.md").write_text(f"# Scale-Up Evaluation\n\nStatus: **{payload['status']}**. {payload.get('reason', payload.get('note', ''))}\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
