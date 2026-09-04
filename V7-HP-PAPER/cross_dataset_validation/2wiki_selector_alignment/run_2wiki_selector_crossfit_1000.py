#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    smoke = read_json(ALIGN / "outputs/selector_smoke_300/summary.json")
    gate = smoke.get("gate", {})
    if not gate.get("passed", False):
        result = {
            "status": "skipped_gate_not_passed",
            "reason": "2Wiki selector smoke 300 did not satisfy BM25-relative gate; formal 1000 reader validation is intentionally not run.",
            "smoke_gate": gate,
            "limitation": "2Wiki pipeline works, but v2.3 selector does not yet improve over a strong BM25/lexical baseline.",
        }
        for rel in [
            "outputs/action_table_1000/action_table_summary.json",
            "outputs/selector_crossfit_1000/final_1000_summary.json",
            "outputs/selector_crossfit_1000/significance_report.json",
            "outputs/selector_crossfit_1000/ablation_summary.json",
            "outputs/selector_crossfit_1000/failure_summary.json",
        ]:
            write_json(ALIGN / rel, result)
        (ALIGN / "outputs/selector_crossfit_1000/per_example_delta.jsonl").write_text("", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = {
        "status": "not_implemented_in_this_turn",
        "reason": "Smoke gate passed, but 1000 execution should be launched as a separate monitored reader job.",
        "smoke_gate": gate,
    }
    write_json(ALIGN / "outputs/selector_crossfit_1000/final_1000_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
