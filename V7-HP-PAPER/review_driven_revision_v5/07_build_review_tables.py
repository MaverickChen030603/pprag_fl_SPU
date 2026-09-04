#!/usr/bin/env python3
"""Build paper-ready V5 tables strictly from generated result artifacts."""

from __future__ import annotations

from typing import Any

from v5_common import HERE, V4, read_json, write_text


TABLES = HERE / "outputs" / "tables"


def fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "[NEEDS MEASUREMENT]"
    if isinstance(value, str):
        return value
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[Any]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_recomp() -> str:
    dev = read_json(HERE / "outputs/recomp/recomp_budget_matched_metrics.json")
    holdout = read_json(HERE / "outputs/recomp/recomp_holdout_metrics.json")
    if not dev:
        return "# Budget-Matched Compression Comparison\n\n[NEEDS MEASUREMENT]\n"
    methods = ["frozen_top5_baseline", "full_v4", "recomp_top1", "recomp_budget_660", "baseline_truncated_660"]
    rows = []
    for method in methods:
        value = dev["metrics"][method]
        rows.append([method, fmt(value.get("context_tokens"), 1), fmt(value.get("retained_sentences"), 1), fmt(value.get("represented_documents"), 1), fmt(value["answer_f1"]), fmt(value["sp_f1"]), fmt(value["joint_f1"]), fmt(value.get("reader_latency_seconds_batched"), 4)])
    text = "# Budget-Matched Compression Comparison\n\n" + table(["Method", "Tokens", "Sent.", "Docs", "Answer F1", "SP F1", "Joint F1", "Reader latency"], rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])
    if holdout:
        rows = []
        for method, value in holdout["metrics"].items():
            rows.append([method, fmt(value.get("context_tokens"), 1), fmt(value["answer_f1"]), fmt(value["sp_f1"]), fmt(value["joint_f1"])])
        text += "\n\n## Frozen 3,000 Holdout\n\n" + table(["Method", "Tokens", "Answer F1", "SP F1", "Joint F1"], rows, ["---", "---:", "---:", "---:", "---:"])
    return text + "\n"


def build_lite() -> str:
    dev = read_json(HERE / "outputs/lite_model/lite_nested_metrics.json")
    holdout = read_json(HERE / "outputs/lite_model/lite_holdout_metrics.json")
    decision = read_json(HERE / "outputs/lite_model/lite_architecture_decision.json", {})
    if not dev:
        return "# Full vs Lite\n\n[NEEDS MEASUREMENT]\n"
    rows = []
    for method, value in dev["metrics"].items():
        ni = dev["noninferiority"].get(method)
        rows.append([method, fmt(value["answer_f1"]), fmt(value["sp_f1"]), fmt(value["joint_f1"]), fmt(ni["joint_f1_vs_full"]["delta"], signed=True) if ni else "reference", str(ni["point_estimate_noninferior"]).lower() if ni else "reference", str(ni["ci_noninferior"]).lower() if ni else "reference"])
    text = "# Full vs Lite\n\n" + table(["Method", "Answer F1", "SP F1", "Joint F1", "Joint vs Full", "Point NI", "CI NI"], rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"])
    text += f"\n\nFrozen Lite variant: `{decision.get('selected_variant', '[NEEDS MEASUREMENT]')}`.\n"
    if holdout:
        rows = [[method, fmt(value["answer_f1"]), fmt(value["sp_f1"]), fmt(value["joint_f1"])] for method, value in holdout["metrics"].items()]
        text += "\n## Untouched Revision Holdout (3,405)\n\n" + table(["Method", "Answer F1", "SP F1", "Joint F1"], rows, ["---", "---:", "---:", "---:"])
    return text + "\n"


def build_main() -> str:
    scale = read_json(V4 / "outputs/scaleup/official_metrics/scaleup_official_summary.json", {})
    lite = read_json(HERE / "outputs/lite_model/lite_holdout_metrics.json")
    rows = []
    if scale:
        reader = scale["readers"]["flan"]
        for method, values in reader["metrics"].items():
            rows.append(["Hotpot frozen holdout", 3000, method, fmt(values["answer_f1"]), fmt(values["sp_f1"]), fmt(values["joint_f1"])])
    if lite:
        for method, values in lite["metrics"].items():
            rows.append(["Hotpot revision holdout", 3405, method, fmt(values["answer_f1"]), fmt(values["sp_f1"]), fmt(values["joint_f1"])])
    return "# Main Results\n\n" + (table(["Split", "N", "Method", "Answer F1", "SP F1", "Joint F1"], rows, ["---", "---:", "---", "---:", "---:", "---:"]) if rows else "[NEEDS MEASUREMENT]") + "\n"


def build_external() -> str:
    payload = read_json(HERE / "outputs/2wiki_calibration/calibration_results.json")
    if not payload:
        return "# 2Wiki Transfer and Calibration\n\n[NEEDS MEASUREMENT]\n"
    zero = payload["zero_shot"]
    rows = [["zero-shot", 0, "frozen", fmt(zero["coverage"], 3), fmt(zero["answer_drop_rate"], 3), fmt(zero["metrics"]["answer_f1"]), fmt(zero["metrics"]["sp_f1"]), fmt(zero["metrics"]["joint_f1"])]]
    for k, methods in payload["summary"].items():
        for method, value in methods.items():
            rows.append(["few-shot", k, method, fmt(value["coverage_mean"], 3), fmt(value["answer_drop_rate_mean"], 3), fmt(value["answer_f1_mean"]), fmt(value["sp_f1_mean"]), fmt(value["joint_f1_mean"])])
    return "# 2Wiki Transfer and Calibration\n\n" + table(["Setting", "K", "Method", "Coverage", "Answer-drop", "Answer F1", "SP F1", "Joint F1"], rows, ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:"]) + "\n"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "recomp_budget_matched.md": build_recomp(),
        "full_vs_lite.md": build_lite(),
        "main_results_v5.md": build_main(),
        "external_transfer_calibration.md": build_external(),
    }
    for name, text in outputs.items():
        write_text(TABLES / name, text)
    cost = HERE / "outputs/tables/complexity_and_latency.md"
    if not cost.exists():
        write_text(cost, "# Complexity and Latency\n\n[NEEDS MEASUREMENT]\n")
    print("\n".join(str(TABLES / name) for name in outputs))


if __name__ == "__main__":
    main()
