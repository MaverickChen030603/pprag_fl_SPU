#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
CDV = ROOT / "V7-HP-PAPER/cross_dataset_validation"
REPORTS = CDV / "reports"
MIRROR = ROOT / "实验分析报告/V7-HP-PAPER"


def load(rel: str):
    return json.loads((CDV / rel).read_text(encoding="utf-8"))


def fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def method_table(summary: dict, metric_key: str = "metrics") -> list[str]:
    rows = [
        "| method | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = summary.get(metric_key, {})
    for method in summary.get("methods", metrics.keys()):
        m = metrics.get(method, {})
        rows.append(
            f"| {method} | {m.get('n', summary.get('n', ''))} | "
            f"{fmt(m.get('answer_access_at_k', m.get('answer_access@5', '')))} | "
            f"{fmt(m.get('support_recall_at_k', m.get('support_recall@5', '')))} | "
            f"{fmt(m.get('sp_f1', ''))} | "
            f"{fmt(m.get('answer_em', ''))} | "
            f"{fmt(m.get('answer_f1', ''))} | "
            f"{fmt(m.get('joint_f1', m.get('joint_access@5', '')))} |"
        )
    return rows


def retrieval_table(summary: dict) -> list[str]:
    rows = [
        "| method | n | answer_access@5 | support_recall@5 | all_support_access@5 | joint_access@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = summary.get("metrics", {})
    for method in summary.get("methods", metrics.keys()):
        m = metrics.get(method, {})
        rows.append(
            f"| {method} | {summary.get('n', '')} | {fmt(m.get('answer_access@5', ''))} | "
            f"{fmt(m.get('support_recall@5', ''))} | {fmt(m.get('all_support_access@5', ''))} | "
            f"{fmt(m.get('joint_access@5', ''))} |"
        )
    return rows


def delta_table(summary: dict, metrics: list[str]) -> list[str]:
    rows = ["| metric | mean_delta | wins | losses | ties |", "|---|---:|---:|---:|---:|"]
    dct = summary.get("deltas_vs_context_order", {})
    for metric in metrics:
        d = dct.get(metric, {})
        rows.append(
            f"| {metric} | {fmt(d.get('mean_delta', 0))} | {d.get('wins', 0)} | "
            f"{d.get('losses', 0)} | {d.get('ties', 0)} |"
        )
    return rows


def main() -> None:
    adapter = load("outputs/2wiki_adapter/adapter_summary.json")
    retrieval = load("outputs/2wiki_smoke_300/summary.json")
    reader = load("outputs/2wiki_reader_smoke_300/reader_summary.json")

    dev = adapter["dev"]
    test = adapter["test"]
    lines = [
        "# V7-HP-PAPER 2WikiMultiHopQA Cross-Dataset Current Report",
        "",
        "## 1. Current Status",
        "",
        "2WikiMultiHopQA cross-dataset preparation and dev-300 smoke validation are complete. No 2Wiki/V7-HP-PAPER process is currently running on the server.",
        "",
        "Completed artifacts:",
        "",
        "- data preparation with answer, context docs, and dev support/evidence labels",
        "- retrieval/access dev-300 smoke",
        "- reader-backed dev-300 smoke with `google/flan-t5-large`",
        "",
        "Not yet completed:",
        "",
        "- formal 1000-sample 2Wiki reader validation",
        "- direct migration of Hotpot frozen selector v2.3 action-table model to 2Wiki action features",
        "- statistical significance report for 2Wiki final validation",
        "",
        "## 2. Data Readiness",
        "",
        f"- Adapter status: `{adapter.get('status')}`",
        f"- Preferred eval split: `{adapter.get('preferred_eval_split')}`",
        f"- Dev examples: `{dev['num_examples']}`",
        f"- Dev answer/context/support/evidence: `{dev['num_with_answer']}` / `{dev['num_with_context']}` / `{dev['num_with_supporting_facts']}` / `{dev['num_with_evidences']}`",
        f"- Dev avg context docs: `{dev['avg_context_docs']:.1f}`",
        f"- Dev avg support docs: `{dev['avg_support_docs']:.4f}`",
        f"- Test examples: `{test['num_examples']}`",
        f"- Test support/evidence sentence labels: `{test['num_with_supporting_facts']}` / `{test['num_with_evidences']}`",
        "",
        "Conclusion: use dev for support/evidence/joint metrics. Test is prepared but should not be used for sentence-level support F1 unless an ID-level metric is added.",
        "",
        "## 3. Retrieval/Access Smoke",
        "",
        "Configuration: stratified dev 300, Top-5, no reader. Labels are used only for metric computation.",
        "",
        *retrieval_table(retrieval),
        "",
        "Delta vs context order:",
        "",
        *delta_table(retrieval, ["answer_access@5", "support_recall@5", "all_support_access@5", "joint_access@5"]),
        "",
        "Interpretation: the 2Wiki adapter is not only field-complete; lexical/BM25 ranking substantially improves support and joint access, so the dataset is suitable for reader-backed cross-dataset validation.",
        "",
        "## 4. Reader-Backed Smoke",
        "",
        f"Configuration: stratified dev `{reader['n']}`, `{reader['num_prompts']}` reader prompts, Top-5, reader `{reader['reader_model']}`, batch size `{reader['reader_batch_size']}`.",
        "",
        *method_table(reader),
        "",
        "Delta vs context order:",
        "",
        *delta_table(reader, ["answer_access_at_k", "support_recall_at_k", "sp_f1", "answer_em", "answer_f1", "joint_f1"]),
        "",
        "Interpretation: the reader-backed smoke shows a clear positive transfer signal for the fixed lexical/BM25 routing baseline over raw context order. The strongest gains appear in support routing (`sp_f1 +0.3529`) and propagate into answer and joint scores (`answer_f1 +0.1263`, `joint_f1 +0.2507`).",
        "",
        "## 5. Claim Boundary",
        "",
        "This is a cross-dataset smoke result, not a formal external generalization endpoint. The current 2Wiki method is a frozen, no-training lexical/BM25 selector connected to the reader pipeline. It is not yet the Hotpot frozen selector v2.3 action-table model directly transferred to 2Wiki.",
        "",
        "Paper-safe wording:",
        "",
        "> We prepared 2WikiMultiHopQA and verified that the V7-HP-PAPER reader/evaluation path can consume a new multi-hop QA dataset. In a dev-300 smoke check, a non-leaky lexical routing baseline substantially improves reader-backed support and joint metrics over raw context order. Formal cross-dataset claims require a larger validation run and selector-feature alignment.",
        "",
        "## 6. Artifacts",
        "",
        "```text",
        "V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/adapter_summary.json",
        "V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_smoke_300/summary.json",
        "V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_reader_smoke_300/reader_summary.json",
        "V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_reader_smoke_300/per_example_reader.jsonl",
        "V7-HP-PAPER/cross_dataset_validation/reports/2wiki_data_preparation_report.md",
        "V7-HP-PAPER/cross_dataset_validation/reports/2wiki_smoke_300_report.md",
        "V7-HP-PAPER/cross_dataset_validation/reports/2wiki_reader_smoke_300_report.md",
        "```",
        "",
        "## 7. Recommended Next Step",
        "",
        "Run a bounded 1000-sample 2Wiki reader validation only after deciding whether the paper needs external validation as a main claim or an appendix robustness check. If the goal is method transfer rather than dataset plumbing, first build a 2Wiki action-feature table compatible with selector v2.3; otherwise the current result should be described as a reader-backed lexical routing smoke.",
        "",
    ]

    text = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)
    out1 = REPORTS / "2wiki_current_progress_report_20260622.md"
    out2 = REPORTS / "2wiki_current_progress_report_latest.md"
    out3 = MIRROR / "2wiki_current_progress_report_20260622.md"
    out4 = MIRROR / "2wiki_current_progress_report_latest.md"
    for p in [out1, out2, out3, out4]:
        p.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "reports": [str(out1), str(out2), str(out3), str(out4)],
        "reader_joint_f1_gap": reader["deltas_vs_context_order"]["joint_f1"]["mean_delta"],
        "reader_answer_f1_gap": reader["deltas_vs_context_order"]["answer_f1"]["mean_delta"],
        "reader_sp_f1_gap": reader["deltas_vs_context_order"]["sp_f1"]["mean_delta"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
