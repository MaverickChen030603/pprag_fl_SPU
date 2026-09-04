#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CDV = ROOT / "cross_dataset_validation"


DATASETS = {
    "2WikiMultiHopQA": {
        "patterns": ["*2wiki*", "*2wikimultihop*"],
        "priority": "P1",
        "requires_new_retrieval_index": True,
        "requires_new_metric": True,
        "estimated_reader_cost": "medium-high for 300, high for 1000 with flan-t5-large",
        "recommended_priority": "proceed_first_if_available",
    },
    "MuSiQue": {
        "patterns": ["*musique*"],
        "priority": "P2",
        "requires_new_retrieval_index": True,
        "requires_new_metric": True,
        "estimated_reader_cost": "medium for smoke_300; do not run 1000 until smoke passes",
        "recommended_priority": "after_2wiki",
    },
    "IIRC": {
        "patterns": ["*iirc*"],
        "priority": "P3_optional",
        "requires_new_retrieval_index": True,
        "requires_new_metric": True,
        "estimated_reader_cost": "unknown; adapter likely non-trivial",
        "recommended_priority": "optional",
    },
    "MultiHop-RAG": {
        "patterns": ["*multihoprag*", "*multi-hop-rag*", "*multihop_rag*"],
        "priority": "P3_optional",
        "requires_new_retrieval_index": True,
        "requires_new_metric": True,
        "estimated_reader_cost": "unknown; domain transfer likely requires new corpus handling",
        "recommended_priority": "optional",
    },
}


def ensure_dirs():
    for rel in [
        "outputs/2wiki_adapter",
        "outputs/2wiki_smoke_300",
        "outputs/2wiki_final_1000",
        "outputs/musique_smoke_300",
        "outputs/diagnostics",
        "outputs/tables",
        "reports",
    ]:
        (CDV / rel).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def local_find(patterns):
    roots = [
        ROOT,
        ROOT / "data",
        Path.home() / ".cache/huggingface/datasets",
        Path.home() / ".cache/huggingface/hub",
    ]
    hits = []
    for r in roots:
        if not r.exists():
            continue
        for pat in patterns:
            try:
                hits.extend(str(p) for p in r.rglob(pat) if len(hits) < 50)
            except Exception:
                pass
    return sorted(set(hits))[:50]


def inspect_json_candidate(path):
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl"}:
        return {}
    try:
        if p.suffix.lower() == ".jsonl":
            line = p.open().readline()
            ex = json.loads(line) if line else {}
            n = sum(1 for _ in p.open())
        else:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                n = len(data)
                ex = data[0] if data else {}
            elif isinstance(data, dict):
                for key in ["data", "examples", "validation", "dev", "train"]:
                    if isinstance(data.get(key), list):
                        n = len(data[key])
                        ex = data[key][0] if data[key] else {}
                        break
                else:
                    n, ex = 1, data
            else:
                n, ex = 0, {}
        keys = set(ex) if isinstance(ex, dict) else set()
        return {
            "num_examples": n,
            "has_answer": bool(keys & {"answer", "answers", "gold_answer"}),
            "has_supporting_facts_or_evidence": bool(keys & {"supporting_facts", "supporting_titles", "evidence", "evidences", "supporting_paragraphs", "reasoning_path"}),
            "has_context_docs": bool(keys & {"context", "paragraphs", "contexts", "documents"}),
            "sample_keys": sorted(keys)[:40],
        }
    except Exception as exc:
        return {"inspect_error": str(exc)}


def audit_candidate_datasets():
    ensure_dirs()
    summary = {}
    for name, spec in DATASETS.items():
        hits = local_find(spec["patterns"])
        usable = {}
        for h in hits:
            info = inspect_json_candidate(h)
            if info.get("num_examples"):
                usable[h] = info
        best_path = next(iter(usable), None)
        best = usable.get(best_path, {})
        available = bool(best_path)
        summary[name] = {
            "dataset_available": available,
            "download_or_local_path": best_path or "not_found_locally",
            "num_examples": best.get("num_examples", 0),
            "has_answer": bool(best.get("has_answer", False)),
            "has_supporting_facts_or_evidence": bool(best.get("has_supporting_facts_or_evidence", False)),
            "has_context_docs": bool(best.get("has_context_docs", False)),
            "has_distractors": "manual_check_required" if available else False,
            "has_corpus": "manual_check_required" if available else False,
            "hotpot_format_compatibility": "high_if_context_and_support_titles_present" if available else "not_assessable_without_data",
            "requires_new_retrieval_index": spec["requires_new_retrieval_index"],
            "requires_new_metric": spec["requires_new_metric"],
            "estimated_reader_cost": spec["estimated_reader_cost"],
            "recommended_priority": spec["recommended_priority"],
            "priority": spec["priority"],
            "candidate_paths_considered": hits[:20],
            "usable_json_candidates": usable,
        }
    write_json(CDV / "outputs/dataset_feasibility_summary.json", summary)
    write_dataset_selection_memo(summary)
    return summary


def write_dataset_selection_memo(summary):
    two = summary.get("2WikiMultiHopQA", {})
    musique = summary.get("MuSiQue", {})
    lines = ["# Dataset Selection Memo\n"]
    lines.append("## Executive Decision\n")
    if two.get("dataset_available") and two.get("has_answer") and two.get("has_supporting_facts_or_evidence"):
        lines.append("2WikiMultiHopQA is available with usable labels. Proceed to adapter and smoke test first.\n")
    else:
        lines.append("2WikiMultiHopQA is not available locally with usable evidence labels. Do not run smoke/final evaluation yet; prepare or download the dataset first.\n")
    lines.append("MuSiQue should remain second priority and should only run after 2Wiki feasibility is resolved.\n")
    lines.append("IIRC and MultiHop-RAG remain optional appendix/limitation candidates.\n")
    lines.append("## Feasibility Table\n")
    lines.append("| dataset | available | path | examples | answer | evidence | context | recommendation |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for name, s in summary.items():
        lines.append(f"| {name} | {s['dataset_available']} | {s['download_or_local_path']} | {s['num_examples']} | {s['has_answer']} | {s['has_supporting_facts_or_evidence']} | {s['has_context_docs']} | {s['recommended_priority']} |")
    lines.append("\n## Paper Impact\n")
    lines.append("The HotpotQA v2.3 main result remains frozen. Cross-dataset validation is blocked until P1 data is available, so current paper claims should remain HotpotQA-centered with cross-dataset validation listed as pending robustness work.\n")
    (CDV / "reports/dataset_selection_memo.md").write_text("\n".join(lines))


def write_skipped_2wiki_outputs():
    ensure_dirs()
    feasibility = json.loads((CDV / "outputs/dataset_feasibility_summary.json").read_text()) if (CDV / "outputs/dataset_feasibility_summary.json").exists() else audit_candidate_datasets()
    two = feasibility["2WikiMultiHopQA"]
    converted = []
    adapter_summary = {
        "status": "skipped_dataset_unavailable" if not two["dataset_available"] else "adapter_not_run",
        "num_examples": 0,
        "num_with_answer": 0,
        "num_with_supporting_titles": 0,
        "avg_context_docs": 0,
        "avg_support_docs": 0,
        "reasoning_type_distribution": {},
        "unsupported_examples_removed": 0,
        "reason": "2WikiMultiHopQA local file with answer/evidence/context was not found; no smoke test is valid without data.",
    }
    write_json(CDV / "outputs/2wiki_adapter/2wiki_converted.json", converted)
    write_json(CDV / "outputs/2wiki_adapter/adapter_summary.json", adapter_summary)
    summary = {
        "status": "not_run_dataset_unavailable",
        "n": 0,
        "methods": [],
        "gate_pass": False,
        "reason": "2WikiMultiHopQA is unavailable locally; per execution rules, smoke test is blocked until adapter data exists.",
    }
    write_json(CDV / "outputs/2wiki_smoke_300/summary.json", summary)
    write_json(CDV / "outputs/2wiki_smoke_300/significance_report.json", {"status": "not_run_dataset_unavailable", "metrics": {}})
    write_json(CDV / "outputs/2wiki_smoke_300/failure_summary.json", {"status": "not_run_dataset_unavailable", "blocking_issue": "missing_2wiki_dataset"})
    (CDV / "outputs/2wiki_smoke_300/per_example_delta.jsonl").write_text("")
    return summary


def make_cross_dataset_report():
    ensure_dirs()
    summary = json.loads((CDV / "outputs/dataset_feasibility_summary.json").read_text()) if (CDV / "outputs/dataset_feasibility_summary.json").exists() else audit_candidate_datasets()
    two_smoke = json.loads((CDV / "outputs/2wiki_smoke_300/summary.json").read_text()) if (CDV / "outputs/2wiki_smoke_300/summary.json").exists() else write_skipped_2wiki_outputs()
    rows = [
        {"dataset": "HotpotQA", "n": 1000, "method": "selector_v2.3", "answer_f1_delta": 0.0023, "joint_or_evidence_delta": 0.0150, "support/evidence_recall_delta": 0.0190, "fallback_rate": 0.5, "positive_candidate_recall": 0.3288, "gate_pass": True, "paper_role": "main_result"},
        {"dataset": "2WikiMultiHopQA", "n": two_smoke.get("n", 0), "method": "not_run", "answer_f1_delta": "", "joint_or_evidence_delta": "", "support/evidence_recall_delta": "", "fallback_rate": "", "positive_candidate_recall": "", "gate_pass": False, "paper_role": "pending_external_validation"},
        {"dataset": "MuSiQue", "n": 0, "method": "not_run", "answer_f1_delta": "", "joint_or_evidence_delta": "", "support/evidence_recall_delta": "", "fallback_rate": "", "positive_candidate_recall": "", "gate_pass": False, "paper_role": "pending_stress_test"},
    ]
    table = md_table(rows, ["dataset", "n", "method", "answer_f1_delta", "joint_or_evidence_delta", "support/evidence_recall_delta", "fallback_rate", "positive_candidate_recall", "gate_pass", "paper_role"])
    (CDV / "outputs/tables/cross_dataset_main_table.md").write_text(table)
    (CDV / "outputs/tables/cross_dataset_ablation_table.md").write_text("| dataset | status |\n|---|---|\n| 2Wiki | not run: dataset unavailable |\n")
    (CDV / "outputs/tables/dataset_comparison_table.md").write_text(table)
    report = f"""# Cross-Dataset Validation Report

## 1. Executive Summary

HotpotQA `selector_v2.3` remains frozen as the paper main result. Cross-dataset validation was initiated, but the first required step, dataset feasibility audit, found no local usable 2WikiMultiHopQA/MuSiQue/IIRC/MultiHop-RAG data with answer/evidence/context fields. Therefore 2Wiki smoke/final and MuSiQue smoke were not run, to avoid fabricating external validation results.

## 2. Dataset Selection Rationale

2WikiMultiHopQA remains the highest-priority external validation target because it is Wikipedia-based and multi-hop. However, it must first be prepared locally with answer, context documents, and evidence/support labels.

## 3. 2Wiki Adapter and Smoke Test

Adapter and smoke-test placeholder outputs were created with status `not_run_dataset_unavailable`. No reader inference was run.

## 4. 2Wiki Final 1000 Result

Not executed. Decision rule requires smoke test success or clear positive trend before running final_1000.

## 5. MuSiQue Smoke Test

Not executed. MuSiQue is second priority and should start only after 2Wiki feasibility is resolved.

## 6. Cross-Dataset Comparison

{table}

## 7. Failure Analysis

The blocking failure is dataset availability, not method failure. Current evidence does not support any cross-dataset generalization claim.

## 8. Paper Recommendation

Keep the paper main claim HotpotQA-centered. Phrase cross-dataset validation as planned robustness work unless 2Wiki data is prepared and smoke/final results are run.
"""
    (CDV / "reports/cross_dataset_validation_report.md").write_text(report)
    recommendation = """# Paper Update Recommendation

## Recommendation

Do not update the paper claim to cross-dataset generalization yet.

## Rationale

The feasibility audit did not find local usable 2WikiMultiHopQA or MuSiQue data. Per the experiment rules, smoke/final evaluations should not be run without adapter-ready answer/evidence/context fields.

## Suggested Paper Wording

Current wording should remain:

> We validate the answer-neutral positive-action selector on HotpotQA and leave cross-dataset validation to future robustness experiments.

If 2Wiki is later prepared and succeeds, update to:

> The answer-neutral positive-action selector generalizes beyond HotpotQA to another Wikipedia-based multi-hop QA benchmark.
"""
    (CDV / "reports/paper_update_recommendation.md").write_text(recommendation)


def md_table(rows, fields):
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    return "\n".join(out) + "\n"


def run_all():
    ensure_dirs()
    audit_candidate_datasets()
    write_skipped_2wiki_outputs()
    make_cross_dataset_report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all", choices=["audit", "2wiki_skipped", "report", "all"])
    args = ap.parse_args()
    if args.task == "audit":
        audit_candidate_datasets()
    elif args.task == "2wiki_skipped":
        write_skipped_2wiki_outputs()
    elif args.task == "report":
        make_cross_dataset_report()
    else:
        run_all()


if __name__ == "__main__":
    main()
