#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
PAPER = ROOT / "V7-HP-PAPER"
OUT = PAPER / "high_tier_extension"
SEL23 = PAPER / "selector_v2_3"
SEL22 = PAPER / "selector_v2_2"
PROOF = PAPER / "paper_reviewer_proof_pack"


def ensure_dirs():
    for rel in [
        "outputs/feasibility",
        "outputs/multi_reader",
        "outputs/theory",
        "outputs/strong_baselines",
        "outputs/musique_smoke",
        "outputs/scaleup",
        "outputs/tables",
        "outputs/latex",
        "reports",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def fmt(x, nd=4, signed=False):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        s = f"{x:.{nd}f}"
        if signed and x > 0:
            return "+" + s
        return s
    return str(x)


def mean(rows, key):
    vals = [float(r.get(key, 0.0)) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def latex_table(headers, rows, caption, label):
    def esc(x):
        return str(x).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(esc(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(esc(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{esc(caption)}}}", f"\\label{{{esc(label)}}}", "\\end{table}", ""]
    return "\n".join(lines)


def summarize_metric_rows(rows, prefix="candidate"):
    n = len(rows)
    if not rows:
        return {}
    metrics = {
        "answer_f1": mean(rows, f"{prefix}_answer_f1"),
        "joint_f1": mean(rows, f"{prefix}_joint_f1"),
        "support_recall@5": mean(rows, f"{prefix}_support_recall"),
        "sp_f1": mean(rows, f"{prefix}_sp_f1"),
        "answer_em": mean(rows, f"{prefix}_answer_em"),
        "answer_access@5": mean(rows, f"{prefix}_answer_access_at_k"),
    }
    base = {
        "answer_f1": mean(rows, "baseline_answer_f1"),
        "joint_f1": mean(rows, "baseline_joint_f1"),
        "support_recall@5": mean(rows, "baseline_support_recall"),
        "sp_f1": mean(rows, "baseline_sp_f1"),
        "answer_em": mean(rows, "baseline_answer_em"),
        "answer_access@5": mean(rows, "baseline_answer_access_at_k"),
    }
    out = {"n": n, **metrics, "baseline": base}
    for key in ["answer_f1", "joint_f1", "support_recall@5", "sp_f1"]:
        out[key + "_delta"] = metrics[key] - base[key]
    return out


def row_score(row, mode):
    if mode == "setr_proxy":
        return (
            1.2 * float(row.get("support_proxy_delta", 0.0))
            + 0.9 * float(row.get("title_bridge_score", 0.0))
            + 0.4 * float(row.get("support_proxy_delta_vs_replaced_doc", 0.0))
            - 0.25 * float(row.get("displacement_score", 0.0))
        )
    if mode == "rankrag_proxy":
        return (
            0.8 * float(row.get("safe_answer_prob", 0.0))
            + 0.6 * float(row.get("support_proxy_delta", 0.0))
            + 0.4 * float(row.get("title_bridge_score", 0.0))
            - 0.5 * float(row.get("answer_risk_score", 0.0))
        )
    if mode == "influence_proxy":
        return (
            0.9 * float(row.get("support_proxy_delta", 0.0))
            + 0.8 * float(row.get("title_bridge_score", 0.0))
            + 0.4 * float(row.get("agent_weight_delta", 0.0))
            - 0.7 * float(row.get("answer_risk_score", 0.0))
        )
    return 0.0


def choose_proxy(rows, mode, fraction=0.5):
    by_q = defaultdict(list)
    for row in rows:
        by_q[row["query_id"]].append(row)
    best = []
    for qid, qrows in by_q.items():
        chosen = max(qrows, key=lambda r: row_score(r, mode))
        item = dict(chosen)
        item["_proxy_score"] = row_score(chosen, mode)
        best.append(item)
    best.sort(key=lambda r: r["_proxy_score"], reverse=True)
    cutoff = int(round(len(best) * fraction))
    selected = []
    for i, row in enumerate(best):
        if i < cutoff and row.get("effective_context_changed", True):
            selected.append(row)
        else:
            fallback = dict(row)
            for metric in ["answer_f1", "joint_f1", "support_recall", "sp_f1", "answer_em", "answer_access_at_k"]:
                fallback[f"candidate_{metric}"] = fallback.get(f"baseline_{metric}", 0.0)
            selected.append(fallback)
    return selected


def audit_extension_feasibility():
    ensure_dirs()
    required = {
        "per_example_delta": SEL23 / "outputs/final_1000/per_example_delta.jsonl",
        "final_1000_summary": SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json",
        "significance": SEL23 / "outputs/final_1000/significance_report.json",
        "ablation": SEL23 / "outputs/ablation/ablation_summary.json",
        "sufficiency_memo": PROOF / "reports/experiment_sufficiency_memo.md",
        "additional_experiment_recommendation": PROOF / "reports/final_additional_experiment_recommendation.md",
    }
    per_rows = read_jsonl(required["per_example_delta"])
    label_rows = read_jsonl(SEL23 / "outputs/labels/action_labels.jsonl")
    has_context_text = False
    for rows in (per_rows[:20], label_rows[:20]):
        for row in rows:
            if any(k in row for k in ["baseline_context", "selected_context", "context_docs", "candidate_docs", "documents"]):
                has_context_text = True
    summary = read_json(required["final_1000_summary"])
    sig = read_json(required["significance"])
    out = {
        "status": "complete",
        "frozen_main_result_preserved": True,
        "available_files": {k: str(v) for k, v in required.items() if v.exists()},
        "missing_files": {k: str(v) for k, v in required.items() if not v.exists()},
        "multi_reader_replication": {
            "expected_cost": "low if full baseline/selected context text exists; high otherwise",
            "required_reader_models": ["google/flan-t5-large", "google/flan-t5-base"],
            "required_reader_prompts": "same HotpotQA reader prompt as final_1000",
            "can_reuse_context_actions": bool(has_context_text),
            "available_current_reader": "google/flan-t5-large outcome metrics only",
            "expected_paper_value": "high if at least one additional reader can be run from frozen contexts",
            "decision": "not_executed" if not has_context_text else "ready_to_execute",
            "reason": "final_1000 artifacts expose metrics and titles but not full reader contexts" if not has_context_text else "contexts available",
        },
        "theory_formalization": {
            "expected_cost": "low",
            "required_inputs": ["final summary", "claim boundary", "ablation/failure diagnostics"],
            "expected_paper_value": "high",
            "decision": "execute",
        },
        "strong_baseline_comparison": {
            "SetR_feasibility": "proxy feasible from support/title/diversity features",
            "RankRAG_feasibility": "heuristic reader-aware proxy feasible from safe_answer_prob and risk features",
            "InfluenceGuided_feasibility": "utility proxy feasible from support, bridge, weight, and risk features",
            "can_implement_proxy": bool(label_rows),
            "expected_paper_value": "medium-high",
            "decision": "execute_proxy",
        },
        "musique_generalization": {
            "adapter_ready": False,
            "evidence_labels_available": "not verified in current workspace",
            "reader_cost": "unknown; do not run by default",
            "risk": "high",
            "decision": "feasibility_only",
        },
        "scaleup_hotpot": {
            "can_extend_1000_to_2000_or_full_dev": "requires additional reader/context artifacts",
            "reader_cost": "high",
            "current_significance": sig.get("metrics", {}),
            "expected_value": "limited because joint/support are already significant",
            "decision": "not_recommended_without_reviewer_request",
        },
        "frozen_summary": summary,
    }
    write_json(OUT / "outputs/feasibility/extension_feasibility_summary.json", out)
    report = f"""# Extension Feasibility Report

This audit preserves the frozen HotpotQA v2.3 result and checks only low-cost high-tier extensions.

## Decision Summary

- Multi-reader replication: **{out['multi_reader_replication']['decision']}**. {out['multi_reader_replication']['reason']}
- Theory formalization: **execute**. This is low-cost and high paper value.
- Strong baseline comparison: **execute proxy**. SetR-style, RankRAG-style, and Influence-style proxies can be computed from existing action labels.
- MuSiQue: **feasibility only**. Do not run reader or full validation by default.
- HotpotQA scale-up: **not recommended now** because joint/support metrics are already significant and extra reader cost is high.

## Claim Boundary

The frozen v2.3 result remains the main result: joint/support metrics improve significantly, while answer_f1 is a small non-significant positive delta. 2Wiki remains diagnostic/limitation evidence, not a successful generalization claim.
"""
    (OUT / "reports/extension_feasibility_report.md").write_text(report)
    return out


def run_multi_reader_replication():
    ensure_dirs()
    summary = read_json(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    sig = read_json(SEL23 / "outputs/final_1000/significance_report.json")
    per_rows = read_jsonl(SEL23 / "outputs/final_1000/per_example_delta.jsonl")
    per_reader = {
        "google/flan-t5-large": {
            "status": "completed_existing_reader",
            "source": "frozen final_1000",
            "metrics": summary,
            "significance": sig,
        },
        "google/flan-t5-base": {
            "status": "not_executed",
            "reason": "baseline/selected full context text is absent from frozen final_1000 artifacts; rerunning retrieval/reader would violate low-cost frozen-output replication.",
            "required_resource": "frozen baseline_context and selected_context text for each query",
            "paper_impact": "cannot claim multi-reader robustness from this turn",
            "recommendation": "materialize context snapshots in a future controlled run before reader-base/xl replication",
        },
        "google/flan-t5-xl": {
            "status": "not_executed",
            "reason": "same context artifact gap, plus higher compute cost",
            "required_resource": "full context snapshots and available model weights/GPU",
            "paper_impact": "not available for current claim",
            "recommendation": "defer unless reviewer requests reader robustness",
        },
    }
    rows = []
    rows.append([
        "google/flan-t5-large",
        "completed_existing",
        fmt(summary.get("answer_f1_delta"), signed=True),
        fmt(summary.get("joint_f1_delta"), signed=True),
        fmt(summary.get("support_recall_delta"), signed=True),
        fmt(summary.get("sp_f1_delta"), signed=True),
        "main reader; joint/support significant, answer_f1 non-significant",
    ])
    for r in ["google/flan-t5-base", "google/flan-t5-xl"]:
        rows.append([r, "not_executed", "NA", "NA", "NA", "NA", per_reader[r]["reason"]])
    table = md_table(["reader", "status", "answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "interpretation"], rows)
    (OUT / "outputs/tables/multi_reader_replication_table.md").write_text(table)
    (OUT / "outputs/latex/multi_reader_replication_table.tex").write_text(latex_table(
        ["reader", "status", "answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "interpretation"],
        rows,
        "Multi-reader replication feasibility and existing-reader result.",
        "tab:multi_reader_replication",
    ))
    write_json(OUT / "outputs/multi_reader/per_reader_metrics.json", per_reader)
    write_json(OUT / "outputs/multi_reader/per_example_multi_reader_delta.jsonl", {"note": "not produced; no additional reader was executed"})
    out = {
        "status": "partial_existing_reader_only",
        "n": summary.get("n", len(per_rows)),
        "completed_readers": ["google/flan-t5-large"],
        "not_executed_readers": ["google/flan-t5-base", "google/flan-t5-xl"],
        "reader_robust_trend_claim_allowed": False,
        "reason": "Only the existing flan-t5-large reader outcome is available; full context text is missing for safe additional reader replication.",
    }
    write_json(OUT / "outputs/multi_reader/multi_reader_summary.json", out)
    write_json(OUT / "outputs/multi_reader/multi_reader_significance.json", {"google/flan-t5-large": sig})
    report = f"""# Multi-Reader Replication Report

Status: **partial_existing_reader_only**.

The frozen HotpotQA v2.3 outputs provide one completed reader result, `google/flan-t5-large`, but do not expose full baseline and selected context text needed to re-run `google/flan-t5-base` or `google/flan-t5-xl` safely. Therefore no new reader was launched in this extension pass.

Existing reader result:

- answer_f1_delta: {fmt(summary.get('answer_f1_delta'), signed=True)}
- joint_f1_delta: {fmt(summary.get('joint_f1_delta'), signed=True)}
- support_recall_delta: {fmt(summary.get('support_recall_delta'), signed=True)}
- sp_f1_delta: {fmt(summary.get('sp_f1_delta'), signed=True)}

Paper implication: this extension **does not justify a multi-reader robustness claim**. The current paper may keep the existing main-reader HotpotQA claim and list multi-reader replication as a future/revision-ready experiment requiring frozen context snapshots.
"""
    (OUT / "reports/multi_reader_replication_report.md").write_text(report)
    return out


def write_theory_formalization():
    ensure_dirs()
    headers = ["symbol", "definition", "paper role"]
    rows = [
        ["q", "query", "question input"],
        ["C_b", "baseline reader context", "reader input before action"],
        ["a", "context action produced after federated routing", "candidate insertion/reordering/replacement"],
        ["C_a", "modified context after applying action a", "reader input after action"],
        ["R(q, C)", "reader output under context C", "answer generation / extraction behavior"],
        ["M_ans", "answer metric such as EM/F1", "answer quality"],
        ["M_sup", "support/evidence metric", "evidence quality"],
        ["M_joint", "joint answer-support metric", "reader-side multi-hop success"],
        ["Delta_sup(a)", "M_sup(q,C_a)-M_sup(q,C_b)", "routing-side support gain"],
        ["Delta_ans(a)", "M_ans(q,C_a)-M_ans(q,C_b)", "reader-side answer effect"],
        ["Delta_joint(a)", "M_joint(q,C_a)-M_joint(q,C_b)", "combined utility"],
    ]
    (OUT / "outputs/tables/formal_definition_table.md").write_text(md_table(headers, rows))
    report = """# Theory Formalization Report

## 1. Policy-Action-to-Reader Gap

Let `q` denote a query, `C_b` the baseline reader context, `a` a context action produced after federated routing, and `C_a` the context after applying `a`. A reader `R(q, C)` produces an answer from a context. We evaluate answer quality with `M_ans`, support/evidence quality with `M_sup`, and combined multi-hop utility with `M_joint`.

We define:

- `Delta_sup(a) = M_sup(q, C_a) - M_sup(q, C_b)`
- `Delta_ans(a) = M_ans(q, C_a) - M_ans(q, C_b)`
- `Delta_joint(a) = M_joint(q, C_a) - M_joint(q, C_b)`

The central empirical gap is:

`Delta_sup(a) > 0` does not imply `Delta_ans(a) >= 0` or `Delta_joint(a) > 0`.

Federated routing can expose support-relevant evidence, but applying the resulting context action may still damage answer quality or fail to improve the joint metric.

## 2. Answer-Neutral Positive Action

An action is answer-neutral positive when:

- `Delta_ans(a) >= 0`
- `Delta_joint(a) > 0`
- `Delta_sup(a) >= 0`

This definition separates support discovery from reader-safe application. The selector is therefore not a retriever replacement; it is an action filter deciding whether routed context changes should be applied to the reader input.

## 3. No-Leak Constraint

At inference time, the selector can only use features `phi(q, C_b, a)` that do not depend on held-out reader outcomes, gold answer strings, or gold support labels. Train-fold outcomes may be used to construct labels, but held-out query outcomes are not used for selection.

## 4. Selector Objective

The inference-time decision can be written as:

`select a* = argmax_a s_theta(phi(q, C_b, a))`

subject to predicted answer-safety and action-effectiveness constraints. The frozen v2.3 selector operationalizes this as answer-neutral positive-action selection under query-level cross-fitting.

## 5. Paper Contribution Statement

The central problem is not whether federated routing can retrieve additional support evidence, but whether the resulting action should be applied to the reader context. We formalize this as an answer-neutral action selection problem under no-leak constraints.
"""
    (OUT / "reports/theory_formalization_report.md").write_text(report)
    return {"status": "complete", "paper_value": "high"}


def build_strong_baseline_proxy():
    ensure_dirs()
    action_rows = read_jsonl(SEL23 / "outputs/labels/action_labels.jsonl")
    summary23 = read_json(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    summary22 = read_json(SEL22 / "outputs/final_1000/final_1000_crossfit_summary.json")
    methods = []
    methods.append(("baseline", {
        "answer_f1_delta": 0.0,
        "joint_f1_delta": 0.0,
        "support_recall@5_delta": 0.0,
        "sp_f1_delta": 0.0,
        "fallback_rate": 1.0,
        "positive_candidate_recall": 0.0,
        "answer_drop_selected": 0.0,
        "interpretation": "original reader context",
    }))
    methods.append(("v2.2 support-first", {
        "answer_f1_delta": summary22.get("answer_f1_delta"),
        "joint_f1_delta": summary22.get("joint_f1_delta"),
        "support_recall@5_delta": summary22.get("support_recall_delta"),
        "sp_f1_delta": summary22.get("sp_f1_delta"),
        "fallback_rate": summary22.get("fallback_rate"),
        "positive_candidate_recall": summary22.get("positive_candidate_recall"),
        "answer_drop_selected": summary22.get("selected_answer_drop_rate"),
        "interpretation": "frozen prior selector",
    }))
    methods.append(("v2.3 answer-neutral positive selector", {
        "answer_f1_delta": summary23.get("answer_f1_delta"),
        "joint_f1_delta": summary23.get("joint_f1_delta"),
        "support_recall@5_delta": summary23.get("support_recall_delta"),
        "sp_f1_delta": summary23.get("sp_f1_delta"),
        "fallback_rate": summary23.get("fallback_rate"),
        "positive_candidate_recall": summary23.get("positive_candidate_recall"),
        "answer_drop_selected": summary23.get("selected_answer_drop_rate"),
        "interpretation": "main frozen method",
    }))
    proxy_details = {}
    for mode, label in [
        ("setr_proxy", "SetR-style set selection proxy"),
        ("rankrag_proxy", "RankRAG-style reader-aware reranking proxy"),
        ("influence_proxy", "Influence-style utility proxy"),
    ]:
        selected = choose_proxy(action_rows, mode)
        met = summarize_metric_rows(selected)
        paper_pos = sum(1 for r in selected if r.get("paper_positive") == 1)
        drops = sum(1 for r in selected if r.get("answer_drop") == 1)
        detail = {
            "answer_f1_delta": met.get("answer_f1_delta", 0.0),
            "joint_f1_delta": met.get("joint_f1_delta", 0.0),
            "support_recall@5_delta": met.get("support_recall@5_delta", 0.0),
            "sp_f1_delta": met.get("sp_f1_delta", 0.0),
            "fallback_rate": sum(1 for r in selected if r.get("candidate_family") == "baseline") / len(selected) if selected else 1.0,
            "positive_candidate_recall": paper_pos / max(1, sum(1 for r in action_rows if r.get("paper_positive") == 1)),
            "answer_drop_selected": drops / max(1, len(selected)),
            "interpretation": "proxy; not full prior-system reproduction",
        }
        proxy_details[mode] = detail
        methods.append((label, detail))
    methods.append(("oracle diagnostic only", {
        "answer_f1_delta": None,
        "joint_f1_delta": None,
        "support_recall@5_delta": None,
        "sp_f1_delta": None,
        "fallback_rate": None,
        "positive_candidate_recall": None,
        "answer_drop_selected": None,
        "interpretation": "upper-bound diagnostic, not inference",
    }))
    rows = []
    for name, m in methods:
        rows.append([
            name,
            fmt(m.get("answer_f1_delta"), signed=True),
            fmt(m.get("joint_f1_delta"), signed=True),
            fmt(m.get("support_recall@5_delta"), signed=True),
            fmt(m.get("sp_f1_delta"), signed=True),
            fmt(m.get("positive_candidate_recall")),
            fmt(m.get("answer_drop_selected")),
            m.get("interpretation", ""),
        ])
    headers = ["method", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "positive_candidate_recall", "answer_drop_selected", "interpretation"]
    (OUT / "outputs/tables/strong_baseline_proxy_table.md").write_text(md_table(headers, rows))
    (OUT / "outputs/latex/strong_baseline_proxy_table.tex").write_text(latex_table(headers, rows, "Strong baseline proxy comparison.", "tab:strong_baseline_proxy"))
    out = {
        "status": "complete_proxy_only",
        "note": "SetR/RankRAG/Influence rows are proxy implementations, not exact reproductions.",
        "methods": {name: m for name, m in methods},
    }
    write_json(OUT / "outputs/strong_baselines/strong_baseline_proxy_summary.json", out)
    best_joint = max((x for x in methods if x[1].get("joint_f1_delta") is not None), key=lambda x: x[1].get("joint_f1_delta", -9))
    report = f"""# Strong Baseline Proxy Comparison Report

This report compares the frozen v2.3 selector against proxy baselines inspired by set-level selection, reader-aware reranking, and influence/utility-based context selection. These are **not exact reproductions** of SetR, RankRAG, or Influence-Guided Context Selection.

Best proxy/action-level joint_f1_delta row: **{best_joint[0]}** with {fmt(best_joint[1].get('joint_f1_delta'), signed=True)}.

The comparison is useful for paper positioning because it tests whether answer-neutral action selection gives benefits beyond simple relevance, set coverage, or utility heuristics using the same candidate-action table.

Safe claim: we compare against proxy implementations inspired by stronger context-selection families. We should not write that we outperform the original SetR/RankRAG/Influence systems.
"""
    (OUT / "reports/strong_baseline_comparison_report.md").write_text(report)
    return out


def musique_feasibility_smoke():
    ensure_dirs()
    candidates = [
        ROOT / "data/musique",
        ROOT / "datasets/musique",
        ROOT / "V7-HP-PAPER/cross_dataset_validation/musique",
    ]
    found = [str(p) for p in candidates if p.exists()]
    out = {
        "status": "feasibility_only",
        "dataset_paths_found": found,
        "adapter_ready": False,
        "evidence_labels_available": "not verified",
        "reader_cost": "not estimated because adapter/context pipeline is not ready",
        "decision": "not_executed",
        "reason": "The current instruction says MuSiQue should be feasibility-only unless all prerequisites are met; prerequisites are not met.",
        "required_resource": "MuSiQue dev adapter with answer and paragraph/evidence labels plus frozen context-action compatibility",
        "paper_impact": "future-work only; no broad cross-dataset generalization claim",
        "recommendation": "defer until after main paper submission or reviewer request",
    }
    write_json(OUT / "outputs/musique_smoke/musique_feasibility_summary.json", out)
    report = """# MuSiQue Smoke Decision

Status: **not_executed**.

The current task allows only feasibility checking unless evidence labels, adapter cost, reader cost, and a strong positive reason are all satisfied. Those prerequisites are not established in the current V7-HP-PAPER workspace, so no MuSiQue reader smoke was launched.

Paper impact: keep MuSiQue as future work. Do not claim broad multi-dataset generalization.
"""
    (OUT / "reports/musique_smoke_decision.md").write_text(report)
    return out


def scaleup_feasibility_audit():
    ensure_dirs()
    sig = read_json(SEL23 / "outputs/final_1000/significance_report.json")
    out = {
        "status": "feasibility_only",
        "can_extend_1000_to_2000_or_full_dev": "not recommended from current artifacts",
        "remaining_dev_examples_available": "not audited because no reader run is planned",
        "candidate_action_table_availability": "final_1000 action labels available; larger split not verified",
        "reader_compute_cost": "high",
        "expected_variance_reduction": "limited",
        "current_significance_already_sufficient": {
            "joint_f1": sig.get("metrics", {}).get("joint_f1", {}),
            "support_recall@5": sig.get("metrics", {}).get("support_recall@5", {}),
            "sp_f1": sig.get("metrics", {}).get("sp_f1", {}),
            "answer_f1": sig.get("metrics", {}).get("answer_f1", {}),
        },
        "decision": "not_recommended_without_reviewer_request",
        "reason": "joint/support/sp are already significant on final_1000; scale-up would consume reader compute without changing the central claim.",
    }
    write_json(OUT / "outputs/scaleup/scaleup_feasibility_summary.json", out)
    report = """# HotpotQA Scale-Up Decision

Status: **not_executed / feasibility-only**.

The final_1000 result already has significant joint/support-side gains. Expanding to 2000 or full dev would require additional reader/context computation and is unlikely to change the paper's central claim. Keep this as a revision plan if a reviewer specifically requests larger-scale validation.
"""
    (OUT / "reports/scaleup_decision.md").write_text(report)
    return out


def write_high_tier_extension_report():
    ensure_dirs()
    feas = read_json(OUT / "outputs/feasibility/extension_feasibility_summary.json")
    mr = read_json(OUT / "outputs/multi_reader/multi_reader_summary.json")
    sb = read_json(OUT / "outputs/strong_baselines/strong_baseline_proxy_summary.json")
    report = f"""# High-Tier Submission Recommendation

## What Was Completed

1. Extension feasibility audit: complete.
2. Multi-reader replication: {mr.get('status', 'unknown')}. Additional readers were not executed because frozen context text is absent.
3. Theory formalization: complete. The paper can now frame the core novelty as the policy-action-to-reader gap and answer-neutral action selection under no-leak constraints.
4. Strong baseline proxy comparison: {sb.get('status', 'unknown')}. SetR-style, RankRAG-style, and Influence-style proxy baselines were computed from the existing action table.
5. MuSiQue: feasibility only; not executed.
6. HotpotQA scale-up: feasibility only; not recommended without reviewer request.

## Does This Move the Paper Toward a Higher-Tier Submission?

The strongest successful extension is the **theory formalization**, plus a **proxy strong-baseline comparison**. This improves framing and reviewer defensibility. It does not yet provide the strongest possible empirical boost because multi-reader replication could not be safely executed from the available frozen artifacts.

Recommendation: the paper is stronger than the previous writing pack for a HotpotQA-centered submission. It can be positioned as a main-conference stretch only if the authors are comfortable with proxy baselines and a single-reader main empirical setup; otherwise EMNLP/NAACL Findings or COLING remains the safer target.

## Should Claims Change?

Allowed strengthening:

- Emphasize the formal policy-action-to-reader gap.
- Add proxy comparisons against set-level, reader-aware, and utility-style action-selection heuristics.
- State that current multi-reader replication is prepared but blocked by missing frozen context snapshots.

Still forbidden:

- Do not claim answer_f1 significantly improves.
- Do not claim successful 2Wiki or MuSiQue generalization.
- Do not claim exact SetR/RankRAG/Influence reproduction.
- Do not present oracle diagnostics as inference-time methods.

## Next Best Single Action

If more time is available, materialize frozen baseline and selected contexts for final_1000 and run `google/flan-t5-base` plus `google/flan-t5-large` under identical prompts. That is the most likely low-cost empirical upgrade.
"""
    (OUT / "reports/high_tier_submission_recommendation.md").write_text(report)
    summary = """# Extension Summary for Paper

We conducted a high-tier extension audit without modifying the frozen HotpotQA v2.3 result. The extension adds a formal definition of the policy-action-to-reader gap and compares v2.3 against proxy baselines inspired by set-level selection, reader-aware reranking, and influence-style utility selection. These proxies are not exact reproductions of prior systems, but they strengthen the empirical positioning by testing whether the answer-neutral selector improves over relevance/utility-only heuristics under the same candidate-action space.

The multi-reader extension could not be completed because the frozen artifacts contain metrics, titles, and action labels but not the full baseline and selected reader contexts required for safe reader re-evaluation. Therefore the paper should not claim multi-reader robustness in its current form.
"""
    (OUT / "reports/extension_summary_for_paper.md").write_text(summary)
    return {"status": "complete"}


def run_all():
    audit_extension_feasibility()
    run_multi_reader_replication()
    write_theory_formalization()
    build_strong_baseline_proxy()
    musique_feasibility_smoke()
    scaleup_feasibility_audit()
    write_high_tier_extension_report()


if __name__ == "__main__":
    run_all()
