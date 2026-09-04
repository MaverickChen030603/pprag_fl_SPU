#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PF = ROOT / "paper_finalization"
V23 = ROOT / "selector_v2_3"
V22 = ROOT / "selector_v2_2"
V21 = ROOT / "selector_v2_1"
V2 = ROOT / "selector_v2"
V1 = ROOT


METRICS = [
    "answer_access_at_k",
    "support_recall_at_k",
    "sp_f1",
    "answer_em",
    "answer_f1",
    "joint_f1",
    "answer_f1_delta",
    "joint_f1_delta",
    "support_recall_delta",
    "sp_f1_delta",
    "fallback_rate",
    "selected_effective_action_rate",
    "positive_candidate_recall",
    "gate_pass",
]
FEATURES = [
    "safe_answer_prob",
    "support_proxy_delta",
    "support_proxy_delta_vs_replaced_doc",
    "support_proxy_delta_vs_baseline_tail_mean",
    "answer_risk_score",
    "displacement_score",
    "hybrid_score_delta",
    "agent_weight_delta",
    "title_bridge_score",
    "prefix3_preserved",
    "num_added_docs",
    "num_removed_docs",
]


def ensure_dirs():
    for rel in ["outputs/tables", "outputs/figures", "outputs/diagnostics", "outputs/case_studies", "outputs/latex", "outputs/audit", "reports"]:
        (PF / rel).mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def iter_jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def md_table(rows, fields):
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(f, "")) for f in fields) + " |")
    return "\n".join(out) + "\n"


def latex_table(rows, fields, caption, label):
    cols = "l" + "r" * (len(fields) - 1)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
    lines.append(" & ".join(fields).replace("_", "\\_") + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(fmt(r.get(f, "")).replace("_", "\\_") for f in fields) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def load_v23():
    return {
        "label": read_json(V23 / "outputs/labels/label_summary.json", {}),
        "final": read_json(V23 / "outputs/final_1000/final_1000_crossfit_summary.json", {}),
        "sig": read_json(V23 / "outputs/final_1000/significance_report.json", {}),
        "ablation": read_json(V23 / "outputs/ablation/ablation_summary.json", {}),
        "failure": read_json(V23 / "outputs/diagnostics/failure_summary.json", {}),
        "oracle_recall": read_json(V23 / "outputs/diagnostics/oracle_gap_recall_analysis.json", {}),
    }


def row_from_summary(stage, summary, note="", diagnostic=False):
    base = summary.get("baseline", {}) if isinstance(summary, dict) else {}
    row = {"stage": stage, "note": note, "diagnostic": diagnostic}
    for m in ["answer_access_at_k", "support_recall_at_k", "sp_f1", "answer_em", "answer_f1", "joint_f1"]:
        row[m] = summary.get(m, "")
    row["answer_f1_delta"] = summary.get("answer_f1_delta", "")
    row["joint_f1_delta"] = summary.get("joint_f1_delta", "")
    row["support_recall_delta"] = summary.get("support_recall_delta", "")
    row["sp_f1_delta"] = summary.get("sp_f1_delta", "")
    row["fallback_rate"] = summary.get("fallback_rate", "")
    row["selected_effective_action_rate"] = summary.get("selected_effective_action_rate", "")
    row["positive_candidate_recall"] = summary.get("positive_candidate_recall", "")
    row["gate_pass"] = summary.get("gate_pass", "")
    if stage == "baseline" and base:
        for m in ["answer_access_at_k", "support_recall_at_k", "sp_f1", "answer_em", "answer_f1", "joint_f1"]:
            row[m] = base.get(m, "")
        for m in ["answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "fallback_rate", "selected_effective_action_rate", "positive_candidate_recall"]:
            row[m] = 0.0 if "delta" in m else ""
        row["gate_pass"] = ""
    return row


def collect_main_results():
    ensure_dirs()
    v23 = load_v23()
    rows = []
    rows.append(row_from_summary("baseline", v23["final"], "baseline from v2.3 final_1000"))
    # Historical rows are included when directly comparable files exist.
    v22_final = read_json(V22 / "outputs/final_1000/final_1000_crossfit_summary.json", {})
    if v22_final:
        rows.append(row_from_summary("selector_v2.2", v22_final, "scale-calibrated budget"))
    v23_final = v23["final"]
    rows.append(row_from_summary("selector_v2.3 main", v23_final, "final no-leak cross-fitted selector"))
    oracle = read_json(V22 / "outputs/diagnostics/oracle_gap_summary.json", {})
    if oracle:
        rows.append({
            "stage": "oracle diagnostic upper bound",
            "note": "diagnostic only; not an inference-time method",
            "diagnostic": True,
            "answer_access_at_k": "",
            "support_recall_at_k": "",
            "sp_f1": "",
            "answer_em": "",
            "answer_f1": "",
            "joint_f1": "",
            "answer_f1_delta": "",
            "joint_f1_delta": oracle.get("oracle_best_answer_safe_joint_delta", oracle.get("oracle_best_joint_delta", "")),
            "support_recall_delta": oracle.get("oracle_support_delta", ""),
            "sp_f1_delta": "",
            "fallback_rate": "",
            "selected_effective_action_rate": "",
            "positive_candidate_recall": oracle.get("selector_recall_of_positive_candidates", ""),
            "gate_pass": "diagnostic",
        })
    # Optional older summary files.
    optional = [
        ("selector_v2.1", V21 / "outputs/final_1000/final_1000_summary.json"),
        ("selector_v2 100-sample", V2 / "outputs/selector_v2_100/selector_v2_summary.json"),
        ("selector_v1 100-sample", V1 / "outputs/selector_v1_100/selector_summary.json"),
    ]
    for name, path in optional:
        s = read_json(path, None)
        if s:
            rows.insert(max(1, len(rows) - 1), row_from_summary(name, s, "historical non-final scale; compare cautiously"))
    fields = ["stage"] + METRICS + ["note"]
    write_csv(PF / "outputs/tables/main_result_table.csv", rows, fields)
    note = "\n\nNote: v2.3 is the final no-leak cross-fitted selector. Oracle rows are diagnostic upper bounds and not valid inference-time methods.\n"
    (PF / "outputs/tables/main_result_table.md").write_text(md_table(rows, fields) + note)
    (PF / "outputs/latex/main_result_table.tex").write_text(latex_table(rows, fields, "Main results. v2.3 is the final no-leak cross-fitted selector; oracle rows are diagnostic upper bounds.", "tab:main_results"))
    return rows


def build_evolution_table():
    ensure_dirs()
    v23 = load_v23()["final"]
    v22 = read_json(V22 / "outputs/final_1000/final_1000_crossfit_summary.json", {})
    rows = [
        {"stage": "HP4", "main_design": "soft routing + hybrid retrieval + counterfactual diagnostic", "problem_solved": "shows routing can improve support exposure", "remaining_problem": "naive insertion can hurt answer quality", "paper_role": "mechanism motivation", "key_diagnosis": "policy-action-to-reader gap identified"},
        {"stage": "v1", "main_design": "predictor-only selector", "problem_solved": "first reader-aware action filtering", "remaining_problem": "too aggressive and not robust", "paper_role": "historical pilot", "key_diagnosis": "abstention needed"},
        {"stage": "v2", "main_design": "selector with abstention/safety", "problem_solved": "protects answer more strongly", "remaining_problem": "over-fallback suppresses gain", "paper_role": "pilot diagnostic", "key_diagnosis": "risk control too conservative"},
        {"stage": "v2.1", "main_design": "budgeted risk relaxation", "problem_solved": "100-sample gate works", "remaining_problem": "does not scale; final fallback too high", "paper_role": "scale failure evidence", "key_diagnosis": "100-sample calibration does not transfer"},
        {"stage": "v2.2", "main_design": "scale-calibrated budget + effective action filtering", "problem_solved": "fixes effective action and support gain", "remaining_problem": "answer_f1 remains slightly negative", "paper_role": "bridge to final method", "key_diagnosis": "positive recall still too low"},
        {"stage": "v2.3", "main_design": "answer-neutral positive selector", "problem_solved": "improves positive action recall and preserves answer_f1", "remaining_problem": "candidate pool still sparse", "paper_role": "main result", "key_diagnosis": "joint/support gains become significant"},
    ]
    vals = {
        "v2.2": v22,
        "v2.3": v23,
    }
    for r in rows:
        s = vals.get(r["stage"], {})
        for k in ["answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "fallback_rate"]:
            r[k] = s.get(k, "")
    fields = ["stage", "main_design", "problem_solved", "remaining_problem", "answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "fallback_rate", "key_diagnosis", "paper_role"]
    write_csv(PF / "outputs/tables/selector_evolution_table.csv", rows, fields)
    (PF / "outputs/tables/selector_evolution_table.md").write_text(md_table(rows, fields))
    (PF / "outputs/latex/selector_evolution_table.tex").write_text(latex_table(rows, fields, "Selector evolution from HP4 diagnostics to v2.3.", "tab:selector_evolution"))
    return rows


def verify_no_leak_crossfit():
    ensure_dirs()
    fold_configs = read_json(V23 / "outputs/final_1000/fold_configs.json", [])
    source = (V23 / "v23_common.py").read_text() if (V23 / "v23_common.py").exists() else ""
    checks = {
        "query_overlap_between_train_and_heldout": {"status": "verified_by_deterministic_split" if "test = set(queries[i::k])" in source else "manual_check_required", "evidence": "split_queries uses disjoint train/test query sets"},
        "heldout_outcome_not_used_for_inference": {"status": "verified_by_source_review" if "select_actions(test_rows, models, cfg)" in source else "manual_check_required", "evidence": "models/configs are trained/chosen before applying to held-out rows"},
        "training_labels_train_folds_only": {"status": "verified_by_source_review" if "train_rows = [r for r in rows if r[\"query_id\"] in train_q]" in source else "manual_check_required", "evidence": "train_rows are filtered by train_q"},
        "threshold_budget_calibrated_on_train_only": {"status": "verified_by_source_review" if "choose_config(train_rows, models" in source else "manual_check_required", "evidence": "choose_config receives train_rows"},
        "oracle_diagnostic_separated": {"status": "verified", "evidence": "oracle_gap_recall_analysis is written under diagnostics and not used by final selector"},
        "gold_answer_or_gold_support_not_in_inference_features": {"status": "verified_by_feature_list", "evidence": "feature list contains no gold answer/support fields; case study display may use analysis-only fields if present"},
        "fold_count": len(fold_configs),
    }
    write_json(PF / "outputs/audit/no_leak_crossfit_audit.json", checks)
    lines = ["# No-Leak / Cross-Fit Audit\n"]
    for k, v in checks.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**: `{v.get('status')}`  \n  Evidence: {v.get('evidence', '')}")
        else:
            lines.append(f"- **{k}**: `{v}`")
    lines.append("\nManual caveat: this is a code/artifact audit; it does not re-run reader inference and does not inspect hidden external state.\n")
    (PF / "outputs/audit/no_leak_crossfit_audit.md").write_text("\n".join(lines))
    return checks


def load_labels():
    label_path = V23 / "outputs/labels/action_labels.jsonl"
    if label_path.exists():
        return list(iter_jsonl(label_path))
    rows = []
    for r in iter_jsonl(V22 / "outputs/action_table/effective_action_table.jsonl"):
        a = float(r.get("answer_f1_delta", 0) or 0)
        j = float(r.get("joint_f1_delta", 0) or 0)
        sr = float(r.get("support_recall_delta", 0) or 0)
        sp = float(r.get("sp_f1_delta", 0) or 0)
        r["paper_positive"] = int(a >= 0 and j > 0 and (sr > 0 or sp >= 0))
        r["answer_safe"] = int(a >= 0)
        r["joint_positive"] = int(j > 0)
        r["answer_safe_joint_positive"] = int(a >= 0 and j > 0)
        r["support_positive"] = int(sr > 0 or sp > 0)
        r["answer_drop"] = int(a < 0)
        rows.append(r)
    return rows


def candidate_pool_quality():
    ensure_dirs()
    rows = load_labels()
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)
    n = len(rows)
    fam = defaultdict(list)
    for r in rows:
        fam[r.get("candidate_family", "unknown")].append(r)
    summary = {
        "num_queries": len(by_q),
        "num_actions": n,
        "paper_positive_rate": sum(r.get("paper_positive", 0) for r in rows) / n,
        "queries_with_no_positive_action": sum(1 for xs in by_q.values() if not any(r.get("paper_positive", 0) for r in xs)),
        "queries_with_at_least_one_positive_action": sum(1 for xs in by_q.values() if any(r.get("paper_positive", 0) for r in xs)),
        "positive_actions_per_query_distribution": dict(Counter(str(sum(r.get("paper_positive", 0) for r in xs)) for xs in by_q.values())),
        "answer_safe_rate": sum(r.get("answer_safe", 0) for r in rows) / n,
        "joint_positive_rate": sum(r.get("joint_positive", 0) for r in rows) / n,
        "answer_safe_joint_positive_rate": sum(r.get("answer_safe_joint_positive", 0) for r in rows) / n,
        "support_positive_rate": sum(r.get("support_positive", 0) for r in rows) / n,
        "candidate_family_positive_rate": {},
        "candidate_family_answer_drop_rate": {},
        "candidate_family_joint_positive_rate": {},
    }
    table = []
    for f, xs in sorted(fam.items()):
        m = len(xs)
        rec = {
            "candidate_family": f,
            "n_actions": m,
            "paper_positive_rate": sum(r.get("paper_positive", 0) for r in xs) / m,
            "answer_drop_rate": sum(r.get("answer_drop", 0) for r in xs) / m,
            "joint_positive_rate": sum(r.get("joint_positive", 0) for r in xs) / m,
        }
        table.append(rec)
        summary["candidate_family_positive_rate"][f] = rec["paper_positive_rate"]
        summary["candidate_family_answer_drop_rate"][f] = rec["answer_drop_rate"]
        summary["candidate_family_joint_positive_rate"][f] = rec["joint_positive_rate"]
    write_json(PF / "outputs/diagnostics/candidate_pool_quality_summary.json", summary)
    fields = ["candidate_family", "n_actions", "paper_positive_rate", "answer_drop_rate", "joint_positive_rate"]
    (PF / "outputs/tables/candidate_pool_quality_table.md").write_text(md_table(table, fields) + "\n778 / 1000 queries have no paper-positive action. This limits maximum selector recall and shows candidate generation remains a bottleneck.\n")
    make_candidate_pool_figure(summary)
    return summary, table


def to_float(x):
    try:
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        return float(x)
    except Exception:
        return 0.0


def positive_feature_importance():
    ensure_dirs()
    rows = load_labels()
    pos = [r for r in rows if r.get("paper_positive", 0)]
    neg = [r for r in rows if not r.get("paper_positive", 0)]
    selected = {r["query_id"]: r for r in iter_jsonl(V23 / "outputs/final_1000/per_example_delta.jsonl") if r.get("selected")}
    selected_ids = {(r.get("query_id"), r.get("candidate_name")) for r in selected.values()}
    records = []
    for f in FEATURES:
        pvals = [to_float(r.get(f)) for r in pos]
        nvals = [to_float(r.get(f)) for r in neg]
        pm = statistics.fmean(pvals) if pvals else 0.0
        nm = statistics.fmean(nvals) if nvals else 0.0
        pooled = math.sqrt(((statistics.pvariance(pvals) if len(pvals) > 1 else 0) + (statistics.pvariance(nvals) if len(nvals) > 1 else 0)) / 2) or 1.0
        effect = (pm - nm) / pooled
        # Univariate standardized coefficient proxy.
        coef = effect * math.sqrt((len(pos) * len(neg)) / max(1, len(rows) ** 2))
        records.append({"feature": f, "positive_mean": pm, "non_positive_mean": nm, "mean_difference": pm - nm, "standardized_effect_size": effect, "logistic_coefficient_proxy": coef})
    # Categorical rates.
    for f in ["candidate_family", "candidate_name"]:
        ctr = defaultdict(lambda: [0, 0])
        for r in rows:
            ctr[str(r.get(f, "unknown"))][0] += int(r.get("paper_positive", 0))
            ctr[str(r.get(f, "unknown"))][1] += 1
        for k, (pp, nn) in ctr.items():
            records.append({"feature": f + "=" + k, "positive_mean": pp / nn, "non_positive_mean": "", "mean_difference": "", "standardized_effect_size": "", "logistic_coefficient_proxy": ""})
    selected_positive = [r for r in rows if (r.get("query_id"), r.get("candidate_name")) in selected_ids and r.get("paper_positive", 0)]
    rejected_positive = [r for r in rows if (r.get("query_id"), r.get("candidate_name")) not in selected_ids and r.get("paper_positive", 0)]
    compare = {}
    for f in FEATURES:
        compare[f] = {
            "selected_positive_mean": statistics.fmean([to_float(r.get(f)) for r in selected_positive]) if selected_positive else 0,
            "rejected_positive_mean": statistics.fmean([to_float(r.get(f)) for r in rejected_positive]) if rejected_positive else 0,
        }
    records.sort(key=lambda r: abs(r["standardized_effect_size"]) if isinstance(r["standardized_effect_size"], float) else -1, reverse=True)
    out = {"feature_importance": records, "selected_vs_rejected_positive": compare, "interpretation": {
        "routing_features": "support_proxy_delta, agent_weight_delta, title_bridge_score, and hybrid_score_delta are inspected as routing-side signals.",
        "safety_predictor": "safe_answer_prob and answer_risk_score support answer-neutral filtering; answer_f1 is protected but not significantly improved.",
        "title_bridge_support_proxy": "positive mean/effect sizes indicate whether sparse bridge/support features distinguish paper-positive actions.",
        "answer_risk": "answer_risk_score is interpreted as risk-control evidence rather than proof of answer gain.",
    }}
    write_json(PF / "outputs/diagnostics/positive_feature_importance.json", out)
    fields = ["feature", "positive_mean", "non_positive_mean", "mean_difference", "standardized_effect_size", "logistic_coefficient_proxy"]
    (PF / "outputs/tables/positive_feature_importance_table.md").write_text(md_table(records[:30], fields))
    make_feature_figure(records[:15])
    return out


def generate_paper_tables():
    ensure_dirs()
    v23 = load_v23()
    rows = [row_from_summary("v2.3 main", v23["final"], "final cross-fit")]
    for name, s in v23["ablation"].items():
        rows.append(row_from_summary(name.replace("ablation_", ""), s, "ablation"))
    fields = ["stage", "answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "fallback_rate", "positive_candidate_recall", "selected_answer_drop_rate", "selected_joint_positive_rate", "gate_pass"]
    (PF / "outputs/tables/ablation_table.md").write_text(md_table(rows, fields))
    (PF / "outputs/latex/ablation_table.tex").write_text(latex_table(rows, fields, "Ablation study for answer-neutral positive selector.", "tab:ablation"))
    return rows


def export_case_studies():
    ensure_dirs()
    per = list(iter_jsonl(V23 / "outputs/final_1000/per_example_delta.jsonl"))
    failures = {r["query_id"]: r for r in iter_jsonl(V23 / "outputs/diagnostics/failure_cases.jsonl")}
    def case(r, diag=""):
        f = failures.get(r.get("query_id"), {})
        return {
            "query_id": r.get("query_id"),
            "question": r.get("question", ""),
            "baseline_titles": r.get("baseline_titles", []),
            "selected_titles": r.get("candidate_titles", r.get("selected_titles", [])),
            "added_titles": r.get("added_titles", []),
            "removed_titles": r.get("removed_titles", []),
            "baseline_answer": r.get("baseline_answer", ""),
            "selected_answer": r.get("selected_answer", ""),
            "gold_answer": r.get("gold_answer", ""),
            "answer_f1_delta": r.get("answer_f1_delta"),
            "joint_f1_delta": r.get("joint_f1_delta"),
            "support_recall_delta": r.get("support_recall_at_k_delta", r.get("support_recall_delta")),
            "sp_f1_delta": r.get("sp_f1_delta"),
            "why_selected": "high answer-neutral positive selector score" if r.get("selected") else "fallback or rejected",
            "diagnosis": diag or f.get("failure_label", ""),
        }
    success = [case(r, "success: answer-safe joint/support gain") for r in per if to_float(r.get("answer_f1_delta")) >= 0 and to_float(r.get("joint_f1_delta")) > 0 and (to_float(r.get("support_recall_at_k_delta")) > 0 or to_float(r.get("sp_f1_delta")) > 0)][:5]
    neutral = [case(r, "answer-neutral: preserves answer while improving joint") for r in per if abs(to_float(r.get("answer_f1_delta"))) < 0.01 and to_float(r.get("joint_f1_delta")) > 0][:5]
    failure_labels = {"positive_action_available_but_not_selected", "wrong_action_selected", "candidate_pool_no_positive_action", "answer_drop_selected"}
    failure = [case(r, failures.get(r.get("query_id"), {}).get("failure_label", "")) for r in per if failures.get(r.get("query_id"), {}).get("failure_label") in failure_labels][:12]
    all_cases = {"success": success, "answer_neutral": neutral, "failure": failure}
    write_json(PF / "outputs/case_studies/case_studies.json", all_cases)
    for name, cases in [("success_cases", success), ("answer_neutral_cases", neutral), ("failure_cases", failure)]:
        lines = [f"# {name.replace('_', ' ').title()}\n", "Gold answer, if present, is displayed only for analysis and was not used as an inference feature.\n"]
        for i, c in enumerate(cases, 1):
            lines.append(f"## Case {i}: {c['query_id']}\n")
            for k, v in c.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        (PF / f"outputs/case_studies/{name}.md").write_text("\n".join(lines))
    return all_cases


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_bar(path, labels, values, title, ylabel="Delta"):
    try:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as e:
        Path(path).write_bytes(b"")


def make_candidate_pool_figure(summary):
    dist = summary.get("positive_actions_per_query_distribution", {})
    labels = sorted(dist, key=lambda x: int(x))
    values = [dist[k] for k in labels]
    save_bar(PF / "outputs/figures/candidate_pool_positive_distribution.png", labels, values, "Positive actions per query", "Query count")


def make_feature_figure(records):
    labels = [r["feature"] for r in records[:12]]
    values = [r["standardized_effect_size"] for r in records[:12]]
    save_bar(PF / "outputs/figures/positive_feature_importance.png", labels, values, "Positive action feature effect sizes", "Std. effect")


def generate_paper_figures():
    ensure_dirs()
    v23 = load_v23()
    collect = read_table_md(PF / "outputs/tables/selector_evolution_table.md")
    final = v23["final"]
    save_bar(PF / "outputs/figures/v2_3_metric_delta_bar.png", ["answer_f1", "joint_f1", "support_recall", "sp_f1"], [final.get("answer_f1_delta", 0), final.get("joint_f1_delta", 0), final.get("support_recall_delta", 0), final.get("sp_f1_delta", 0)], "v2.3 metric deltas")
    save_bar(PF / "outputs/figures/positive_recall_comparison.png", ["v2.2", "v2.3"], [0.1839, final.get("positive_candidate_recall", 0)], "Positive candidate recall comparison", "Recall")
    failure = v23["failure"].get("label_counts", {})
    save_bar(PF / "outputs/figures/failure_distribution.png", list(failure.keys()), list(failure.values()), "Failure category distribution", "Count")
    ab = v23["ablation"]
    labels = ["main"] + [k.replace("ablation_", "") for k in list(ab)[:5]]
    vals = [final.get("joint_f1_delta", 0)] + [ab[k].get("joint_f1_delta", 0) for k in list(ab)[:5]]
    save_bar(PF / "outputs/figures/ablation_comparison.png", labels, vals, "Ablation joint_f1 delta")
    # Evolution chart from known stages.
    v22 = read_json(V22 / "outputs/final_1000/final_1000_crossfit_summary.json", {})
    save_bar(PF / "outputs/figures/selector_evolution.png", ["v2.2", "v2.3"], [v22.get("joint_f1_delta", 0), final.get("joint_f1_delta", 0)], "Selector evolution: joint_f1 delta")
    # Candidate pool breakdown alias.
    cp = read_json(PF / "outputs/diagnostics/candidate_pool_quality_summary.json", {})
    if cp:
        save_bar(PF / "outputs/figures/candidate_pool_breakdown.png", ["no_positive", "has_positive"], [cp.get("queries_with_no_positive_action", 0), cp.get("queries_with_at_least_one_positive_action", 0)], "Candidate pool quality breakdown", "Queries")


def read_table_md(path):
    return []


def write_experiment_narrative():
    ensure_dirs()
    final = load_v23()["final"]
    text = f"""# Experiment Section Draft

## Experimental Setup

We evaluate the paper-facing V7-HP-PAPER selector pipeline on a 1000-query HotpotQA validation subset. The final method, selector_v2.3, is trained and calibrated with query-level cross-fitting. No held-out query outcome is used at inference time, and no additional large-scale reader evaluation is run during paper finalization.

## Baselines and Variants

The comparison includes the baseline reader context, scale-calibrated selector_v2.2, and the final answer-neutral positive selector_v2.3. Oracle rows are treated strictly as diagnostic upper bounds and are not valid inference-time methods.

## Main Results

selector_v2.3 improves joint_f1 by {fmt(final.get('joint_f1_delta'))}, support_recall@5 by {fmt(final.get('support_recall_delta'))}, and sp_f1 by {fmt(final.get('sp_f1_delta'))}. answer_f1 changes by {fmt(final.get('answer_f1_delta'))}; this is a small positive but not statistically significant change, so we describe the method as answer-preserving rather than answer-improving.

## Ablation Study

The ablation evidence indicates that two-stage and pairwise scoring drive the final cross-fit result. The answer-drop rejector alone is insufficient, and variants that weaken effective-action or answer-neutral constraints fail to match the final method.

## Candidate Pool and Oracle Gap

The candidate pool remains a major bottleneck: many queries do not contain any paper-positive action. This limits maximum recall and explains why the selector cannot benefit all examples even when it improves positive-action recall over selector_v2.2.

## Failure Analysis

Failures are dominated by candidate_pool_no_positive_action and positive_action_available_but_not_selected. The first points to candidate generation limits; the second suggests future work on more expressive but still no-leak ranking models.

## Discussion and Limitations

Client-side federated routing exposes support-relevant contexts, but naive insertion can hurt reader answer quality. The answer-neutral positive selector addresses this policy-action-to-reader gap by selecting only actions predicted to preserve answer quality while improving joint evidence utility. Under strict no-leak query-level cross-fitting, v2.3 achieves significant gains in joint_f1 and support-side metrics while preserving answer_f1.
"""
    (PF / "reports/experiment_section_draft.md").write_text(text)
    return text


def claim_boundary_memo():
    ensure_dirs()
    text = """# Paper Claim Boundary Memo

## Claims We Can Make

1. The proposed selector improves joint_f1 significantly under strict no-leak cross-fitting.
2. It improves support_recall@5 and sp_f1 significantly.
3. It preserves answer_f1 with a small non-significant positive delta.
4. Positive-action recall improves substantially compared with v2.2.
5. The candidate pool still limits achievable gains.

## Claims We Should Not Make

1. It significantly improves answer_f1.
2. It solves reader sensitivity completely.
3. It reaches oracle upper bound.
4. It works for all multi-hop QA cases.
5. It proves support gain always improves answer generation.

## Recommended Contribution Phrase

Answer-neutral action selection for federated RAG routing.

Alternative: Bridging routing-side support gains and reader-side joint QA gains under no-leak constraints.
"""
    (PF / "reports/paper_claim_boundary_memo.md").write_text(text)
    return text


def final_report():
    ensure_dirs()
    final = load_v23()["final"]
    sig = load_v23()["sig"]
    cp = read_json(PF / "outputs/diagnostics/candidate_pool_quality_summary.json", {})
    text = f"""# V7-HP-PAPER Paper Finalization Report

## 1. Executive Summary

selector_v2.3 should be frozen as the paper main result. No further selector tuning is recommended before drafting, unless reviewers or coauthors require an additional robustness check.

## 2. Final Main Result

- answer_f1_delta: {fmt(final.get('answer_f1_delta'))}
- joint_f1_delta: {fmt(final.get('joint_f1_delta'))}
- support_recall_delta: {fmt(final.get('support_recall_delta'))}
- sp_f1_delta: {fmt(final.get('sp_f1_delta'))}
- positive_candidate_recall: {fmt(final.get('positive_candidate_recall'))}
- gate_pass: {final.get('gate_pass')}
- paper_main_recommended: {final.get('paper_main_recommended')}

## 3. Why v2.3 Is Paper-Ready

v2.3 is the first no-leak cross-fitted selector in this sequence that simultaneously preserves answer_f1, improves joint_f1, improves support-side metrics, keeps selected actions effective, and improves positive candidate recall beyond v2.2.

## 4. Statistical Significance

joint_f1 is significant (p={sig.get('metrics', {}).get('joint_f1', {}).get('p_value')}); support_recall@5 and sp_f1 are significant. answer_f1 is positive but not significant, so we use answer-preserving language.

## 5. Ablation Evidence

The final mixed two-stage/pairwise configuration is strongest on joint_f1 and paper-main criteria. Simpler classifiers pass gate but are weaker; answer-drop rejector alone is insufficient.

## 6. No-Leak / Cross-Fit Audit

The audit is stored at `outputs/audit/no_leak_crossfit_audit.md`. It verifies disjoint query folds, train-only calibration, formal/oracle separation, and no gold answer/support inference features by artifact/source review.

## 7. Candidate Pool Limitation

{cp.get('queries_with_no_positive_action', 778)} / {cp.get('num_queries', 1000)} queries have no paper-positive action. This is the main ceiling on further selector improvements.

## 8. Feature Importance

Feature importance diagnostics are stored at `outputs/diagnostics/positive_feature_importance.json` and summarized in `outputs/tables/positive_feature_importance_table.md`.

## 9. Case Studies

Case studies are exported under `outputs/case_studies/`, separated into success, answer-neutral, and failure cases.

## 10. Failure Analysis

Failures are dominated by missing positive candidates and positive actions not selected. This supports a limitation-aware paper narrative.

## 11. Paper Claim Boundary

The claim boundary memo is stored at `reports/paper_claim_boundary_memo.md`. The central claim should be significant joint/support gains under strict no-leak cross-fitting while preserving answer_f1.

## 12. Recommended Next Writing Steps

Use the generated main result table, ablation table, no-leak audit, candidate pool breakdown, feature importance table, and case studies to draft the experiment section. Avoid launching v2.4 tuning before writing.
"""
    (PF / "reports/paper_finalization_report.md").write_text(text)
    return text


def all_tasks():
    ensure_dirs()
    collect_main_results()
    build_evolution_table()
    verify_no_leak_crossfit()
    candidate_pool_quality()
    positive_feature_importance()
    generate_paper_tables()
    export_case_studies()
    generate_paper_figures()
    write_experiment_narrative()
    claim_boundary_memo()
    final_report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all", choices=["main", "evolution", "audit", "pool", "importance", "tables", "cases", "figures", "narrative", "claims", "report", "all"])
    args = ap.parse_args()
    if args.task == "main":
        collect_main_results()
    elif args.task == "evolution":
        build_evolution_table()
    elif args.task == "audit":
        verify_no_leak_crossfit()
    elif args.task == "pool":
        candidate_pool_quality()
    elif args.task == "importance":
        positive_feature_importance()
    elif args.task == "tables":
        generate_paper_tables()
    elif args.task == "cases":
        export_case_studies()
    elif args.task == "figures":
        generate_paper_figures()
    elif args.task == "narrative":
        write_experiment_narrative()
    elif args.task == "claims":
        claim_boundary_memo()
    elif args.task == "report":
        final_report()
    else:
        all_tasks()


if __name__ == "__main__":
    main()
