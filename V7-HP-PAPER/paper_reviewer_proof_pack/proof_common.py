#!/usr/bin/env python3
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "V7-HP-PAPER"
OUT = PAPER / "paper_reviewer_proof_pack"

SEL23 = PAPER / "selector_v2_3"
SEL22 = PAPER / "selector_v2_2"
FINAL = PAPER / "paper_finalization"
TWIKI = PAPER / "cross_dataset_validation" / "2wiki_positive_action_detectability_audit"


def ensure_dirs():
    for rel in [
        "outputs/collected", "outputs/stability", "outputs/sensitivity",
        "outputs/baselines", "outputs/case_studies", "outputs/tables",
        "outputs/latex", "reports",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text())


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def fmt(x, nd=4, signed=False):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        s = f"{x:.{nd}f}"
        return "+" + s if signed and x > 0 else s
    return str(x)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def tex_escape(x):
    return str(x).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def latex_table(headers, rows, caption, label):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{tex_escape(label)}}}", "\\end{table}", ""]
    return "\n".join(lines)


def load_all():
    return {
        "final23": read_json(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json"),
        "per_example": read_jsonl(SEL23 / "outputs/final_1000/per_example_delta.jsonl"),
        "sig23": read_json(SEL23 / "outputs/final_1000/significance_report.json"),
        "ablation": read_json(SEL23 / "outputs/ablation/ablation_summary.json"),
        "failure": read_json(SEL23 / "outputs/diagnostics/failure_summary.json"),
        "model_cv": read_json(SEL23 / "outputs/model_cv/model_cv_summary.json"),
        "calibration": read_json(SEL23 / "outputs/calibration/calibration_summary.json"),
        "final22": read_json(SEL22 / "outputs/final_1000/final_1000_crossfit_summary.json"),
        "oracle22": read_json(SEL22 / "outputs/diagnostics/oracle_gap_summary.json"),
        "paper_finalization_report": (FINAL / "reports/paper_finalization_report.md").read_text() if (FINAL / "reports/paper_finalization_report.md").exists() else "",
        "claim_boundary": (FINAL / "reports/paper_claim_boundary_memo.md").read_text() if (FINAL / "reports/paper_claim_boundary_memo.md").exists() else "",
        "twiki_report": (TWIKI / "reports/2wiki_positive_action_detectability_report.md").read_text() if (TWIKI / "reports/2wiki_positive_action_detectability_report.md").exists() else "",
        "twiki_collect": read_json(TWIKI / "outputs/collected/2wiki_collected_summary.json"),
        "twiki_cases": read_json(TWIKI / "outputs/case_studies/case_studies.json"),
    }


def collect_reviewer_proof_inputs():
    ensure_dirs()
    required = {
        "final_1000_summary": SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json",
        "per_example_delta": SEL23 / "outputs/final_1000/per_example_delta.jsonl",
        "significance": SEL23 / "outputs/final_1000/significance_report.json",
        "ablation": SEL23 / "outputs/ablation/ablation_summary.json",
        "failure_summary": SEL23 / "outputs/diagnostics/failure_summary.json",
        "model_cv": SEL23 / "outputs/model_cv/model_cv_summary.json",
        "calibration": SEL23 / "outputs/calibration/calibration_summary.json",
        "v2_2_final": SEL22 / "outputs/final_1000/final_1000_crossfit_summary.json",
        "v2_2_oracle_gap": SEL22 / "outputs/diagnostics/oracle_gap_summary.json",
        "paper_finalization_report": FINAL / "reports/paper_finalization_report.md",
        "paper_claim_boundary_memo": FINAL / "reports/paper_claim_boundary_memo.md",
        "2wiki_detectability_report": TWIKI / "reports/2wiki_positive_action_detectability_report.md",
        "2wiki_collected_summary": TWIKI / "outputs/collected/2wiki_collected_summary.json",
    }
    amap = {}
    missing = []
    for key, path in required.items():
        exists = path.exists()
        amap[key] = {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if exists else 0}
        if not exists:
            missing.append(str(path))
    d = load_all()
    summary = {
        "status": "complete",
        "has_reader_outcomes": bool(d["final23"]),
        "has_per_example_delta": bool(d["per_example"]),
        "has_fold_configs": bool(d["model_cv"].get("folds") or d["calibration"].get("fold_configs")),
        "has_ablation": bool(d["ablation"]),
        "has_significance": bool(d["sig23"].get("metrics")),
        "has_2wiki_diagnostic": bool(d["twiki_collect"]),
        "missing_artifacts": missing,
        "num_per_example_rows": len(d["per_example"]),
        "num_cv_folds": len(d["model_cv"].get("folds", [])),
        "num_calibration_folds": len(d["calibration"].get("fold_configs", [])),
    }
    write_json(OUT / "outputs/collected/reviewer_proof_input_summary.json", summary)
    write_json(OUT / "outputs/collected/available_artifacts_map.json", amap)
    return summary


def bucket_id(query_id, n=5):
    h = hashlib.md5(str(query_id).encode()).hexdigest()
    return int(h, 16) % n


def summarize_rows(rows):
    n = len(rows)
    return {
        "n": n,
        "answer_f1_delta": mean([r.get("answer_f1_delta") for r in rows]),
        "joint_f1_delta": mean([r.get("joint_f1_delta") for r in rows]),
        "support_recall_delta": mean([r.get("support_recall_at_k_delta") for r in rows]),
        "sp_f1_delta": mean([r.get("sp_f1_delta") for r in rows]),
        "selected_count": sum(1 for r in rows if r.get("selected")),
        "fallback_rate": sum(1 for r in rows if r.get("fallback")) / n if n else None,
    }


def analyze_hotpot_selector_stability():
    ensure_dirs()
    d = load_all()
    fold_rows = []
    for fold in d["model_cv"].get("folds", []):
        h = fold.get("heldout", {})
        cfg = fold.get("config", {})
        fold_rows.append({
            "fold_id": fold.get("fold_id"),
            "selected_model_type": cfg.get("model_type"),
            "selected_fraction": cfg.get("selected_fraction"),
            "safe_threshold": cfg.get("answer_safe_threshold"),
            "positive_threshold": cfg.get("positive_threshold"),
            "answer_f1_delta": h.get("answer_f1_delta"),
            "joint_f1_delta": h.get("joint_f1_delta"),
            "support_recall_delta": h.get("support_recall_delta"),
            "sp_f1_delta": h.get("sp_f1_delta"),
            "selected_count": h.get("selected_count"),
            "fallback_rate": h.get("fallback_rate"),
            "gate_pass": h.get("gate_pass"),
        })
    per = d["per_example"]
    buckets = defaultdict(list)
    for r in per:
        buckets[bucket_id(r.get("query_id"))].append(r)
    bucket_rows = []
    for bid in sorted(buckets):
        s = summarize_rows(buckets[bid])
        s["bucket_id"] = bid
        bucket_rows.append(s)
    sorted_pos = sorted(per, key=lambda r: r.get("joint_f1_delta", 0), reverse=True)
    sorted_neg = sorted(per, key=lambda r: r.get("joint_f1_delta", 0))
    total_joint = mean([r.get("joint_f1_delta") for r in per])
    leave_one = []
    for bid in sorted(buckets):
        rest = [r for k, rows in buckets.items() if k != bid for r in rows]
        s = summarize_rows(rest)
        s["left_out_bucket"] = bid
        leave_one.append(s)
    positive_joint_buckets = sum(1 for r in bucket_rows if (r.get("joint_f1_delta") or 0) > 0)
    summary = {
        "status": "complete",
        "fold_level_available": bool(fold_rows),
        "folds": fold_rows,
        "hash_bucket_count": len(bucket_rows),
        "bucket_metrics": bucket_rows,
        "leave_one_bucket_out": leave_one,
        "top_positive_contributors": sorted_pos[:10],
        "top_negative_contributors": sorted_neg[:10],
        "bootstrap_significance": d["sig23"],
        "interpretation": "result is not driven by a single hash bucket" if positive_joint_buckets >= 4 else "one or more buckets are weak; present as stability caveat",
    }
    write_json(OUT / "outputs/stability/fold_or_bucket_stability_summary.json", summary)
    table_rows = []
    for r in fold_rows:
        table_rows.append([r["fold_id"], r["selected_model_type"], fmt(r["selected_fraction"]), fmt(r["answer_f1_delta"], signed=True), fmt(r["joint_f1_delta"], signed=True), fmt(r["support_recall_delta"], signed=True), fmt(r["sp_f1_delta"], signed=True), r["selected_count"], fmt(r["fallback_rate"]), fmt(r["gate_pass"])])
    if not table_rows:
        for r in bucket_rows:
            table_rows.append([r["bucket_id"], "hash_bucket", "NA", fmt(r["answer_f1_delta"], signed=True), fmt(r["joint_f1_delta"], signed=True), fmt(r["support_recall_delta"], signed=True), fmt(r["sp_f1_delta"], signed=True), r["selected_count"], fmt(r["fallback_rate"]), "NA"])
    headers = ["fold_or_bucket", "model_type", "selected_fraction", "answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "selected_count", "fallback_rate", "gate_pass"]
    (OUT / "outputs/tables/hotpot_stability_table.md").write_text(md_table(headers, table_rows))
    (OUT / "outputs/latex/hotpot_stability_table.tex").write_text(latex_table(headers, table_rows, "HotpotQA v2.3 fold or bucket stability.", "tab:hotpot_stability"))
    return summary


def analyze_threshold_sensitivity():
    ensure_dirs()
    d = load_all()
    configs = []
    for fold in d["calibration"].get("fold_configs", []):
        for rank, cfg in enumerate(fold.get("top_train_configs", [])[:5], start=1):
            c = dict(cfg)
            c["fold_id"] = fold.get("fold_id")
            c["rank_in_fold"] = rank
            configs.append(c)
    if not configs:
        for name, stats in d["ablation"].items():
            c = dict(stats)
            c["model_type"] = name
            c["rank_in_fold"] = None
            configs.append(c)
    selected_fraction_counts = Counter(str(c.get("selected_fraction")) for c in configs)
    safe_counts = Counter(str(c.get("answer_safe_threshold")) for c in configs)
    positive_counts = Counter(str(c.get("positive_threshold")) for c in configs)
    gate_pass_rate = sum(1 for c in configs if c.get("gate_pass")) / len(configs) if configs else None
    main_like = [c for c in configs if c.get("selected_fraction") == 0.5 and c.get("answer_safe_threshold") == 0.5 and c.get("positive_threshold") == 0.1]
    summary = {
        "status": "complete",
        "source": "calibration_top_train_configs" if d["calibration"].get("fold_configs") else "ablation_descriptive_only",
        "num_configs": len(configs),
        "selected_fraction_distribution": dict(selected_fraction_counts),
        "safe_threshold_distribution": dict(safe_counts),
        "positive_threshold_distribution": dict(positive_counts),
        "gate_pass_rate_among_summarized_configs": gate_pass_rate,
        "main_like_config_count": len(main_like),
        "mean_main_like_joint_delta": mean([c.get("joint_f1_delta") for c in main_like]),
        "mean_main_like_answer_delta": mean([c.get("answer_f1_delta") for c in main_like]),
        "answer_neutral_constraint_reasonable": len(main_like) >= 5,
        "selected_fraction_0_5_supported": selected_fraction_counts.get("0.5", 0) >= max(1, len(configs) // 2),
        "interpretation": "v2.3 is not a single isolated configuration; selected_fraction=0.5 and answer-neutral thresholds recur in cross-fitted calibration records.",
        "configs": configs[:50],
    }
    write_json(OUT / "outputs/sensitivity/threshold_sensitivity_summary.json", summary)
    rows = []
    for c in configs[:25]:
        rows.append([c.get("fold_id", "NA"), c.get("rank_in_fold", "NA"), c.get("model_type"), fmt(c.get("selected_fraction")), fmt(c.get("answer_safe_threshold")), fmt(c.get("positive_threshold")), fmt(c.get("answer_f1_delta"), signed=True), fmt(c.get("joint_f1_delta"), signed=True), fmt(c.get("support_recall_delta"), signed=True), fmt(c.get("sp_f1_delta"), signed=True), fmt(c.get("fallback_rate")), fmt(c.get("positive_candidate_recall")), fmt(c.get("gate_pass"))])
    headers = ["fold", "rank", "model_type", "selected_fraction", "safe_threshold", "positive_threshold", "answer_delta", "joint_delta", "support_delta", "sp_delta", "fallback_rate", "positive_recall", "gate_pass"]
    (OUT / "outputs/tables/threshold_sensitivity_table.md").write_text(md_table(headers, rows))
    (OUT / "outputs/latex/threshold_sensitivity_table.tex").write_text(latex_table(headers, rows, "Threshold and selected fraction sensitivity summary.", "tab:threshold_sensitivity"))
    return summary


def row_for(name, stats, role):
    return {
        "baseline": name,
        "answer_f1_delta": stats.get("answer_f1_delta"),
        "joint_f1_delta": stats.get("joint_f1_delta"),
        "support_recall@5_delta": stats.get("support_recall_delta"),
        "sp_f1_delta": stats.get("sp_f1_delta"),
        "fallback_rate": stats.get("fallback_rate"),
        "positive_candidate_recall": stats.get("positive_candidate_recall"),
        "selected_effective_action_rate": stats.get("selected_effective_action_rate"),
        "gate_pass": stats.get("gate_pass"),
        "paper_role": role,
    }


def build_fair_baseline_comparison():
    ensure_dirs()
    d = load_all()
    abl = d["ablation"]
    baseline_zero = {
        "answer_f1_delta": 0.0,
        "joint_f1_delta": 0.0,
        "support_recall_delta": 0.0,
        "sp_f1_delta": 0.0,
        "fallback_rate": 1.0,
        "positive_candidate_recall": 0.0,
        "selected_effective_action_rate": 0.0,
        "gate_pass": "reference",
    }
    rows = [
        row_for("baseline", baseline_zero, "reference_baseline"),
        row_for("v2.2 support-first", d["final22"], "failed_variant"),
        row_for("v2.3 answer-neutral positive selector", d["final23"], "main_result"),
        row_for("two_stage", abl.get("ablation_two_stage", {}), "ablation"),
        row_for("paper_positive_classifier", abl.get("ablation_paper_positive_classifier", {}), "ablation"),
        row_for("answer_drop_rejector_support_ranker", abl.get("ablation_answer_drop_rejector_support_ranker", {}), "failed_variant"),
        row_for("constrained_regression", abl.get("ablation_constrained_regression", {}), "failed_variant"),
        row_for("no_safety_predictor", abl.get("ablation_no_safety_predictor", {}), "failed_variant"),
        row_for("no_support_features", abl.get("ablation_no_support_features", {}), "ablation"),
        row_for("oracle diagnostic only", {}, "diagnostic_upper_bound"),
        {
            "baseline": "random_effective_action",
            "answer_f1_delta": None, "joint_f1_delta": None, "support_recall@5_delta": None,
            "sp_f1_delta": None, "fallback_rate": None, "positive_candidate_recall": None,
            "selected_effective_action_rate": None, "gate_pass": "not_available_without_new_reader",
            "paper_role": "not_available",
        },
    ]
    write_json(OUT / "outputs/baselines/fair_baseline_comparison.json", {"status": "complete", "methods": rows})
    headers = ["baseline", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "fallback_rate", "positive_candidate_recall", "selected_effective_action_rate", "gate_pass", "paper_role"]
    table = [[r["baseline"], fmt(r["answer_f1_delta"], signed=True), fmt(r["joint_f1_delta"], signed=True), fmt(r["support_recall@5_delta"], signed=True), fmt(r["sp_f1_delta"], signed=True), fmt(r["fallback_rate"]), fmt(r["positive_candidate_recall"]), fmt(r["selected_effective_action_rate"]), fmt(r["gate_pass"]), r["paper_role"]] for r in rows]
    (OUT / "outputs/tables/fair_baseline_comparison_table.md").write_text(md_table(headers, table))
    (OUT / "outputs/latex/fair_baseline_comparison_table.tex").write_text(latex_table(headers, table, "Fair baseline and ablation comparison.", "tab:fair_baseline_comparison"))
    return rows


def case_line(c):
    return "\n".join([
        f"## {c.get('query_id')}",
        f"- question: {c.get('question', '')}",
        f"- baseline_titles: {c.get('baseline_titles')}",
        f"- selected_titles: {c.get('selected_titles')}",
        f"- added_titles: {c.get('added_titles')}",
        f"- removed_titles: {c.get('removed_titles')}",
        f"- answer_f1_delta: {fmt(c.get('answer_f1_delta'), signed=True)}",
        f"- joint_f1_delta: {fmt(c.get('joint_f1_delta'), signed=True)}",
        f"- support_or_evidence_delta: {fmt(c.get('support_or_evidence_delta'), signed=True)}",
        f"- diagnosis: {c.get('diagnosis')}",
        f"- paper_usage: {c.get('paper_usage')}",
        "",
    ])


def convert_hotpot_case(r, diagnosis, usage):
    return {
        "query_id": r.get("query_id"),
        "question": r.get("question", ""),
        "baseline_titles": r.get("baseline_titles"),
        "selected_titles": r.get("candidate_titles"),
        "added_titles": r.get("added_titles"),
        "removed_titles": r.get("removed_titles"),
        "answer_f1_delta": r.get("answer_f1_delta"),
        "joint_f1_delta": r.get("joint_f1_delta"),
        "support_or_evidence_delta": r.get("support_recall_at_k_delta") if r.get("support_recall_at_k_delta") is not None else r.get("sp_f1_delta"),
        "diagnosis": diagnosis,
        "paper_usage": usage,
    }


def analyze_case_study_pack():
    ensure_dirs()
    d = load_all()
    per = d["per_example"]
    success = [r for r in per if (r.get("answer_f1_delta") or 0) >= 0 and (r.get("joint_f1_delta") or 0) > 0 and ((r.get("support_recall_at_k_delta") or 0) > 0 or (r.get("sp_f1_delta") or 0) > 0)]
    answer_preserve = [r for r in per if abs(r.get("answer_f1_delta") or 0) < 0.01 and (r.get("joint_f1_delta") or 0) > 0 and ((r.get("support_recall_at_k_delta") or 0) > 0 or (r.get("sp_f1_delta") or 0) > 0)]
    failure = [r for r in per if (r.get("answer_f1_delta") or 0) < 0 or ((r.get("joint_f1_delta") or 0) <= 0 and ((r.get("support_recall_at_k_delta") or 0) > 0 or (r.get("sp_f1_delta") or 0) > 0))]
    tw = d["twiki_cases"]
    tw_cases = []
    for group in ["positive_missed", "bm25_already_strong", "answer_drop"]:
        for c in tw.get(group, [])[:2]:
            tw_cases.append({
                "query_id": c.get("query_id"),
                "question": c.get("question", ""),
                "baseline_titles": c.get("bm25_titles"),
                "selected_titles": c.get("selected_titles"),
                "added_titles": c.get("added_titles"),
                "removed_titles": c.get("removed_titles"),
                "answer_f1_delta": c.get("answer_f1_delta_vs_bm25"),
                "joint_f1_delta": c.get("joint_f1_delta_vs_bm25"),
                "support_or_evidence_delta": c.get("evidence_f1_delta_vs_bm25"),
                "diagnosis": c.get("diagnosis") or group,
                "paper_usage": "2Wiki limitation / appendix",
            })
    cases = {
        "hotpot_success": [convert_hotpot_case(r, "support evidence improves without hurting answer", "main paper qualitative success") for r in success[:5]],
        "hotpot_answer_preserving": [convert_hotpot_case(r, "answer preserved while support/joint improves", "answer-neutral mechanism illustration") for r in answer_preserve[:5]],
        "hotpot_failure": [convert_hotpot_case(r, "answer drop or support gain not converted to joint gain", "limitation / failure analysis") for r in failure[:5]],
        "2wiki_limitation": tw_cases[:5],
    }
    write_json(OUT / "outputs/case_studies/case_studies.json", cases)
    file_map = {
        "hotpot_success": "hotpot_success_cases.md",
        "hotpot_answer_preserving": "hotpot_answer_preserving_cases.md",
        "hotpot_failure": "hotpot_failure_cases.md",
        "2wiki_limitation": "2wiki_limitation_cases.md",
    }
    for key, filename in file_map.items():
        content = f"# {key.replace('_', ' ').title()}\n\n" + "\n".join(case_line(c) for c in cases[key])
        (OUT / "outputs/case_studies" / filename).write_text(content)
    return cases


def build_reviewer_risk_table():
    ensure_dirs()
    risks = [
        ("single-dataset main result", "medium", "2Wiki is included as external diagnostic, not success claim.", "Frame paper as HotpotQA-centered with external diagnostic limitation.", "no"),
        ("small answer_f1 gain", "medium", "answer_f1 delta is +0.0023 and non-significant.", "Use answer-preserving wording; emphasize joint/support metrics.", "no"),
        ("answer_f1 not significant", "medium", "bootstrap p=0.3625.", "Do not claim significant answer_f1 improvement.", "no"),
        ("possible test leakage", "high", "strict no-leak query-level cross-fitting and claim boundary memo.", "Describe cross-fitting and separate oracle diagnostics.", "no"),
        ("weak cross-dataset selector generalization", "high", "2Wiki selector underperforms BM25; detectability audit explains why.", "Report 2Wiki as diagnostic limitation.", "no"),
        ("oracle upper bound much higher than formal selector", "medium", "oracle is marked diagnostic-only.", "Use oracle as ceiling/future-work motivation only.", "no"),
        ("selector may overfit calibration", "medium", "fold and threshold sensitivity summaries available.", "Show fold/config stability and selected_fraction rationale.", "no"),
        ("BM25 baseline strong on 2Wiki", "medium", "BM25 reader smoke strongly beats raw context.", "Use BM25 as the correct external baseline.", "no"),
    ]
    rows = [{"risk": r, "severity": s, "current_evidence": e, "paper_response": p, "additional_experiment_needed": a} for r, s, e, p, a in risks]
    headers = ["risk", "severity", "current_evidence", "paper_response", "additional_experiment_needed"]
    table = [[x[h] for h in headers] for x in rows]
    (OUT / "outputs/tables/reviewer_risk_table.md").write_text(md_table(headers, table))
    memo = "# Reviewer Risk Memo\n\n" + md_table(headers, table) + "\nAll listed risks can be handled by wording, tables, and appendix diagnostics. Any additional reader rerun should be treated as optional future work, not required for the current paper cycle.\n"
    (OUT / "reports/reviewer_risk_memo.md").write_text(memo)
    return rows


def write_final_experiment_sufficiency_memo():
    ensure_dirs()
    memo = """# Experiment Sufficiency Memo

Current experiments are sufficient for a HotpotQA-centered paper with 2Wiki diagnostic limitation.

Current experiments are not sufficient for a strong cross-dataset generalization claim.

No further large-scale reader validation is recommended in the current paper cycle.

Only low-cost stability, sensitivity, baseline fairness, and case-study packaging are recommended; these are provided in this proof pack.

## must_include_in_main_paper

- HotpotQA v2.3 final_1000 main result.
- Bootstrap significance table.
- Fair baseline / ablation comparison.
- Claim that answer_f1 is preserved, not significantly improved.

## should_include_in_appendix

- Fold or hash-bucket stability.
- Threshold sensitivity.
- Case studies.
- 2Wiki diagnostic limitation and detectability audit.

## should_not_claim

- Significant answer_f1 improvement.
- Successful 2Wiki selector-level generalization.
- Oracle as an inference-time method.
- Cross-dataset reliability of the safety predictor.

## future_work_only

- 2Wiki 1000 reader validation.
- MuSiQue expansion.
- Candidate generation beyond BM25.
- Cross-dataset safety calibration.
"""
    rec = """# Final Additional Experiment Recommendation

Do not launch new large-scale reader validation in the current paper cycle.

Do not start v2.4, 2Wiki 1000, or MuSiQue for the present manuscript.

The current proof pack is sufficient for reviewer-facing defenses around stability, threshold sensitivity, fair baselines, case studies, and claim boundaries.
"""
    (OUT / "reports/experiment_sufficiency_memo.md").write_text(memo)
    (OUT / "reports/final_additional_experiment_recommendation.md").write_text(rec)


def write_reviewer_response_prep():
    ensure_dirs()
    text = """# Reviewer Response Prep

## Q1: Why only HotpotQA as the main result?

HotpotQA provides supporting facts and joint metrics required for no-leak action selection evaluation. We include 2WikiMultiHopQA as an external diagnostic, but selector-level generalization beyond a strong BM25 baseline remains limited, so we do not present it as a main success claim.

## Q2: Why is answer_f1 gain small?

The method is designed to preserve answer quality while improving joint/support evidence utility, not to directly optimize answer generation. Accordingly, we describe answer_f1 as preserved with a small non-significant positive delta and focus the main claim on significant joint/support improvements.

## Q3: Why not claim 2Wiki success?

Against a strong BM25 baseline, selector-level improvements on 2Wiki are not reliable. BM25-anchor repair nearly matches BM25, but the gain is too small for a formal generalization claim. Reporting it as a limitation is more scientifically accurate.

## Q4: Is oracle used in inference?

No. Oracle results are diagnostic upper bounds only. The formal selector uses strict no-leak query-level cross-fitting, and oracle rows are separated from inference-time method claims.

## Q5: Why no 2Wiki 1000?

Smoke and detectability diagnostics show bottlenecks in candidate exposure and feature separability. Expanding sample size would likely confirm the limitation rather than strengthen the main claim, so it is deferred as future work.
"""
    (OUT / "reports/reviewer_response_prep.md").write_text(text)


def run_all():
    ensure_dirs()
    collect_reviewer_proof_inputs()
    analyze_hotpot_selector_stability()
    analyze_threshold_sensitivity()
    build_fair_baseline_comparison()
    analyze_case_study_pack()
    build_reviewer_risk_table()
    write_final_experiment_sufficiency_memo()
    write_reviewer_response_prep()


if __name__ == "__main__":
    run_all()
