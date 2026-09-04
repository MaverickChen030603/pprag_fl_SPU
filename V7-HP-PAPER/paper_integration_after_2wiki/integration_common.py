#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "V7-HP-PAPER"
OUT = PAPER / "paper_integration_after_2wiki"

SEL23 = PAPER / "selector_v2_3"
SEL22 = PAPER / "selector_v2_2"
FINAL = PAPER / "paper_finalization"
XDATA = PAPER / "cross_dataset_validation"
TWIKI_AUDIT = XDATA / "2wiki_positive_action_detectability_audit"
TWIKI_ALIGN = XDATA / "2wiki_selector_alignment"
TWIKI_REPAIR = XDATA / "2wiki_selector_repair_bm25_anchor"


def ensure_dirs():
    for rel in ["outputs/tables", "outputs/latex", "outputs/figures", "outputs/summaries", "reports"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text())


def read_text(path, default=""):
    p = Path(path)
    return p.read_text() if p.exists() else default


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
        return ("+" + s) if signed and x > 0 else s
    return str(x)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def tex_escape(x):
    s = str(x)
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}"))


def latex_table(headers, rows, caption, label):
    cols = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(x) for x in row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{tex_escape(label)}}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def method_row(name, stats):
    return {
        "method": name,
        "answer_f1_delta": stats.get("answer_f1_delta"),
        "joint_f1_delta": stats.get("joint_f1_delta"),
        "support_recall@5_delta": stats.get("support_recall_delta"),
        "sp_f1_delta": stats.get("sp_f1_delta"),
        "fallback_rate": stats.get("fallback_rate"),
        "positive_candidate_recall": stats.get("positive_candidate_recall"),
        "gate_pass": stats.get("gate_pass"),
        "paper_main_recommended": stats.get("paper_main_recommended"),
    }


def load_all():
    final23 = read_json(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    sig23 = read_json(SEL23 / "outputs/final_1000/significance_report.json")
    abl23 = read_json(SEL23 / "outputs/ablation/ablation_summary.json")
    fail23 = read_json(SEL23 / "outputs/diagnostics/failure_summary.json")
    final22 = read_json(SEL22 / "outputs/final_1000/final_1000_crossfit_summary.json")

    tw_reader = read_json(XDATA / "outputs/2wiki_reader_smoke_300/reader_summary.json")
    tw_align = read_json(TWIKI_ALIGN / "outputs/selector_smoke_300/summary.json")
    tw_repair = read_json(TWIKI_REPAIR / "outputs/selector_smoke_300/summary.json")
    tw_collect = read_json(TWIKI_AUDIT / "outputs/collected/2wiki_collected_summary.json")
    tw_feature = read_json(TWIKI_AUDIT / "outputs/feature_margin/feature_margin_summary.json")
    tw_pool = read_json(TWIKI_AUDIT / "outputs/candidate_pool/candidate_pool_vs_bm25_summary.json")
    tw_recall = read_json(TWIKI_AUDIT / "outputs/selector_recall/selector_recall_failure_summary.json")
    tw_safety = read_json(TWIKI_AUDIT / "outputs/safety_predictor/safety_predictor_weakness_summary.json")
    return {
        "final23": final23,
        "sig23": sig23,
        "abl23": abl23,
        "fail23": fail23,
        "final22": final22,
        "tw_reader": tw_reader,
        "tw_align": tw_align,
        "tw_repair": tw_repair,
        "tw_collect": tw_collect,
        "tw_feature": tw_feature,
        "tw_pool": tw_pool,
        "tw_recall": tw_recall,
        "tw_safety": tw_safety,
    }


def collect_all_final_results():
    ensure_dirs()
    d = load_all()
    final = d["final23"]
    sig = d["sig23"].get("metrics", {})
    abl = d["abl23"]
    tw_reader = d["tw_reader"]
    tw_align = d["tw_align"]
    tw_repair = d["tw_repair"]
    tw_collect = d["tw_collect"]
    tw_feature = d["tw_feature"]
    tw_pool = d["tw_pool"]
    tw_recall = d["tw_recall"]
    tw_safety = d["tw_safety"]
    summary = {
        "status": "complete",
        "experiment_decision": {
            "freeze_hotpot_v2_3": True,
            "do_not_launch_2wiki_1000": True,
            "do_not_launch_musique": True,
            "use_2wiki_as": "external diagnostic / limitation / appendix",
        },
        "hotpotqa_formal_result": {
            "method": "selector_v2.3_answer_neutral_positive_selector",
            "n": final.get("n"),
            "answer_f1_delta": final.get("answer_f1_delta"),
            "joint_f1_delta": final.get("joint_f1_delta"),
            "support_recall@5_delta": final.get("support_recall_delta"),
            "sp_f1_delta": final.get("sp_f1_delta"),
            "answer_f1_p": sig.get("answer_f1", {}).get("p_value"),
            "joint_f1_p": sig.get("joint_f1", {}).get("p_value"),
            "support_recall_p": sig.get("support_recall@5", {}).get("p_value"),
            "sp_f1_p": sig.get("sp_f1", {}).get("p_value"),
            "positive_candidate_recall": final.get("positive_candidate_recall"),
            "fallback_rate": final.get("fallback_rate"),
            "gate_pass": final.get("gate_pass"),
            "paper_main_recommended": final.get("paper_main_recommended"),
        },
        "hotpotqa_ablation": {
            "v2.3 main": method_row("v2.3 main", final),
            "two_stage": method_row("two_stage", abl.get("ablation_two_stage", {})),
            "paper_positive_classifier": method_row("paper_positive_classifier", abl.get("ablation_paper_positive_classifier", {})),
            "answer_drop_rejector_support_ranker": method_row("answer_drop_rejector_support_ranker", abl.get("ablation_answer_drop_rejector_support_ranker", {})),
            "constrained_regression": method_row("constrained_regression", abl.get("ablation_constrained_regression", {})),
            "no_safety_predictor": method_row("no_safety_predictor", abl.get("ablation_no_safety_predictor", {})),
            "no_support_features": method_row("no_support_features", abl.get("ablation_no_support_features", {})),
            "v2.2 support_first": method_row("v2.2 support_first", d["final22"]),
        },
        "two_wiki_diagnostic": {
            "bm25_smoke_delta_vs_raw_context": tw_reader.get("deltas_vs_context_order", {}),
            "original_selector_alignment_vs_bm25": {
                "hotpot_v23_frozen_transfer": tw_align.get("methods", {}).get("hotpot_v23_frozen_transfer", {}),
                "2wiki_v23_crossfit_selector": tw_align.get("methods", {}).get("2wiki_v23_crossfit_selector", {}),
            },
            "bm25_anchor_repair_vs_bm25": tw_repair.get("methods", {}).get("bm25_anchor_answer_neutral_selector", {}),
            "oracle_positive_vs_bm25_rate": tw_collect.get("positive_vs_bm25_rate"),
            "strict_action_table_positive_rate": tw_pool.get("strict_action_label_positive_rate"),
            "candidate_pool_no_positive_vs_BM25": tw_pool.get("queries_without_positive_vs_bm25"),
            "feature_margin_conclusion": tw_feature.get("interpretation"),
            "safety_predictor_auc": tw_safety.get("answer_safe_auc"),
            "paper_positive_auc": tw_safety.get("paper_positive_auc"),
            "selector_recall_failure": {
                "oracle_positive_query_count": tw_recall.get("oracle_positive_query_count"),
                "strict_action_label_positive_query_count": tw_recall.get("strict_action_label_positive_query_count"),
                "positive_recall": tw_recall.get("positive_recall"),
                "oracle_positive_but_no_strict_positive_action_in_anchor_table": tw_recall.get("oracle_positive_but_no_strict_positive_action_in_anchor_table"),
            },
        },
    }
    evidence_map = {
        "hotpot_main_result": {
            "files": [
                str(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json"),
                str(SEL23 / "outputs/final_1000/significance_report.json"),
            ],
            "paper_claim": "formal main result under strict no-leak query-level cross-fitting",
        },
        "hotpot_ablation": {
            "files": [str(SEL23 / "outputs/ablation/ablation_summary.json")],
            "paper_claim": "safety predictor and support-side features contribute to the support-to-joint bridge",
        },
        "hotpot_claim_boundary": {
            "files": [str(FINAL / "reports/paper_claim_boundary_memo.md")],
            "paper_claim": "answer_f1 is preserved with non-significant positive delta; oracle is diagnostic only",
        },
        "2wiki_external_diagnostic": {
            "files": [
                str(XDATA / "outputs/2wiki_reader_smoke_300/reader_summary.json"),
                str(TWIKI_AUDIT / "reports/2wiki_positive_action_detectability_report.md"),
                str(TWIKI_AUDIT / "outputs/collected/2wiki_collected_summary.json"),
                str(TWIKI_AUDIT / "outputs/candidate_pool/candidate_pool_vs_bm25_summary.json"),
                str(TWIKI_AUDIT / "outputs/safety_predictor/safety_predictor_weakness_summary.json"),
            ],
            "paper_claim": "2Wiki is an external sanity check and diagnostic limitation, not a main selector generalization result",
        },
    }
    write_json(OUT / "outputs/summaries/final_result_summary.json", summary)
    write_json(OUT / "outputs/summaries/final_paper_evidence_map.json", evidence_map)
    return summary


def build_paper_main_tables():
    ensure_dirs()
    d = load_all()
    final = d["final23"]
    abl = d["abl23"]
    final22 = d["final22"]
    methods = [
        ("v2.2 support-first", final22),
        ("v2.3 answer-neutral positive selector", final),
        ("two_stage", abl.get("ablation_two_stage", {})),
        ("paper_positive_classifier", abl.get("ablation_paper_positive_classifier", {})),
        ("no_safety_predictor", abl.get("ablation_no_safety_predictor", {})),
        ("no_support_features", abl.get("ablation_no_support_features", {})),
    ]
    # Oracle is not present in final_1000 as an inference-time method; use diagnostic placeholder from paper finalization if unavailable.
    oracle = {
        "answer_f1_delta": None,
        "joint_f1_delta": None,
        "support_recall_delta": None,
        "sp_f1_delta": None,
        "fallback_rate": None,
        "positive_candidate_recall": None,
        "gate_pass": "diagnostic_only",
    }
    methods.append(("oracle diagnostic only", oracle))
    rows = []
    for name, s in methods:
        rows.append([
            name,
            fmt(s.get("answer_f1_delta"), signed=True),
            fmt(s.get("joint_f1_delta"), signed=True),
            fmt(s.get("support_recall_delta"), signed=True),
            fmt(s.get("sp_f1_delta"), signed=True),
            fmt(s.get("fallback_rate")),
            fmt(s.get("positive_candidate_recall")),
            fmt(s.get("gate_pass")),
        ])
    headers = ["method", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "fallback_rate", "positive_candidate_recall", "gate_pass"]
    note = "\nOracle diagnostic is not an inference-time method and is reported only as an upper-bound analysis.\n"
    (OUT / "outputs/tables/hotpot_main_result_table.md").write_text(md_table(headers, rows) + note)
    (OUT / "outputs/latex/hotpot_main_result_table.tex").write_text(latex_table(headers, rows, "HotpotQA main result. Oracle diagnostic is not an inference-time method and is reported only as an upper-bound analysis.", "tab:hotpot_main_result"))

    abl_methods = [
        ("v2.3 main", final),
        ("two_stage", abl.get("ablation_two_stage", {})),
        ("paper_positive_classifier", abl.get("ablation_paper_positive_classifier", {})),
        ("answer_drop_rejector_support_ranker", abl.get("ablation_answer_drop_rejector_support_ranker", {})),
        ("constrained_regression", abl.get("ablation_constrained_regression", {})),
        ("no_safety_predictor", abl.get("ablation_no_safety_predictor", {})),
        ("no_support_features", abl.get("ablation_no_support_features", {})),
        ("v2.2 support-first", final22),
    ]
    abl_rows = []
    for name, s in abl_methods:
        abl_rows.append([
            name,
            fmt(s.get("answer_f1_delta"), signed=True),
            fmt(s.get("joint_f1_delta"), signed=True),
            fmt(s.get("support_recall_delta"), signed=True),
            fmt(s.get("sp_f1_delta"), signed=True),
            fmt(s.get("selected_answer_drop_rate")),
            fmt(s.get("selected_effective_action_rate")),
            fmt(s.get("gate_pass")),
        ])
    abl_headers = ["method", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "answer_drop_rate", "effective_action_rate", "gate_pass"]
    (OUT / "outputs/tables/hotpot_ablation_table.md").write_text(md_table(abl_headers, abl_rows))
    (OUT / "outputs/latex/hotpot_ablation_table.tex").write_text(latex_table(abl_headers, abl_rows, "HotpotQA ablation summary.", "tab:hotpot_ablation"))

    sig = d["sig23"].get("metrics", {})
    sig_rows = []
    for metric, key in [("answer_f1", "answer_f1"), ("joint_f1", "joint_f1"), ("support_recall@5", "support_recall@5"), ("sp_f1", "sp_f1")]:
        m = sig.get(key, {})
        ci = m.get("ci95") or [None, None]
        sig_rows.append([metric, fmt(m.get("mean_delta"), signed=True), f"[{fmt(ci[0])}, {fmt(ci[1])}]", fmt(m.get("p_value"))])
    sig_headers = ["metric", "mean_delta", "95% CI", "p_value"]
    (OUT / "outputs/tables/hotpot_significance_table.md").write_text(md_table(sig_headers, sig_rows))
    (OUT / "outputs/latex/hotpot_significance_table.tex").write_text(latex_table(sig_headers, sig_rows, "Bootstrap significance report for the HotpotQA main result.", "tab:hotpot_significance"))


def build_2wiki_diagnostic_tables():
    ensure_dirs()
    d = load_all()
    align = d["tw_align"].get("methods", {})
    repair = d["tw_repair"].get("methods", {})
    rows = []
    specs = [
        ("Hotpot v2.3 frozen transfer", align.get("hotpot_v23_frozen_transfer", {}), "negative_transfer"),
        ("2Wiki v2.3 crossfit selector", align.get("2wiki_v23_crossfit_selector", {}), "failed_selector_alignment"),
        ("BM25-anchor answer-neutral selector", repair.get("bm25_anchor_answer_neutral_selector", {}), "near_BM25_repair"),
        ("oracle diagnostic only", repair.get("oracle_diagnostic_only", align.get("oracle_diagnostic_only", {})), "diagnostic_upper_bound"),
    ]
    for name, s, role in specs:
        rows.append([
            name,
            fmt(s.get("answer_f1_delta_vs_bm25"), signed=True),
            fmt(s.get("evidence_f1_delta_vs_bm25"), signed=True),
            fmt(s.get("joint_f1_delta_vs_bm25"), signed=True),
            role,
        ])
    headers = ["method", "answer_f1_delta_vs_BM25", "evidence_f1_delta_vs_BM25", "joint_f1_delta_vs_BM25", "paper_role"]
    (OUT / "outputs/tables/2wiki_external_diagnostic_table.md").write_text(md_table(headers, rows))
    (OUT / "outputs/latex/2wiki_external_diagnostic_table.tex").write_text(latex_table(headers, rows, "2Wiki external diagnostic against a strong BM25 baseline.", "tab:2wiki_external_diagnostic"))

    collect = d["tw_collect"]
    pool = d["tw_pool"]
    repair_best = repair.get("bm25_anchor_answer_neutral_selector", {})
    oracle_rows = [[
        f"{collect.get('num_positive_vs_bm25_queries')} / {collect.get('num_queries')}",
        f"{pool.get('strict_action_label_queries_with_positive')} / {pool.get('num_queries')}",
        f"{pool.get('queries_without_positive_vs_bm25')} / {pool.get('num_queries')}",
        fmt(collect.get("oracle_best_joint_delta_vs_bm25"), signed=True),
        fmt(repair_best.get("joint_f1_delta_vs_bm25"), signed=True),
    ]]
    oracle_headers = ["oracle_positive_queries", "strict_action_table_positive_queries", "candidate_pool_no_positive_vs_BM25", "oracle_best_joint_delta_vs_BM25", "best_no_leak_joint_delta_vs_BM25"]
    (OUT / "outputs/tables/2wiki_oracle_gap_table.md").write_text(md_table(oracle_headers, oracle_rows))
    (OUT / "outputs/latex/2wiki_oracle_gap_table.tex").write_text(latex_table(oracle_headers, oracle_rows, "2Wiki oracle opportunity and action exposure gap.", "tab:2wiki_oracle_gap"))

    feature = d["tw_feature"]
    frows = []
    for r in sorted(feature.get("feature_margin", []), key=lambda x: abs(x.get("standardized_effect_size") or 0), reverse=True)[:8]:
        frows.append([r.get("feature"), fmt(r.get("positive_mean")), fmt(r.get("non_positive_mean")), fmt(r.get("standardized_effect_size")), fmt(r.get("auc_univariate")), fmt(r.get("rank_correlation_with_joint_delta"))])
    (OUT / "outputs/tables/2wiki_feature_detectability_table.md").write_text(md_table(["feature", "positive_mean", "non_positive_mean", "effect_size", "auc", "rho_joint"], frows))

    safety = d["tw_safety"]
    srows = [
        ["answer_safe_auc", fmt(safety.get("answer_safe_auc"))],
        ["paper_positive_auc", fmt(safety.get("paper_positive_auc"))],
        ["answer_safe_rate", fmt(safety.get("answer_safe_class_balance", {}).get("safe_rate"))],
        ["paper_positive_rate", fmt(safety.get("paper_positive_class_balance", {}).get("positive_rate"))],
        ["positive_safe_prob_mean", fmt(safety.get("safe_answer_prob_distribution", {}).get("positive_mean"))],
        ["non_positive_safe_prob_mean", fmt(safety.get("safe_answer_prob_distribution", {}).get("non_positive_mean"))],
        ["answer_drop_safe_prob_mean", fmt(safety.get("safe_answer_prob_distribution", {}).get("answer_drop_mean"))],
    ]
    (OUT / "outputs/tables/2wiki_safety_predictor_table.md").write_text(md_table(["metric", "value"], srows))


def write_main_result_section():
    ensure_dirs()
    d = load_all()
    final = d["final23"]
    sig = d["sig23"].get("metrics", {})
    text = f"""# Main Result Section Draft

Under strict no-leak query-level cross-fitting, the answer-neutral positive-action selector (`selector_v2.3`) is the formal HotpotQA main result. It passes the paper gate and is recommended as the main paper result.

The selector improves the downstream joint and support-side metrics over the baseline: `joint_f1` increases by {fmt(final.get('joint_f1_delta'), signed=True)}, `support_recall@5` by {fmt(final.get('support_recall_delta'), signed=True)}, and `sp_f1` by {fmt(final.get('sp_f1_delta'), signed=True)}. The bootstrap significance report supports these gains: `joint_f1` p={fmt(sig.get('joint_f1', {}).get('p_value'))}, `support_recall@5` p={fmt(sig.get('support_recall@5', {}).get('p_value'))}, and `sp_f1` p={fmt(sig.get('sp_f1', {}).get('p_value'))}.

In contrast, `answer_f1` shows a small positive but non-significant delta ({fmt(final.get('answer_f1_delta'), signed=True)}, p={fmt(sig.get('answer_f1', {}).get('p_value'))}). We therefore describe the method as answer-preserving rather than claiming a significant answer-F1 improvement.

Under strict no-leak query-level cross-fitting, the answer-neutral positive-action selector significantly improves joint_f1 and support-side metrics while preserving answer_f1. This indicates that federated routing signals can be converted into downstream reader-side gains when routed actions are filtered through an answer-neutral selection policy.

The ablation study supports the design choice. Removing the safety predictor reduces answer-side robustness and fails the main gate, while removing support features weakens support-side and joint improvements. Earlier support-first or unconstrained variants can improve support proxies but do not provide the same answer-neutral bridge into reader-side joint gains.

Oracle diagnostics are reported only as upper-bound analyses. They are not inference-time methods and are not used to claim formal performance.
"""
    (OUT / "reports/main_result_section_draft.md").write_text(text)


def write_2wiki_limitation_section():
    ensure_dirs()
    d = load_all()
    collect = d["tw_collect"]
    pool = d["tw_pool"]
    repair_best = d["tw_repair"].get("methods", {}).get("bm25_anchor_answer_neutral_selector", {})
    safety = d["tw_safety"]
    text = f"""# 2Wiki Limitation Section Draft

We further tested the pipeline on 2WikiMultiHopQA as an external sanity check. A strong lexical/BM25 baseline substantially improved reader-backed evidence and joint metrics over the raw context order, indicating that the dataset adapter and reader evaluation pipeline transfer correctly. However, when evaluated against this strong BM25 baseline, the HotpotQA-trained selector and the 2Wiki cross-fitted selector did not establish reliable selector-level generalization.

A BM25-anchor repair reduced negative transfer and nearly matched BM25, with answer-F1 delta {fmt(repair_best.get('answer_f1_delta_vs_bm25'), signed=True)}, evidence-F1 delta {fmt(repair_best.get('evidence_f1_delta_vs_bm25'), signed=True)}, and joint-F1 delta {fmt(repair_best.get('joint_f1_delta_vs_bm25'), signed=True)} against BM25. This gain is too small to justify a full 1000-sample validation.

Oracle diagnostics show that positive actions beyond BM25 exist ({collect.get('num_positive_vs_bm25_queries')} / {collect.get('num_queries')} queries), but only {pool.get('strict_action_label_queries_with_positive')} / {pool.get('num_queries')} queries expose strict positive actions in the current BM25-anchor action table. The current no-leak features and safety predictor do not identify these actions reliably: the answer-safe AUC is {fmt(safety.get('answer_safe_auc'))} and the paper-positive AUC is {fmt(safety.get('paper_positive_auc'))}.

We therefore report 2Wiki as a diagnostic limitation rather than as a main generalization claim.
"""
    (OUT / "reports/2wiki_limitation_section_draft.md").write_text(text)


def write_appendix_2wiki_diagnostic():
    ensure_dirs()
    d = load_all()
    reader = d["tw_reader"]
    collect = d["tw_collect"]
    pool = d["tw_pool"]
    recall = d["tw_recall"]
    safety = d["tw_safety"]
    repair = d["tw_repair"].get("methods", {}).get("bm25_anchor_answer_neutral_selector", {})
    bm25_delta = reader.get("deltas_vs_context_order", {})
    text = f"""# Appendix X: Cross-Dataset Diagnostic on 2WikiMultiHopQA

## X.1 Dataset Adapter and Reader-Backed Smoke

We first validated the 2Wiki adapter and reader-backed evaluation path on dev-300. BM25/lexical routing improved over raw context order by {fmt(bm25_delta.get('answer_f1', {}).get('mean_delta'), signed=True)} answer-F1, {fmt(bm25_delta.get('sp_f1', {}).get('mean_delta'), signed=True)} SP-F1, and {fmt(bm25_delta.get('joint_f1', {}).get('mean_delta'), signed=True)} joint-F1. This establishes the external evaluation pipeline as usable.

## X.2 Selector Alignment against Strong BM25

When evaluated against the strong BM25 baseline, direct Hotpot v2.3 transfer and 2Wiki cross-fitting did not establish selector-level generalization. The appropriate baseline for this dataset is BM25/lexical routing, not raw context order.

## X.3 BM25-Anchor Repair

The BM25-anchor repair preserved the top BM25 answer anchors and reduced negative transfer. The best no-leak repair, `bm25_anchor_answer_neutral_selector`, nearly matched BM25 but produced only {fmt(repair.get('joint_f1_delta_vs_bm25'), signed=True)} joint-F1 delta, so it was not expanded to 1000 samples.

## X.4 Oracle Opportunity and Action Exposure Gap

Oracle diagnostics identified positive actions beyond BM25 for {collect.get('num_positive_vs_bm25_queries')} / {collect.get('num_queries')} queries. However, the strict BM25-anchor action table exposed positive actions for only {pool.get('strict_action_label_queries_with_positive')} / {pool.get('num_queries')} queries, leaving an action exposure gap.

## X.5 Feature Detectability and Safety Calibration

Feature-margin analysis concludes: {d['tw_feature'].get('interpretation')}. Safety calibration is weak on 2Wiki, with answer-safe AUC {fmt(safety.get('answer_safe_auc'))} and paper-positive AUC {fmt(safety.get('paper_positive_auc'))}.

## X.6 Failure Analysis

The dominant failure mode is candidate-pool limitation: {pool.get('queries_without_positive_vs_bm25')} / {pool.get('num_queries')} queries have no oracle positive action beyond BM25. Among strict action-table positive queries, selector recall is {fmt(recall.get('positive_recall'))}; many oracle-positive queries do not expose strict positive actions in the no-leak action table.

## X.7 Claim Boundary

2Wiki is reported as an external sanity check and diagnostic limitation. It is not used as a main selector-level generalization claim, and all oracle diagnostics are upper-bound analyses rather than inference-time methods.
"""
    (OUT / "reports/appendix_2wiki_diagnostic_draft.md").write_text(text)


def write_final_claim_boundary_memo():
    ensure_dirs()
    text = """# Final Claim Boundary Memo

## Claims We Can Make

1. HotpotQA v2.3 significantly improves joint_f1, support_recall@5, and sp_f1 under strict no-leak cross-fitting.
2. HotpotQA v2.3 preserves answer_f1 with a small non-significant positive delta.
3. Answer-neutral action selection helps bridge routing-side support gains and reader-side joint gains.
4. 2Wiki verifies that the adapter and reader-backed evaluation pipeline transfer to another multi-hop dataset.
5. 2Wiki reveals that cross-dataset selector generalization is limited by candidate exposure, feature detectability, and safety calibration.

## Claims We Cannot Make

1. answer_f1 significantly improves.
2. v2.3 selector generalizes successfully to 2Wiki.
3. 2Wiki validates the method as a main external result.
4. the selector reaches oracle upper bound.
5. no-leak selector can reliably identify all positive actions across datasets.

## Required Wording Discipline

Use "answer-preserving" for the HotpotQA answer-F1 result. Use "external diagnostic" or "limitation" for 2Wiki. Use "oracle diagnostic" only for upper-bound rows and never for inference-time method claims.
"""
    (OUT / "reports/final_claim_boundary_memo.md").write_text(text)


def write_full_paper_experiment_outline():
    ensure_dirs()
    decision = """# Final Experiment Decision Memo

Freeze HotpotQA v2.3 as the paper main result.

Do not launch 2Wiki 1000 reader validation.

Do not launch MuSiQue in the current paper cycle.

Use 2Wiki as diagnostic limitation / appendix.

Future work should focus on candidate generation beyond BM25 and cross-dataset safety calibration.
"""
    (OUT / "reports/final_experiment_decision_memo.md").write_text(decision)
    outline = """# Paper Experiment Outline

## 1. Main Experiment: HotpotQA

Describe the strict no-leak query-level cross-fitting setup, the reader-backed evaluation protocol, and the baseline.

## 2. Main Result: v2.3 Answer-Neutral Positive Selector

Report the formal final-1000 result. Emphasize significant gains in joint_f1, support_recall@5, and sp_f1, with answer_f1 preserved but not significantly improved.

## 3. Ablations

Compare two-stage selection, paper-positive classification, safety removal, support-feature removal, and earlier support-first variants. Use the ablations to motivate answer-neutral action selection.

## 4. Oracle Diagnostic

Report oracle rows only as upper-bound analyses and explicitly separate them from inference-time methods.

## 5. External Diagnostic: 2WikiMultiHopQA

Present 2Wiki as an external sanity check. BM25 lexical routing validates the adapter and reader path, while selector alignment and detectability audits expose cross-dataset limitations.

## 6. Limitation and Future Work

State that cross-dataset selector generalization is limited by candidate exposure, feature detectability, and safety calibration. Future work should improve candidate generation beyond BM25 and dataset-robust answer-neutral calibration.
"""
    (OUT / "reports/paper_experiment_outline.md").write_text(outline)

    contribution = """# Paper-Ready Contribution Statement

This paper introduces answer-neutral action selection for federated RAG routing. On HotpotQA, the final v2.3 selector converts routing-side support gains into significant reader-side improvements in joint_f1, support_recall@5, and sp_f1 under strict no-leak query-level cross-fitting, while preserving answer_f1 with a small non-significant positive delta.

The method should be positioned as a support-to-joint bridge rather than as an answer-F1 optimizer. The key empirical contribution is that answer-neutral filtering prevents routing improvements from damaging answer anchors, enabling support-side retrieval gains to survive downstream reader evaluation.

As an external diagnostic, 2WikiMultiHopQA shows that the adapter and reader-backed evaluation pipeline transfer to another multi-hop dataset, but selector-level generalization remains limited against a strong BM25 baseline. This limitation motivates future work on candidate generation beyond BM25 and dataset-robust safety calibration.
"""
    (OUT / "reports/paper_ready_contribution_statement.md").write_text(contribution)


def run_all():
    ensure_dirs()
    collect_all_final_results()
    build_paper_main_tables()
    build_2wiki_diagnostic_tables()
    write_main_result_section()
    write_2wiki_limitation_section()
    write_appendix_2wiki_diagnostic()
    write_final_claim_boundary_memo()
    write_full_paper_experiment_outline()


if __name__ == "__main__":
    run_all()
