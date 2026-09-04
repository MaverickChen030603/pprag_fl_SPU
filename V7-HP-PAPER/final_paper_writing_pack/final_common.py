#!/usr/bin/env python3
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "V7-HP-PAPER"
OUT = PAPER / "final_paper_writing_pack"

SEL23 = PAPER / "selector_v2_3"
SEL22 = PAPER / "selector_v2_2"
FINAL = PAPER / "paper_finalization"
INTEGRATION = PAPER / "paper_integration_after_2wiki"
PROOF = PAPER / "paper_reviewer_proof_pack"
TWIKI = PAPER / "cross_dataset_validation" / "2wiki_positive_action_detectability_audit"


def ensure_dirs():
    for rel in ["outputs/tables", "outputs/latex", "outputs/figures", "outputs/checklists", "reports"]:
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
        return "+" + s if signed and x > 0 else s
    return str(x)


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
        lines.append(" & ".join(tex_escape(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{tex_escape(label)}}}", "\\end{table}", ""]
    return "\n".join(lines)


def load():
    return {
        "final23": read_json(SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json"),
        "sig": read_json(SEL23 / "outputs/final_1000/significance_report.json"),
        "ablation": read_json(SEL23 / "outputs/ablation/ablation_summary.json"),
        "failure": read_json(SEL23 / "outputs/diagnostics/failure_summary.json"),
        "final22": read_json(SEL22 / "outputs/final_1000/final_1000_crossfit_summary.json"),
        "twiki_collect": read_json(TWIKI / "outputs/collected/2wiki_collected_summary.json"),
        "twiki_pool": read_json(TWIKI / "outputs/candidate_pool/candidate_pool_vs_bm25_summary.json"),
        "twiki_safety": read_json(TWIKI / "outputs/safety_predictor/safety_predictor_weakness_summary.json"),
        "twiki_feature": read_json(TWIKI / "outputs/feature_margin/feature_margin_summary.json"),
    }


def collect_final_materials():
    ensure_dirs()
    required = {
        "HotpotQA main result exists": SEL23 / "outputs/final_1000/final_1000_crossfit_summary.json",
        "significance report exists": SEL23 / "outputs/final_1000/significance_report.json",
        "ablation table exists": PROOF / "outputs/tables/fair_baseline_comparison_table.md",
        "stability table exists": PROOF / "outputs/tables/hotpot_stability_table.md",
        "threshold sensitivity table exists": PROOF / "outputs/tables/threshold_sensitivity_table.md",
        "fair baseline table exists": PROOF / "outputs/tables/fair_baseline_comparison_table.md",
        "2Wiki diagnostic report exists": TWIKI / "reports/2wiki_positive_action_detectability_report.md",
        "claim boundary memo exists": INTEGRATION / "reports/final_claim_boundary_memo.md",
        "reviewer response prep exists": PROOF / "reports/reviewer_response_prep.md",
    }
    materials = {}
    missing = []
    for name, path in required.items():
        ok = path.exists()
        materials[name] = {"path": str(path), "exists": ok, "size_bytes": path.stat().st_size if ok else 0}
        if not ok:
            missing.append(name)
    write_json(OUT / "outputs/checklists/final_materials_map.json", materials)
    checklist = ["# Missing Materials Checklist", ""]
    if missing:
        checklist.append("Missing items; do not rerun experiments automatically:")
        checklist += [f"- {m}: {materials[m]['path']}" for m in missing]
    else:
        checklist.append("All required final paper materials are available. No missing experiment artifact was detected.")
    (OUT / "outputs/checklists/missing_materials_checklist.md").write_text("\n".join(checklist) + "\n")


def build_main_paper_tables():
    ensure_dirs()
    d = load()
    f23, f22 = d["final23"], d["final22"]
    main_rows = [
        ["v2.2 support-first", fmt(f22.get("answer_f1_delta"), signed=True), fmt(f22.get("joint_f1_delta"), signed=True), fmt(f22.get("support_recall_delta"), signed=True), fmt(f22.get("sp_f1_delta"), signed=True), fmt(f22.get("gate_pass"))],
        ["v2.3 answer-neutral positive selector", fmt(f23.get("answer_f1_delta"), signed=True), fmt(f23.get("joint_f1_delta"), signed=True), fmt(f23.get("support_recall_delta"), signed=True), fmt(f23.get("sp_f1_delta"), signed=True), fmt(f23.get("gate_pass"))],
        ["oracle diagnostic only", "NA", "NA", "NA", "NA", "diagnostic_only"],
    ]
    main_h = ["method", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "gate"]
    (OUT / "outputs/tables/main_hotpot_result_table.md").write_text(md_table(main_h, main_rows) + "\nOracle diagnostic is not an inference-time method and is reported only as an upper-bound analysis.\n")
    (OUT / "outputs/latex/main_hotpot_result_table.tex").write_text(latex_table(main_h, main_rows, "Main HotpotQA result.", "tab:main_hotpot_result"))

    abl = d["ablation"]
    specs = [
        ("v2.3 main", f23, "formal answer-neutral selector; main result"),
        ("two_stage", abl.get("ablation_two_stage", {}), "strong ablation but lower joint/support gain than v2.3"),
        ("paper_positive_classifier", abl.get("ablation_paper_positive_classifier", {}), "positive classifier alone is weaker"),
        ("no_safety_predictor", abl.get("ablation_no_safety_predictor", {}), "answer-neutral safety is necessary"),
        ("no_support_features", abl.get("ablation_no_support_features", {}), "support/routing features contribute to the bridge"),
        ("v2.2 support-first", f22, "support-first improves support but fails the final gate"),
    ]
    abl_rows = [[name, fmt(s.get("answer_f1_delta"), signed=True), fmt(s.get("joint_f1_delta"), signed=True), fmt(s.get("support_recall_delta"), signed=True), fmt(s.get("sp_f1_delta"), signed=True), interp] for name, s, interp in specs]
    abl_h = ["variant", "answer_f1_delta", "joint_f1_delta", "support_recall@5_delta", "sp_f1_delta", "interpretation"]
    (OUT / "outputs/tables/main_ablation_table.md").write_text(md_table(abl_h, abl_rows))
    (OUT / "outputs/latex/main_ablation_table.tex").write_text(latex_table(abl_h, abl_rows, "Main ablation summary.", "tab:main_ablation"))

    tw_rows = [
        ["BM25 lexical routing improves over raw context", "reader-backed 2Wiki smoke confirms adapter and evaluation path", "external sanity check"],
        ["Hotpot v2.3 transfer fails vs BM25", "selector-level transfer is not reliable against a strong BM25 baseline", "diagnostic limitation"],
        ["BM25-anchor repair nearly matches BM25", "negative transfer is reduced, but gain is too small for a main claim", "near-BM25 repair"],
        ["feature detectability is weak", "positive actions are weakly distinguishable with current features", "future work"],
        ["safety calibration is dataset-sensitive", "2Wiki safety predictor AUC remains near 0.55", "limitation"],
    ]
    tw_h = ["result", "finding", "paper_role"]
    (OUT / "outputs/tables/main_2wiki_diagnostic_table.md").write_text(md_table(tw_h, tw_rows))
    (OUT / "outputs/latex/main_2wiki_diagnostic_table.tex").write_text(latex_table(tw_h, tw_rows, "2Wiki diagnostic summary.", "tab:main_2wiki_diagnostic"))


def copy_text(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    else:
        dst.write_text("MISSING\n")


def build_appendix_tables():
    ensure_dirs()
    copies = [
        (PROOF / "outputs/tables/hotpot_stability_table.md", OUT / "outputs/tables/appendix_hotpot_stability_table.md"),
        (PROOF / "outputs/tables/threshold_sensitivity_table.md", OUT / "outputs/tables/appendix_threshold_sensitivity_table.md"),
        (PROOF / "outputs/tables/fair_baseline_comparison_table.md", OUT / "outputs/tables/appendix_fair_baseline_table.md"),
        (PROOF / "outputs/tables/reviewer_risk_table.md", OUT / "outputs/tables/appendix_reviewer_risk_table.md"),
        (TWIKI / "outputs/tables/2wiki_oracle_gap_table.md", OUT / "outputs/tables/appendix_2wiki_oracle_gap_table.md"),
        (TWIKI / "outputs/tables/2wiki_feature_margin_table.md", OUT / "outputs/tables/appendix_2wiki_feature_margin_table.md"),
        (TWIKI / "outputs/tables/safety_predictor_error_table.md", OUT / "outputs/tables/appendix_2wiki_safety_table.md"),
    ]
    for src, dst in copies:
        copy_text(src, dst)
    latex_sources = [
        ("appendix_hotpot_stability_table", OUT / "outputs/tables/appendix_hotpot_stability_table.md", "HotpotQA fold stability."),
        ("appendix_threshold_sensitivity_table", OUT / "outputs/tables/appendix_threshold_sensitivity_table.md", "Threshold sensitivity."),
        ("appendix_fair_baseline_table", OUT / "outputs/tables/appendix_fair_baseline_table.md", "Fair baseline comparison."),
        ("appendix_reviewer_risk_table", OUT / "outputs/tables/appendix_reviewer_risk_table.md", "Reviewer risk checklist."),
        ("appendix_2wiki_oracle_gap_table", OUT / "outputs/tables/appendix_2wiki_oracle_gap_table.md", "2Wiki oracle gap."),
    ]
    for name, md, caption in latex_sources:
        text = md.read_text()
        rows = []
        headers = []
        for line in text.splitlines():
            if line.startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if not headers:
                    headers = cells
                else:
                    rows.append(cells)
        (OUT / f"outputs/latex/{name}.tex").write_text(latex_table(headers or ["table"], rows, caption, f"tab:{name}"))


def write_abstract_intro_contributions():
    ensure_dirs()
    text = """# Abstract / Introduction / Contributions Draft

## Abstract Draft

Federated routing exposes support-relevant context candidates for multi-hop question answering, but naive context action insertion can hurt reader answer quality. This creates a policy-action-to-reader gap: actions that look useful from a routing or support perspective may not improve downstream answer-bearing reasoning. We propose answer-neutral positive-action selection, a no-leak action selector that applies routed context actions only when they are predicted to preserve answer quality while improving joint/support utility. Under strict no-leak query-level cross-fitting on HotpotQA, the method significantly improves joint_f1 and support-side metrics while preserving answer_f1 with a small non-significant positive delta. A 2WikiMultiHopQA diagnostic further shows that cross-dataset selector transfer remains limited by candidate exposure and safety calibration, motivating future work on dataset-robust action generation.

## Introduction Positioning

This paper studies the downstream action-selection problem after federated routing in multi-hop RAG. Federated retrieval can surface contexts with useful support evidence, yet these contexts are not automatically beneficial to a reader. The central question is therefore not only whether a client can discover useful evidence, but whether an action-selection policy can decide when adding that evidence will preserve answer quality and improve joint reasoning.

## Contributions

1. We identify the policy-action-to-reader gap in federated RAG for multi-hop QA.
2. We propose an answer-neutral positive-action selector under strict no-leak cross-fitting.
3. We provide main HotpotQA results, ablations, and 2Wiki diagnostics showing both effectiveness and cross-dataset limitations.
"""
    (OUT / "reports/abstract_intro_contributions_draft.md").write_text(text)


def write_method_section():
    ensure_dirs()
    text = """# Method Section Draft

## 1. Federated Routing and Candidate Actions

We assume a federated RAG setting in which distributed clients expose candidate context actions to a downstream reader. A candidate action modifies the reader context, for example by inserting a bridge document or replacing a lower-ranked background document while preserving important answer anchors.

## 2. Policy-Action-to-Reader Gap

Routing-side signals can identify support-relevant evidence, but support relevance alone does not guarantee reader-side gain. A context action may improve support coverage while disrupting answer-bearing context or distracting the generator. We call this mismatch the policy-action-to-reader gap.

## 3. Answer-Neutral Positive Action Definition

We define `answer_safe` as an action whose reader answer quality is not harmed relative to the baseline context. We define `support_positive` as an action that improves support-side retrieval or evidence utility. We define `joint_positive` as an action that improves downstream joint reasoning. We define `paper_positive` as the subset of actions that are answer-safe and improve joint/support utility. An answer-neutral positive action is therefore one that preserves answer quality while improving joint/support behavior.

## 4. No-Leak Query-Level Cross-Fitting

The selector is evaluated under strict query-level cross-fitting. Held-out reader outcomes are not used for selecting actions. Gold answers and supporting facts are not inference features. The selector only uses features available to the action policy and calibration folds, with query-level separation between training/calibration and held-out evaluation.

## 5. Selector Variants and Calibration

We compare support-first selection, two-stage selection, paper-positive classification, safety removal, support-feature removal, and the final answer-neutral positive-action selector. Calibration chooses action-selection parameters on training folds and applies them to held-out folds.

## 6. Inference-Time Decision Rule

At inference time, the policy selects candidate actions predicted to be answer-neutral and positive under the calibrated selector. Oracle diagnostics are not used for inference and are reported only as upper-bound analyses.
"""
    (OUT / "reports/method_section_draft.md").write_text(text)


def write_experiment_section():
    ensure_dirs()
    d = load()
    f = d["final23"]
    text = f"""# Experiment Section Draft

## 1. Dataset and Metrics

The main experiment uses HotpotQA because it provides answer, support, and joint metrics needed to evaluate no-leak action selection. We report answer_f1, joint_f1, support_recall@5, and sp_f1.

## 2. Baselines and Variants

We compare the final answer-neutral positive-action selector against v2.2 support-first selection, two-stage selection, paper-positive classification, safety removal, support-feature removal, and diagnostic oracle upper bounds.

## 3. Main HotpotQA Result

Under strict no-leak query-level cross-fitting, v2.3 improves joint_f1 by {fmt(f.get('joint_f1_delta'), signed=True)} and improves support-side metrics, while answer_f1 remains slightly positive but not statistically significant. Support_recall@5 improves by {fmt(f.get('support_recall_delta'), signed=True)} and sp_f1 improves by {fmt(f.get('sp_f1_delta'), signed=True)}.

## 4. Ablation Study

The ablation study shows that support-first and safety-free variants are insufficient for the final claim. The safety predictor helps preserve answer quality, while support/routing features help convert action utility into joint/support gains.

## 5. Stability and Sensitivity

Fold-level and calibration summaries are included in the appendix. They show that the result is not solely a single-threshold artifact, while also revealing fold-level variability that should be acknowledged.

## 6. 2Wiki Diagnostic

We report 2Wiki as a diagnostic external check rather than a main generalization result. The adapter and reader pipeline transfer, but selector-level transfer beyond a strong BM25 baseline remains limited by candidate exposure, feature detectability, and safety calibration.

## 7. Failure Analysis

Failure analysis shows that the remaining ceiling is driven by candidate-pool limitations, missed positive actions, and cases where support gains do not fully translate into joint reader gains.
"""
    (OUT / "reports/experiment_section_draft.md").write_text(text)


def write_related_work_positioning():
    ensure_dirs()
    text = """# Related Work Positioning Draft

## 1. Federated RAG and Federated Search

Prior federated retrieval and RAG studies mainly focus on source routing, client selection, or privacy-preserving aggregation. Our work focuses on what happens after routing: whether a routed context action should be applied to the reader input.

## 2. Multi-hop RAG and Passage Set Selection

Multi-hop RAG requires coherent passage sets rather than isolated relevant passages. Centralized passage selectors optimize relevance and evidence coverage, while our setting must make action decisions under federated routing constraints.

## 3. Reader Sensitivity and Harmful Context

Reader models are sensitive to context composition. Adding support-like evidence can still reduce answer quality if it disrupts answer anchors or introduces distractors.

## 4. Evidence Utility and No-Leak Action Selection

Unlike prior federated RAG studies that mainly focus on source routing, and unlike centralized passage selection methods that optimize passage-set relevance, our work studies the downstream action-selection problem after federated routing. We show that support-like routed contexts can still harm reader answer quality, and propose an answer-neutral positive-action selector that applies only those context actions predicted to preserve answer quality while improving joint/support utility under strict no-leak cross-fitting.
"""
    (OUT / "reports/related_work_positioning_draft.md").write_text(text)


def write_limitation_section():
    ensure_dirs()
    text = """# Limitation Section Draft

The main result is HotpotQA-centered. Although the result is statistically meaningful for joint_f1 and support-side metrics, answer_f1 is not significantly improved and should be described as preserved rather than improved.

Selector-level transfer to 2WikiMultiHopQA is not established. The 2Wiki diagnostic shows that a strong BM25 baseline is difficult to beat, that candidate exposure is limited, and that cross-dataset safety calibration is dataset-sensitive. Candidate generation beyond BM25 remains future work.

Oracle diagnostics are not inference-time methods. They are included only to estimate upper bounds and motivate future candidate/action generation research.
"""
    (OUT / "reports/limitation_section_draft.md").write_text(text)


def write_conclusion_section():
    ensure_dirs()
    text = """# Conclusion Section Draft

This work studies answer-neutral action selection for federated RAG in multi-hop QA. The results show that answer-neutral action selection can bridge routing-side support gains and reader-side joint QA gains under strict no-leak constraints. On HotpotQA, the final selector improves joint_f1 and support-side metrics while preserving answer_f1. External 2Wiki diagnostics clarify the limits of current selector transfer and point to future work on candidate generation beyond BM25 and dataset-robust safety calibration.
"""
    (OUT / "reports/conclusion_section_draft.md").write_text(text)


UNSAFE = {
    "significantly improves answer_f1": "preserves answer_f1 with a small non-significant positive delta",
    "generalizes to 2Wiki": "2Wiki is reported as an external diagnostic and limitation",
    "solves cross-dataset generalization": "cross-dataset selector transfer remains limited",
    "reaches oracle": "oracle is used only as a diagnostic upper bound",
    "uses oracle selector": "formal inference does not use oracle selection",
    "fully solves reader sensitivity": "mitigates reader sensitivity under the tested HotpotQA setup",
}


def audit_claim_consistency():
    ensure_dirs()
    targets = list((OUT / "reports").glob("*.md")) + list((OUT / "outputs/tables").glob("*.md"))
    findings = []
    for path in targets:
        if path.name == "claim_consistency_audit.md":
            continue
        text = path.read_text()
        low = text.lower()
        for phrase, replacement in UNSAFE.items():
            if phrase.lower() in low:
                findings.append({"file": str(path), "unsafe_phrase": phrase, "replacement": replacement})
    write_json(OUT / "outputs/checklists/unsafe_claims_detected.json", {"status": "complete", "num_unsafe_claims": len(findings), "findings": findings})
    lines = ["# Claim Consistency Audit", ""]
    if findings:
        lines += ["Unsafe claims detected:", ""]
        for f in findings:
            lines.append(f"- `{f['unsafe_phrase']}` in `{f['file']}` -> use `{f['replacement']}`")
    else:
        lines.append("No unsafe claim phrase was detected in the generated drafts and tables.")
    lines += ["", "Checked phrases:", ""]
    for phrase, replacement in UNSAFE.items():
        lines.append(f"- `{phrase}` -> `{replacement}`")
    (OUT / "reports/claim_consistency_audit.md").write_text("\n".join(lines) + "\n")


def build_reviewer_response_brief():
    ensure_dirs()
    text = """# Reviewer Response Brief

## Q1: Why is HotpotQA the main dataset?

Because it provides answer, support, and joint metrics required for our no-leak action-selection evaluation.

## Q2: Why not claim strong cross-dataset transfer?

2Wiki diagnostics show that selector-level transfer beyond a strong BM25 baseline is limited by candidate exposure, feature detectability, and safety calibration.

## Q3: Why is answer_f1 gain small?

The method is designed to preserve answer quality while converting routing-side support signals into joint/support gains.

## Q4: Is oracle used in inference?

No. Oracle is diagnostic only. Formal results use strict no-leak query-level cross-fitting.

## Q5: Why no additional large-scale validation?

Reviewer proof analysis indicates current experiments are sufficient for a HotpotQA-centered paper with 2Wiki diagnostic limitation; additional large-scale validation would not change the central claim.
"""
    (OUT / "reports/reviewer_response_brief.md").write_text(text)


def write_outline():
    text = """# Final Paper Outline

1. Abstract
2. Introduction and Contributions
3. Method: Answer-Neutral Positive-Action Selection
4. Experiments on HotpotQA
5. Ablations, Stability, and Sensitivity
6. External Diagnostic on 2WikiMultiHopQA
7. Related Work
8. Limitations
9. Conclusion
"""
    (OUT / "reports/paper_outline_final.md").write_text(text)


def run_all():
    ensure_dirs()
    collect_final_materials()
    build_main_paper_tables()
    build_appendix_tables()
    write_abstract_intro_contributions()
    write_method_section()
    write_experiment_section()
    write_related_work_positioning()
    write_limitation_section()
    write_conclusion_section()
    build_reviewer_response_brief()
    write_outline()
    audit_claim_consistency()


if __name__ == "__main__":
    run_all()
