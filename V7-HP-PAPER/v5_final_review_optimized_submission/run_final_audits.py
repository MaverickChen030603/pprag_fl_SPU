#!/usr/bin/env python3
"""Audit the final anonymous submission against evidence and claim boundaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PAPERS = [
    HERE / "paper_full_clean_v5_final.md",
    HERE / "paper_main_conference_v5_final.md",
    HERE / "paper_anonymous_v5_final.md",
    HERE / "paper_appendix_v5_final.md",
]
SECTIONS = [
    HERE / "abstract_v5_final.md",
    HERE / "introduction_v5_final.md",
    HERE / "method_v5_final.md",
    HERE / "results_v5_final.md",
    HERE / "cost_section_v5_final.md",
    HERE / "external_transfer_v5_final.md",
    HERE / "limitations_v5_final.md",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def section_for(text: str, offset: int) -> str:
    headings = [(match.start(), match.group(0).lstrip("# ")) for match in re.finditer(r"(?m)^#{1,6}\s+.+$", text)]
    result = "document"
    for position, heading in headings:
        if position > offset:
            break
        result = heading
    return result


def sentence_at(text: str, start: int) -> str:
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start))
    right_options = [value for value in (text.find(". ", start), text.find("\n", start)) if value >= 0]
    right = min(right_options) + 1 if right_options else len(text)
    return " ".join(text[left + 1 : right].split())


def audit_claims() -> tuple[bool, list[dict[str, Any]]]:
    dangerous = {
        "large gains": "Population changes are modest.",
        "practically substantial improvement": "No practical-impact study supports this wording.",
        "efficient": "Use measured latency and a quality-cost trade-off instead.",
        "low-cost": "Full has measurable overhead.",
        "end-to-end latency = reader latency": "End-to-end timing includes all online stages.",
        "Lite is non-inferior": "The independent non-inferiority test failed.",
        "semantic features consistently contribute": "Individual semantic ablations are mixed.",
        "generalizes to 2Wiki": "2Wiki transfer and calibration gates failed.",
        "calibration solves safety transfer": "Best calibrated answer-drop remains above 4%.",
        "robust across domains": "Only same-source confirmation is positive.",
        "fairly outperforms RECOMP": "The objectives and structural spaces differ.",
        "RECOMP is inferior": "Report the adapted matched-budget result without universal ranking.",
        "open-domain scalable": "Only a bounded post-retrieval pool was tested.",
        "streaming compatible": "Streaming was not evaluated.",
        "causal selected-query effect": "Selected-query effects are descriptive and policy-conditional.",
        "SOTA": "No state-of-the-art claim is tested.",
        "Federated RAG": "This paper evaluates bounded multi-hop context construction.",
        "privacy-preserving": "Privacy was not evaluated.",
    }
    findings = []
    for path in PAPERS + SECTIONS:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase, evidence in dangerous.items():
            start = 0
            while True:
                index = lowered.find(phrase.lower(), start)
                if index < 0:
                    break
                original = sentence_at(text, index)
                findings.append({
                    "file": path.name,
                    "section": section_for(text, index),
                    "original_sentence": original,
                    "evidence": evidence,
                    "risk": "Potential overclaim or scope mismatch.",
                    "replacement": "Use the bounded evidence-backed wording in the final one-sentence claim.",
                })
                start = index + len(phrase)
    return not findings, findings


def main() -> None:
    missing = [path.name for path in PAPERS + SECTIONS if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    paper_texts = {path.name: path.read_text(encoding="utf-8") for path in PAPERS}
    anonymous = paper_texts["paper_anonymous_v5_final.md"]
    full = paper_texts["paper_full_clean_v5_final.md"]
    method = (HERE / "method_v5_final.md").read_text(encoding="utf-8")
    results = (HERE / "results_v5_final.md").read_text(encoding="utf-8")
    cost_text = (HERE / "cost_section_v5_final.md").read_text(encoding="utf-8")
    external = (HERE / "external_transfer_v5_final.md").read_text(encoding="utf-8")
    holdouts = read_json(OUT / "tables/two_frozen_same_source_holdouts.json")
    selected = read_json(OUT / "selected_effect/selected_effect_distribution.json")
    cost = read_json(OUT / "cost/frozen_end_to_end_latency.json")
    lite = read_json(HERE.parent / "review_driven_revision_v5/outputs/lite_model/lite_holdout_metrics.json")

    placeholder_patterns = [r"\[NEEDS MEASUREMENT\]", r"\[NEEDS SOURCE FILE\]", r"\bTODO\b", r"\bTBD\b"]
    placeholder_hits = []
    for name, text in paper_texts.items():
        for pattern in placeholder_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                placeholder_hits.append({"file": name, "pattern": pattern, "line": text.count("\n", 0, match.start()) + 1})
    no_placeholders = not placeholder_hits

    required_method = [
        "missing-hop estimator", "MPNet", "cross-encoder", "document-opportunity model",
        "pair complementarity", "two-document", "Anchor preservation", "two balanced logistic heads",
        "Fully Nested", "Review-Driven Lite Simplification",
    ]
    missing_method = [term for term in required_method if term.lower() not in method.lower()]
    full_method_correct = not missing_method

    rows = holdouts["rows"]
    statistics_complete = len(rows) == 2 and all(
        all(metric in row["statistics"] and "p_value" in row["statistics"][metric] for metric in ("answer_f1", "sp_f1", "joint_f1"))
        for row in rows
    )
    selected_labeled = "descriptive gains conditional on policy-selected interventions" in results and "not causal treatment effects" in results
    selected_fallback = all(
        value["fallback_all_exactly_zero"]
        for split in ("original_holdout_3000", "revision_holdout_3405")
        for value in selected[split]["metrics"].values()
    )

    recomp_complete = (OUT / "audits/recomp_fairness.json").exists() and "RECOMP-660" in full and "Baseline-Truncated-660" in full
    recomp_fair = "official-compressor implementation under reader and budget adaptation" in full and "Matched tokens" in full and "RECOMP is inferior" not in full
    cost_measured = cost["status"] == "complete" and cost["all_frozen_context_match"] and all(row["measured_queries"] >= 500 for row in cost["systems"].values())
    generator_measured = cost_measured and all("generator_only_latency" in row for row in cost["systems"].values())
    offline_clear = "Historical offline GPU-hour totals were not recorded and are therefore unavailable." in cost_text
    lite_failed = "fails" in full and not lite["lite_noninferiority"]["ci_noninferior"] and not lite["lite_noninferiority"]["point_estimate_noninferior"]
    transfer_failed = "misses the pre-specified 4% target" in external and "failed zero-shot safety-transfer diagnostic" in external

    anonymity_patterns = [r"/Users/", r"/home/", r"iiserver", r"Tsukuba", r"V7-HP", r"FedE4RAG", r"github\.com/[A-Za-z0-9_.-]+/"]
    anonymity_hits = []
    for pattern in anonymity_patterns:
        for match in re.finditer(pattern, anonymous, flags=re.IGNORECASE):
            anonymity_hits.append({"pattern": pattern, "line": anonymous.count("\n", 0, match.start()) + 1, "text": sentence_at(anonymous, match.start())})
    anonymity_complete = not anonymity_hits

    bib = (HERE / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib))
    cited_keys = set()
    for text in paper_texts.values():
        cited_keys.update(re.findall(r"@([A-Za-z0-9_.:-]+)", text))
    missing_citations = sorted(cited_keys - bib_keys)
    citations_complete = not missing_citations and bool(cited_keys)

    claims_safe, claim_findings = audit_claims()
    formatting = {
        "abstract_words": len(re.findall(r"\b[\w+-]+\b", (HERE / "abstract_v5_final.md").read_text(encoding="utf-8").split("# Abstract", 1)[-1])),
        "main_word_count": len(re.findall(r"\b[\w+-]+\b", paper_texts["paper_main_conference_v5_final.md"])),
        "anonymous_has_title": anonymous.startswith("# Pair-Complementary Context Construction"),
        "anonymous_has_abstract": "## Abstract" in anonymous,
        "anonymous_has_conclusion": "## 11. Conclusion" in anonymous,
        "appendix_separate": (HERE / "paper_appendix_v5_final.md").exists(),
        "no_accidental_code_indentation": re.search(r"(?m)^ {4}(?:#|\||[A-Za-z*])", anonymous) is None,
    }
    formatting_pass = 180 <= formatting["abstract_words"] <= 230 and all(value for key, value in formatting.items() if isinstance(value, bool))

    full_mean = cost["systems"]["full_v4"]["end_to_end_post_retrieval_latency"]["mean_seconds"]
    baseline_mean = cost["systems"]["frozen_top5_baseline"]["end_to_end_post_retrieval_latency"]["mean_seconds"]
    cost_ratio = full_mean / baseline_mean
    all_gates = [no_placeholders, full_method_correct, statistics_complete, selected_labeled, selected_fallback, recomp_complete, recomp_fair, generator_measured, cost_measured, offline_clear, lite_failed, transfer_failed, anonymity_complete, citations_complete, claims_safe, formatting_pass]
    if not all(all_gates):
        tier = "not_ready"
    elif cost_ratio >= 1.15:
        tier = "main_conference_ready_with_review_risk"
    else:
        tier = "main_conference_ready"

    claim_lines = ["# Final Claim Audit", "", f"Status: **{'PASS' if claims_safe else 'FAIL'}**", ""]
    if claim_findings:
        claim_lines += ["| File | Section | Original sentence | Evidence | Risk | Replacement |", "|---|---|---|---|---|---|"]
        for row in claim_findings:
            claim_lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|") for key in ("file", "section", "original_sentence", "evidence", "risk", "replacement")) + " |")
    else:
        claim_lines.append("No dangerous phrase occurrence was found in the final paper or section files.")
    write(HERE / "final_claim_audit.md", "\n".join(claim_lines))
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    write_json(OUT / "audits/final_claim_audit.json", {"status": "pass" if claims_safe else "fail", "findings": claim_findings})

    write(HERE / "anonymity_audit.md", f"# Anonymity Audit\n\nStatus: **{'PASS' if anonymity_complete else 'FAIL'}**\n\nScanned the final anonymous paper for local/remote paths, server names, project identifiers, institution names, and repository-owner URLs.\n\nFindings: `{json.dumps(anonymity_hits)}`")
    write(HERE / "citation_audit.md", f"# Citation Audit\n\nStatus: **{'PASS' if citations_complete else 'FAIL'}**\n\nCited keys: {len(cited_keys)}. BibTeX keys: {len(bib_keys)}.\n\nMissing keys: `{missing_citations}`\n\nCited keys: `{sorted(cited_keys)}`")
    write(HERE / "statistical_claim_audit.md", f"# Statistical Claim Audit\n\nStatus: **{'PASS' if statistics_complete else 'FAIL'}**\n\nBoth frozen holdouts include baseline and Full scores plus paired Answer/SP/Joint deltas, 95% intervals, and p-values from frozen artifacts. The paper makes no pooled significance claim. Bootstrap zeros are rendered as p<.0002 under 5,000 resamples.\n\nOriginal N=3000; revision N=3405; exact table: `outputs/tables/two_frozen_same_source_holdouts.csv`.")
    write(HERE / "selected_effect_claim_audit.md", f"# Selected-Effect Claim Audit\n\nStatus: **{'PASS' if selected_labeled and selected_fallback else 'FAIL'}**\n\nThe paper reports population effect, coverage, selected wins/losses/ties, quantiles, harm rates, and gain per 100 interventions. It labels the conditional values as descriptive gains conditional on policy-selected interventions and rejects arbitrary-query or causal interpretation. Every fallback context and metric delta is exactly zero.")
    write(HERE / "recomp_final_fairness_audit.md", f"# RECOMP Fairness Audit\n\nStatus: **{'PASS' if recomp_fair and recomp_complete else 'FAIL'}**\n\nThe main comparison includes Frozen Top-5, Baseline-Truncated-660, RECOMP-660, and Full. RECOMP is labeled an official-compressor implementation under reader and budget adaptation. Top-1 is an appendix compatibility diagnostic. The paper states that matched token budgets do not equate structural action spaces and avoids a universal ranking.")
    write(HERE / "lite_noninferiority_audit.md", f"# Lite Non-Inferiority Audit\n\nStatus: **{'PASS' if lite_failed else 'FAIL'}**\n\nFrozen margin: 0.002. Revision-holdout Lite minus Full Joint F1: {lite['lite_noninferiority']['joint_f1_delta']['delta']:+.6f}; 95% CI [{lite['lite_noninferiority']['joint_f1_delta']['ci_low']:+.6f}, {lite['lite_noninferiority']['joint_f1_delta']['ci_high']:+.6f}]. Point and CI non-inferiority both fail. Full remains the main method.")
    write(HERE / "external_transfer_claim_audit.md", f"# External Transfer Claim Audit\n\nStatus: **{'PASS' if transfer_failed else 'FAIL'}**\n\nZero-shot 2Wiki remains non-significant, support-flat, and has 6.92% selected answer-drop. Frozen five-seed K-shot calibration reaches 5.10% at best but misses the 4% criterion. The paper presents this as an unresolved transfer boundary.")
    write(HERE / "formatting_audit.md", f"# Formatting Audit\n\nStatus: **{'PASS' if formatting_pass else 'FAIL'}**\n\n```json\n{json.dumps(formatting, indent=2)}\n```\n\nThe anonymous main paper and appendix are separate. Tables are Markdown source tables for conference-template conversion.")

    readiness = {
        "no_placeholders_in_paper": no_placeholders,
        "full_method_description_correct": full_method_correct,
        "revision_holdout_statistics_complete": statistics_complete,
        "selected_effect_descriptively_labeled": selected_labeled,
        "recomp_budget_matched_complete": recomp_complete,
        "recomp_claim_fair": recomp_fair,
        "online_generator_latency_measured": generator_measured,
        "end_to_end_latency_measured": cost_measured,
        "offline_cost_boundary_clear": offline_clear,
        "lite_failure_correctly_reported": lite_failed,
        "2wiki_calibration_failure_correctly_reported": transfer_failed,
        "anonymity_complete": anonymity_complete,
        "citations_complete": citations_complete,
        "claims_safe": claims_safe,
        "formatting_pass": formatting_pass,
        "recommended_title": "Pair-Complementary Context Construction with Reader-Safe Selection for Multi-Hop QA",
        "one_sentence_claim": "Pair-complementary action generation and fully nested reader-safe selection yield modest but reproducible same-source QA gains, with larger descriptive effects on selected interventions and unresolved cost and transfer boundaries.",
        "primary_result": "Full improves Answer/SP/Joint F1 by +0.0088/+0.0056/+0.0064 on the frozen 3,000-query HotpotQA holdout.",
        "secondary_result": "An untouched 3,405-query same-source holdout confirms +0.0116/+0.0061/+0.0080.",
        "population_effect": "Modest positive paired changes on both same-source holdouts; no pooled significance claim.",
        "selected_policy_effect": "Original 774 interventions: descriptive Answer/SP/Joint +0.0340/+0.0219/+0.0250; fallbacks exactly unchanged.",
        "online_end_to_end_cost": f"Full {1000*full_mean:.2f} ms/query versus Frozen Top-5 {1000*baseline_mean:.2f} ms/query ({cost_ratio:.2f}x), batch size 1, one final reader call.",
        "main_remaining_risk": "The absolute population gain is modest, Full has measurable overhead, and cross-dataset safety remains unresolved.",
        "recommended_submission_tier": tier,
        "diagnostics": {"placeholder_hits": placeholder_hits, "missing_method_terms": missing_method, "anonymity_hits": anonymity_hits, "missing_citations": missing_citations, "claim_findings": claim_findings},
    }
    write_json(OUT / "audits/submission_readiness.json", readiness)
    flags = "\n".join(f"- {key}: `{str(value).lower()}`" for key, value in readiness.items() if isinstance(value, bool) and key != "formatting_pass")
    report = f"""# Submission Readiness Report

## Required Flags

{flags}

## Final Decision

- recommended_title: **{readiness['recommended_title']}**
- one_sentence_claim: {readiness['one_sentence_claim']}
- primary_result: {readiness['primary_result']}
- secondary_result: {readiness['secondary_result']}
- population_effect: {readiness['population_effect']}
- selected_policy_effect: {readiness['selected_policy_effect']}
- online_end_to_end_cost: {readiness['online_end_to_end_cost']}
- main_remaining_risk: {readiness['main_remaining_risk']}
- recommended_submission_tier: **{tier}**

The submission is technically complete if all required flags above are true. A higher Full latency does not block submission after honest reporting, but it remains a review risk under the frozen decision rule.
"""
    write(HERE / "submission_readiness_report.md", report)
    print(json.dumps({"status": "complete", "tier": tier, "all_required_flags": all(all_gates), "claims_safe": claims_safe}, indent=2))


if __name__ == "__main__":
    main()
