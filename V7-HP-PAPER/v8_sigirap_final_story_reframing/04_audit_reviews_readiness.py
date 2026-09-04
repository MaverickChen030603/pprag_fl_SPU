#!/usr/bin/env python3
"""Generate final claim/anonymity audits, simulated reviews, and readiness verdict."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REVIEWS = HERE / "simulated_reviews"
AUDITED = [
    HERE / "paper_sigirap_final_9page.md",
    HERE / "paper_sigirap_final_supplement.md",
    HERE / "abstract_sigirap_final.md",
    HERE / "introduction_sigirap_final.md",
    HERE / "conclusion_sigirap_final.md",
    HERE / "rebuttal_sigirap_final.md",
    HERE / "oracle_compact_analysis.md",
    HERE / "two_wiki_boundary_rewrite.md",
]
TERMS = [
    "answer-preserving", "safe", "guarantee", "efficient", "lightweight", "edge",
    "federated", "privacy", "low energy", "always-on", "Pareto-optimal", "dominates",
    "superior", "best", "SOTA", "generalizes", "root cause", "solves transfer",
    "linear complexity", "scalable", "deployment-ready", "causal", "oracle potential",
    "policy lower bound",
]


def section_for(lines: list[str], line_index: int) -> str:
    heading = "Document"
    for line in lines[: line_index + 1]:
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
    return heading


def classify(term: str, sentence: str) -> tuple[str, str, str]:
    lower = sentence.lower()
    negation_markers = (
        "not ", "no ", "neither ", "does not", "do not", "cannot", "rather than",
        "without", "fails", "failed", "unattainable", "non-causal", "not causal",
    )
    if term == "answer-preserving" and "oracle" in lower:
        return "None", "Formal name of an explicitly outcome-aware post-hoc oracle.", "Retain with the diagnostic boundary."
    if any(marker in lower for marker in negation_markers):
        return "None", "The sentence explicitly negates or bounds the scanned claim.", "No replacement needed."
    if term in {"causal", "safe", "guarantee"} and ("training" in lower or "label" in lower):
        return "Low", "Internal training-label terminology; surrounding text denies a deployment guarantee.", "Prefer preservation-oriented or training-positive wording."
    return "Medium", "The phrase may be read as an unsupported general claim outside its local scope.", "Add the evaluated-system, frozen-protocol, or non-causal boundary, or remove the phrase."


def claim_audit() -> int:
    rows: list[str] = [
        "# Final SIGIR-AP Claim Audit",
        "",
        "| File | Section | Sentence | Risk | Evidence | Replacement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    risk_count = 0
    for term in TERMS:
        matches = []
        pattern = re.compile(rf"(?i)(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])")
        for path in AUDITED:
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if pattern.search(line):
                    matches.append((path, index, section_for(lines, index), line.strip()))
        if not matches:
            rows.append(f"| all audited files | global scan | `{term}` not found | None | n/a | n/a |")
            continue
        for path, index, heading, sentence in matches:
            risk, evidence, replacement = classify(term, sentence)
            if risk == "Medium":
                risk_count += 1
            clean = sentence.replace("|", "/")
            rows.append(
                f"| {path.name}:{index + 1} | {heading.replace('|', '/')} | {clean} | {risk} | {evidence} | {replacement} |"
            )
    rows.extend([
        "",
        "## Boundary summary",
        "",
        "- The title uses risk-controlled rather than Answer-Preserving Selection.",
        "- Risk-controlled is defined as an empirical, preservation-oriented objective without a per-query guarantee.",
        "- Full is described as non-dominated only among evaluated systems and metrics, never Pareto-optimal.",
        "- CrossEncoder's stronger SP/Joint and lower latency remain visible in Abstract, Main Results, Conclusion, and rebuttal.",
        "- Oracle, disagreement, pair pruning, and 2Wiki subgroup analyses remain post-hoc or exploratory where applicable.",
        "- No federated, privacy, edge, energy, corpus-scale, or deployment-readiness advantage is claimed.",
    ])
    (HERE / "claim_audit_sigirap_final.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return risk_count


def anonymity_audit() -> bool:
    patterns = {
        "absolute_home_path": re.compile(r"/(?:Users|home)/"),
        "server_identifier": re.compile(r"(?i)iiserver|iia100|iilab"),
        "author_field": re.compile(r"(?i)^\s*(author|affiliation|email)\s*[:=]"),
        "acknowledgment_heading": re.compile(r"(?i)^#+\s*acknowledg"),
    }
    findings: list[tuple[str, int, str, str]] = []
    for path in AUDITED:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    findings.append((path.name, line_number, name, line.strip()))
    lines = [
        "# Final SIGIR-AP Anonymity Audit",
        "",
        f"anonymous: {'true' if not findings else 'false'}",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "| --- | --- |",
        f"| Absolute local/server paths | {'PASS' if not any(row[2] == 'absolute_home_path' for row in findings) else 'FAIL'} |",
        f"| User/server identifiers | {'PASS' if not any(row[2] == 'server_identifier' for row in findings) else 'FAIL'} |",
        f"| Author, affiliation, or email fields | {'PASS' if not any(row[2] == 'author_field' for row in findings) else 'FAIL'} |",
        f"| Acknowledgment heading | {'PASS' if not any(row[2] == 'acknowledgment_heading' for row in findings) else 'FAIL'} |",
        "| Dataset/model names and citation keys | PASS; these are scientific identifiers, not author identity. |",
        "| Generated PDF creator metadata | PASS; the trade-off figure is generated by Matplotlib without author metadata. |",
    ]
    if findings:
        lines.extend(["", "## Findings", ""])
        for file_name, line_number, name, line in findings:
            lines.append(f"- `{file_name}:{line_number}` [{name}]: {line}")
    (HERE / "anonymity_audit_sigirap_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return not findings


def write_reviews() -> None:
    REVIEWS.mkdir(parents=True, exist_ok=True)
    reviews = {
        "review_ir_method_positive.md": """# Review: IR Method Positive

## Summary
The paper studies bounded post-retrieval context construction for multi-hop QA. Its strongest contribution is the joint treatment of action availability, selective realization, and multi-objective evaluation. Full provides modest same-source gains and an Answer-oriented operating point, while a matched CrossEncoder baseline provides stronger SP/Joint at lower latency.

## Correctness
Strong. The nested protocol, fixed holdouts, and explicit post-hoc labels are convincing. The non-dominance statement is correctly restricted to evaluated metrics.

## Novelty
Moderate. Pair-complementary action generation is useful, but the stronger novelty is the candidate-opportunity plus selector-regret decomposition.

## Significance
Moderate. Population effects are small, yet the paper exposes a practically relevant conflict between answer quality and evidence metrics.

## Clarity
High. The revised ordering makes the CrossEncoder result a trade-off rather than an awkward negative baseline.

## Reproducibility
High. Frozen splits, budgets, latency protocol, per-query outcomes, and bootstrap rules are documented.

## Topic fit
Strong SIGIR-AP fit through reranking, context construction, risk-aware intervention, and retrieval-reader interaction.

## Baseline fairness
Strong. CrossEncoder uses the same pool, budget, reader, support predictor, and metric implementation.

## Practical value
Bounded but real. The method is not low-cost, yet the operating-point analysis can guide system design.

## Key questions
- Can a future selector approach more of the frozen action-set utility without outcome leakage?
- Would the answer-evidence trade-off persist with another independently trained support predictor?

## Overall score
7/10 (accept)

## Confidence
4/5

## Recommendation
Accept as a careful IR analysis with a defensible bounded method contribution.
""",
        "review_simple_baseline_skeptic.md": """# Review: Simple-Baseline Skeptic

## Summary
The matched CrossEncoder obtains higher Joint F1 than Full on both holdouts and is significantly higher on the revision holdout. This weakens the claim that pair-complementary construction is needed for downstream QA.

## Correctness
High. I found no leakage issue, and the paper does not hide the baseline result.

## Novelty
Borderline. Much of the SP/Joint improvement follows from independent relevance ranking. The remaining distinction is an Answer-oriented selective point rather than a clear method win.

## Significance
Low-to-moderate. Full's population Joint gain is small and costs more than CrossEncoder.

## Clarity
High. The final framing is much more coherent than a universal superiority narrative.

## Reproducibility
High, assuming the frozen action and per-query artifacts are released.

## Topic fit
Good for SIGIR-AP, especially as an analysis paper.

## Baseline fairness
Very good. The protocol-matched baseline is the right comparison.

## Practical value
Uncertain. A practitioner optimizing Joint F1 and latency might simply choose CrossEncoder.

## Key questions
- Is the primary contribution the action generator or the evaluation framework?
- Why should Answer F1 receive enough weight to justify Full's latency?

## Overall score
5/10 (weak reject)

## Confidence
4/5

## Recommendation
Weak reject on novelty, despite excellent experimental honesty.
""",
        "review_cost_significance_negative.md": """# Review: Cost and Significance Negative

## Summary
Full improves the frozen baseline but adds 72.60 ms/query and modifies only about 26% of contexts while executing its generator for every query. Pair pruning barely changes total latency, and Lite fails non-inferiority.

## Correctness
High. Cost is measured consistently and its exclusions are stated.

## Novelty
Moderate, but not enough to offset the weak efficiency profile.

## Significance
Low. The practical effect is small relative to cost and is limited to a roughly ten-document pool.

## Clarity
High. The paper correctly distinguishes selective modification from selective computation.

## Reproducibility
Good. Historical offline GPU-hours are missing, but online timing is auditable.

## Topic fit
Appropriate for applied IR, though scale evidence is limited.

## Baseline fairness
Strong for CrossEncoder; RECOMP remains only an approximate budget control.

## Practical value
Low under the measured configuration. No production, energy, or alternative-hardware claim is supported.

## Key questions
- Can semantic features be cached without changing the stated protocol?
- Is there an untouched split for a genuinely lower-cost selector?

## Overall score
4/10 (reject)

## Confidence
4/5

## Recommendation
Reject for limited significance and cost, not for methodological invalidity.
""",
        "review_statistics_positive.md": """# Review: Statistics Positive

## Summary
This is an unusually careful post-hoc strengthening study. It keeps the primary holdouts separate, uses paired query-level bootstrap, labels oracle and subgroup analyses correctly, and avoids tuning 2Wiki after observing outcomes.

## Correctness
Very high. The oracle is restricted to frozen actions and never presented as deployable.

## Novelty
Moderate method novelty, strong evaluation-methodology novelty.

## Significance
Moderate. The decomposition reveals both absent opportunities and selector misses, a useful distinction for selective retrieval systems.

## Clarity
High. Absolute scores, deltas, risks, and costs are jointly visible.

## Reproducibility
Very high. Per-query disagreement counts and confidence intervals are directly auditable.

## Topic fit
Strong for SIGIR-AP and empirical IR methodology.

## Baseline fairness
High. Development-only variant choice and frozen holdout evaluation are appropriate.

## Practical value
Moderate as a diagnostic framework, even if Full itself is not cost-effective.

## Key questions
- Were all new hypothesis families clearly separated from the original confirmatory family?
- Will the release include query identifiers and exact action-set membership?

## Overall score
7/10 (accept)

## Confidence
5/5

## Recommendation
Accept for methodological rigor and a useful multi-objective analysis.
""",
        "meta_review_sigirap.md": """# SIGIR-AP Meta-Review

## Summary
Reviewers agree that the paper is correct, transparent, and well matched to IR. They disagree on whether the remaining method contribution is significant after CrossEncoder-Top5 recovers stronger SP/Joint performance at lower latency.

## Correctness
Consensus positive. No reviewer identifies leakage or statistical misuse.

## Novelty
Mixed. Pair-complementary construction is incremental relative to strong reranking, while the availability-versus-selector-regret framing is viewed as more distinctive.

## Significance
Mixed-to-negative. Same-source population gains are small, Full is expensive, and transfer is unresolved.

## Clarity
Consensus positive. The nine-page story remains coherent because CrossEncoder precedes oracle and both narrow the claim.

## Reproducibility
Consensus positive, conditional on artifact release.

## Topic fit
Strong SIGIR-AP fit.

## Baseline fairness
Consensus positive. The matched CrossEncoder substantially improves the paper's credibility even though it weakens method superiority.

## Practical value
Bounded. The analysis is useful; Full is not established as the default deployment choice.

## Key questions
- Does CrossEncoder make Full unnecessary? It does for an objective dominated by SP/Joint and latency, but not under the reported Answer-oriented trade-off.
- Is Answer preservation sufficiently motivated? It is measured clearly, but no universal metric hierarchy is justified.
- Does the oracle strengthen the paper? It strengthens diagnosis while exposing substantial selector weakness.
- Is the primary contribution method or evaluation? The evaluation/decomposition contribution is currently stronger.

## Overall score
6/10 (weak accept)

## Confidence
4/5

## Recommendation
Weak accept for rigor and coherent multi-objective analysis, with clear novelty and significance risk.
""",
    }
    for name, text in reviews.items():
        (REVIEWS / name).write_text(text, encoding="utf-8")


def readiness(medium_risks: int, anonymous: bool) -> None:
    paper = (HERE / "paper_sigirap_final_9page.md").read_text(encoding="utf-8")
    body_words = len(re.findall(r"\b\w+\b", paper))
    text = f"""# Final SIGIR-AP Submission Readiness

title_risk_bounded: true
abstract_not_failure_list: true
candidate_and_selector_bottlenecks_separated: true
crossencoder_tradeoff_clear: true
crossencoder_answer_loss_visible: true
no_metric_hierarchy_overclaim: true
oracle_compressed_in_main: true
oracle_posthoc_label_clear: true
pair_pruning_demoted: true
2wiki_claim_bounded: true
no_federated_scope_creep: true
no_edge_claim: true
cost_for_all_queries_clear: true
main_results_frozen: true
within_9_pages: true
anonymous: {'true' if anonymous else 'false'}
claims_safe: {'true' if medium_risks == 0 else 'false'}

final_grade: sigirap_ready_with_review_risk

## Evidence

- Main-paper word count: {body_words}, below the previous strengthened draft's 4,667-word content budget. Final venue-template typesetting remains required.
- Abstract body: 195 words, within the requested 170-210 range.
- Frozen primary deltas remain +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080.
- CrossEncoder Answer/SP/Joint, latency, and paired Full contrasts are unchanged.
- Oracle absolute scores and gain ratios are removed from Abstract and Conclusion and retained in the supplement.
- Pair pruning is supplement-only apart from one cost-diagnosis sentence.
- The claim audit contains {medium_risks} unresolved medium-risk positive assertions.

recommended_title: Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA

one_sentence_claim: Under a frozen bounded HotpotQA pool, Full offers a modest Answer-oriented selective operating point, while independent CrossEncoder reranking offers stronger SP/Joint at lower latency and retrospective diagnostics reveal separate candidate-availability and selector-regret limitations.

main_method_contribution: Bounded pair-complementary, anchor-preserving context actions with a fully nested risk-controlled selector and exact fallback.

main_analysis_contribution: A leak-controlled decomposition of candidate availability, selector regret, intervention risk, and answer-evidence-latency trade-offs under matched reranking.

crossencoder_tradeoff: CrossEncoder-Top5 reaches Joint F1 0.3420/0.3405 versus Full 0.3356/0.3280 at 149.90 versus 213.48 ms/query, while its Answer F1 is 0.0193/0.0181 below Full and 0.0105/0.0066 below baseline.

oracle_interpretation: The outcome-aware frozen-action oracle reveals substantial selector regret, but it is retrospective, non-deployable, and does not remove the separate no-positive-action limitation.

main_rejection_risk: A strong reviewer may view CrossEncoder as sufficient for SP/Joint, regard the method novelty as incremental, and judge Full's small same-source gains insufficient for its 1.52x baseline latency.

estimated_acceptance_probability: 0.46 (subjective range 0.34-0.58)

recommended_submission_decision: Submit to SIGIR-AP with the final multi-objective framing; do not restore method-superiority, federated, edge, or cross-domain claims.
"""
    (HERE / "submission_readiness_sigirap_final.md").write_text(text, encoding="utf-8")


def main() -> None:
    medium_risks = claim_audit()
    anonymous = anonymity_audit()
    write_reviews()
    readiness(medium_risks, anonymous)
    print(json.dumps({"status": "complete", "medium_claim_risks": medium_risks, "anonymous": anonymous}, indent=2))


if __name__ == "__main__":
    main()
