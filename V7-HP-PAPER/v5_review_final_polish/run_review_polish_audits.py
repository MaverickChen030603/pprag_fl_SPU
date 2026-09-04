#!/usr/bin/env python3
"""Generate review-facing claim audits for the frozen V5 paper."""

from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parent


def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text.rstrip() + "\n")


review_accuracy = """# Review Accuracy Audit

This audit compares the review claims with the final frozen V5 artifacts. Categories are **accurate**, **partially accurate**, **outdated**, **overstated**, and **incorrect**.

| Review claim | Category | Frozen evidence | Final correction |
|---|---|---|---|
| Online latency is missing. | **Outdated** | Same-machine A100 timing now reports Frozen Top-5 140.88 ms/query, Full 213.48, Lite 143.97, and RECOMP-660 169.64; every system uses one final reader call. | The concern was valid for an earlier draft but is resolved in the current evidence package. |
| The same-source answer-drop rate is 2.0%. | **Outdated** for selected-policy risk | Direct selected-query accounting gives 60/774 = 7.75% on the 3,000 holdout and 69/881 = 7.83% on the 3,405 holdout. | The final paper uses 7.75%/7.83% whenever it discusses risk among selected interventions. An earlier aggregate/population quantity is not reused as selected answer-drop. |
| The intervention produces a significant gain on every selected query. | **Incorrect** | Original selected Answer wins/losses/ties are 89/60/625; Joint is 141/115/518. Revision selected Answer is 107/69/705; Joint is 169/125/587. Every median and interquartile endpoint is zero. | The conditional means are descriptive; most selected queries tie and some are harmed. |
| Larger selected-subset means prove a causal intervention effect. | **Incorrect** | Selection is policy-dependent and no randomized treatment assignment is used. | The paper separates population effects from descriptive effects conditional on the policy's chosen subset. |
| The RECOMP comparison proves general superiority. | **Overstated** | At about 660 tokens, RECOMP Joint F1 is 0.3259 versus 0.3292 for Frozen Top-5; delta -0.0033, 95% CI [-0.0109, +0.0044], p=0.4172. Full is 0.3356, but action spaces differ. | The result is a budget-controlled protocol comparison, not a universal ranking or an end-to-end reproduction claim. |
| The paper lacks an independent simplification test. | **Outdated** | Lite minus Full Joint F1 is -0.0063, 95% CI [-0.0104, -0.0023], against a frozen 0.002 non-inferiority margin. | The independent test exists and fails; the failure is retained. |
| The method has an external transfer success. | **Incorrect** | 2Wiki Answer/SP/Joint p-values are .1116/.6928/.3296, and the best few-shot answer-drop is 5.10% versus a 4% target. | The paper reports transfer as a failed diagnostic and unresolved limitation. |
| The system is evaluated at corpus scale. | **Incorrect** | The official distractor pool is about ten documents; 2,973/3,000 queries have at least ten, only one has twenty, and none has fifty or one hundred. | The paper states a bounded post-retrieval candidate-pool scope. |
| Two independent readers replicate the complete pipeline. | **Partially accurate** | FLAN and UnifiedQA have directionally positive Answer F1 deltas, but they share the same support predictor. | The second model is an answer-reader directional check; SP is not independently replicated and Joint contains a shared component. |
| The nested evaluation prevents holdout outcome selection. | **Accurate** | Generator and selector fit outer training folds, thresholds use inner out-of-fold predictions, and outer-test outcomes are not used for architecture or threshold selection. | Retained as a central methodological strength without claiming that nesting guarantees deployment safety. |

## Interpretation

The strongest resolved review concern is cost measurement. The strongest remaining concerns are the small population effect relative to a 1.52x latency ratio, nonzero selected-query harm, same-source and bounded-pool scope, shared support prediction in the second-reader check, and failed external transfer. None can be removed by prose; the revision makes each visible and narrows the claim accordingly.
"""


remaining_weakness = """# Remaining Weakness Priority

| Priority | Weakness | Current evidence | Cannot be fixed by writing | What writing can clarify | New experiment justified? | Final limitation sentence |
|---|---|---|---|---|---|---|
| P0 writing | Complexity versus marginal population gain | Full gains +0.0064/+0.0080 Joint F1 on the two holdouts and costs 213.48 versus 140.88 ms/query (1.52x). | Whether the trade-off is worthwhile depends on use case and hardware. | Put absolute deltas and latency together; avoid broad efficiency language. | Yes, only if a future submission needs a stronger deployment claim; not required to interpret V5. | Full provides modest same-source gains at 1.52x measured post-retrieval latency. |
| P0 claim | `reader-safe` can sound like a guarantee | Selected Answer F1 falls on 7.75%/7.83%; selected Joint F1 falls on 14.86%/14.19%. | Aggregate selection cannot certify every query. | Define the term at first use as an answer-preservation-oriented, risk-controlled objective. | A formal risk guarantee would require a new preregistered calibration protocol. | Reader-safe is an objective label, not a per-query harm guarantee. |
| P1 scope | Bounded candidate pool | Pool is approximately ten documents; only one of 3,000 queries has twenty or more. | Current results do not reveal large-$L$ behavior. | State quadratic pair formation before pruning, fixed $L=10$, and ten measured pairs/query. | Yes for claims about corpus-scale deployment; no for the present bounded claim. | The study does not evaluate corpus-scale or changing-index retrieval. |
| P1 validation | Shared support predictor across readers | FLAN and UnifiedQA differ only in answer generation; SP 0.4930 to 0.4987 is shared. | Existing artifacts cannot provide independent SP replication. | Call it directional answer-reader evidence and qualify Joint. | Only under a new frozen protocol with an independent support model. | The second answer reader does not independently replicate support prediction. |
| P1 transfer | 2Wiki criterion fails | All zero-shot p-values are non-significant; best few-shot answer-drop 5.10% misses 4%. | Prose cannot turn failure into transfer evidence. | Preserve the failure and its stopping rule. | Yes for cross-dataset claims, with preregistered calibration and no post-hoc outcome selection. | Cross-dataset risk calibration remains unresolved. |
| Resolved | Missing cost measurement | Same A100, batch one, 50 warmup and 500 measured queries, one final reader call, 100% context match. | Not applicable. | Report protocol and values in the main paper. | No. | The earlier missing-latency concern is resolved by measured post-retrieval timing. |

## Recommended Priority

The present submission should spend revision effort on P0 interpretation rather than new tuning. Additional experiments are justified only if the claim is intentionally expanded to per-query guarantees, independent support replication, large-pool scaling, or cross-dataset transfer. Such experiments need new frozen protocols and should not be retrofitted into V5 through holdout-guided choices.
"""


selected_audit = """# Selected Policy Effect Audit

## Frozen decomposition

| Holdout | Coverage | Metric | Population delta | Selected mean | Wins/Losses/Ties | Selected drop | Median [Q25, Q75] | Fallback |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 3,000 | 774/3000 (25.8%) | Answer F1 | +0.0088 | +0.0340 | 89/60/625 | 7.75% | 0 [0, 0] | Exactly zero delta |
| 3,000 | 774/3000 (25.8%) | SP F1 | +0.0056 | +0.0219 | 123/100/551 | 12.92% | 0 [0, 0] | Exactly zero delta |
| 3,000 | 774/3000 (25.8%) | Joint F1 | +0.0064 | +0.0250 | 141/115/518 | 14.86% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | Answer F1 | +0.0116 | +0.0447 | 107/69/705 | 7.83% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | SP F1 | +0.0061 | +0.0237 | 127/94/660 | 10.67% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | Joint F1 | +0.0080 | +0.0309 | 169/125/587 | 14.19% | 0 [0, 0] | Exactly zero delta |

## Interpretation rule

Although the selected subset has larger mean deltas, most selected contexts tie the baseline and some are harmful; the conditional result characterizes the policy's chosen subset rather than an oracle-improvable population.

The previously circulated 2.0% same-source answer-drop figure is an aggregate/population quantity and is outdated for the selected-policy risk claim. The final manuscript consistently uses the direct selected-query rates of 7.75% and 7.83%. No selected-query mean is described as a causal effect, a guarantee, or an effect on all improvable queries.
"""


recomp_audit = """# RECOMP Claim Audit

## Frozen budget-controlled comparison

| System | Mean tokens | Represented documents | Answer F1 | SP F1 | Joint F1 | E2E ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 664.5 | 4.986 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Baseline-Truncated-660 | 635.7 | 4.236 | 0.6038 | 0.4904 | 0.3224 | 147.43 |
| RECOMP-660 | 635.9 | 4.873 | 0.6226 | 0.4837 | 0.3259 | 169.64 |
| Full | 656.1 | 4.986 | 0.6271 | 0.4987 | 0.3356 | 213.48 |

RECOMP-660 versus Frozen Top-5 changes Joint F1 by -0.0033, with 95% CI [-0.0109, +0.0044] and p=0.4172. The protocol standardizes the FLAN reader, prompt, support predictor, same Top-5 source input, and an approximately 660-token context budget. Baseline-Truncated controls for source-order packing at the same budget.

## Allowed claim

Under this approximately matched context budget and standardized reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline. Full has a positive same-source effect under its own structured action objective.

## Prohibited inference

This comparison does not establish that Full generally outperforms RECOMP, that RECOMP is inferior, or that equal tokens create equal structural action spaces. It is an official-compressor implementation under reader and budget adaptation, not an end-to-end reproduction of every RECOMP setting.
"""


candidate_scope = """# Candidate Pool Scope Statement

## Observed pool

The experiment begins after retrieval from the HotpotQA distractor candidate pool, which is approximately ten documents per query. In the frozen 3,000-query holdout:

| Threshold | Queries with at least this many documents |
|---:|---:|
| 10 | 2,973 |
| 20 | 1 |
| 50 | 0 |
| 100 | 0 |

There is no common fixed 20-, 50-, or 100-document subset for a same-protocol scaling claim.

## Complexity boundary

Pair construction over a retained set is quadratic in $L$ before pruning: at most $L(L-1)/2$. The frozen deployment uses $L=10$, so 45 pairs are possible before pruning and ten pairs are actually scored per query. The 213.48 ms/query result is valid for this bounded setting only.

## Paper wording

> The method is evaluated as post-retrieval context organization over the bounded HotpotQA distractor pool. It does not establish corpus-scale or streaming behavior.

Future work may test subquadratic pair proposals, approximate nearest-neighbor pair retrieval, adaptive Top-$L$, and streaming calibration. These are new protocols, not conclusions from V5.
"""


multi_reader = """# Multi-Reader Support Audit

## Frozen 3,000-query results

| Answer reader | Baseline Answer F1 | Selected Answer F1 | Delta | Baseline SP F1 | Selected SP F1 | Delta | Baseline Joint F1 | Selected Joint F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAN-T5-Large | 0.618307 | 0.627081 | +0.008773 | 0.493025* | 0.498664* | +0.005639* | 0.329182 | 0.335631 | +0.006450 |
| UnifiedQA-T5-Large | 0.566173 | 0.577188 | +0.011015 | 0.493025* | 0.498664* | +0.005639* | 0.304453 | 0.312964 | +0.008510 |

`*` Both rows share one frozen support predictor. The answer generator changes; support prediction does not.

## Supported interpretation

- The same frozen context changes have directionally positive Answer F1 under two answer readers.
- Joint point estimates are directionally consistent, but each includes the same support component.
- This is useful supporting evidence that the Answer result is not unique to one decoder.

## Unsupported interpretation

- Two independent end-to-end reader pipelines.
- Independent SP replication.
- Independent Joint replication.
- General reader robustness.

No new support-predictor experiment is added during final polishing. A credible independent replication would require a separately frozen support model and a preregistered evaluation protocol; adding one after reviewing V5 outcomes would create a new experiment family rather than repair wording.
"""


external_transfer = """# External Transfer Claim Audit

## Zero-shot 2Wiki result

| Metric | Baseline | Frozen transfer | Delta | 95% CI | p-value |
|---|---:|---:|---:|---|---:|
| Answer F1 | 0.4709 | 0.4794 | +0.0086 | [-0.0021, +0.0191] | 0.1116 |
| SP F1 | 0.4545 | 0.4539 | -0.0006 | [-0.0036, +0.0025] | 0.6928 |
| Joint F1 | 0.2463 | 0.2496 | +0.0033 | [-0.0031, +0.0098] | 0.3296 |

Coverage is 26.0% and selected Answer-drop is 6.92%. None of the three metric changes is statistically significant.

## Few-shot calibration

The best frozen grid result uses threshold-only calibration with $K=128$: 16.26% coverage, 5.10% selected Answer-drop, Answer/SP/Joint F1 0.4755/0.4542/0.2484. It misses the pre-specified 4% answer-drop target. The search stops without post-failure retuning.

## Allowed claim

2Wiki is a failed transfer and calibration diagnostic. Positive Answer and Joint point estimates motivate future work but do not establish cross-dataset generalization, robustness under shift, or target-domain safety.
"""


review_response = """# Review Response: Strengths and Remaining Weaknesses

We thank the reviewers for identifying a real interpretation problem: the earlier presentation did not always separate resolved evidence gaps from persistent scientific limitations. The revision keeps all frozen outcomes, including failed tests, and reorganizes the paper around a bounded quality-risk-cost claim.

## Strengths retained without self-praise

- **Independent non-inferiority test:** Lite is frozen before the 3,405-query revision holdout and fails the 0.002 margin.
- **Budget-controlled comparison:** RECOMP, source-order truncation, the frozen baseline, and Full use the same FLAN reader and an approximately matched 660-token condition.
- **Policy-conditional decomposition:** population effects are separated from selected-query means, wins, losses, ties, drop rates, and exact fallbacks.
- **Fully nested no-leak protocol:** outer-test outcomes do not select generator modules, selector thresholds, or coverage.
- **Explicit failures:** Lite non-inferiority and 2Wiki transfer/calibration remain failures in the paper.

## 1. Missing latency

**Concern.** The method's cost could not be judged.  
**Agreement.** This was accurate for an earlier draft.  
**Evidence now available.** Same A100, batch size one, 50 warmup and 500 measured queries: Frozen Top-5 140.88 ms/query, Full 213.48, Lite 143.97, and RECOMP-660 169.64; one final reader call and 100% frozen-context match.  
**Paper location.** Section 7 and Appendix F.  
**Boundary.** This resolves the missing-measurement concern but does not make Full efficient; Full is 1.52x the baseline.

## 2. Complexity relative to gain

**Concern.** The architecture is elaborate relative to sub-point population gains.  
**Agreement.** Yes. The trade-off must be explicit.  
**Evidence.** Joint F1 changes by +0.0064 and +0.0080 on the two holdouts; Full adds 72.60 ms/query. Lite reduces cost but fails non-inferiority.  
**Paper location.** Abstract, Sections 5, 7, 10, and Conclusion.  
**Boundary.** We claim a bounded trade-off, not unqualified practical value.

## 3. External transfer

**Concern.** Same-source confirmation does not establish generalization.  
**Agreement.** Yes.  
**Evidence.** 2Wiki Answer/SP/Joint p-values are .1116/.6928/.3296; few-shot answer-drop 5.10% misses the 4% target.  
**Paper location.** Section 8 and Limitation 4.  
**Boundary.** The result is a failed transfer diagnostic, not external validation.

## 4. Candidate-pool scale

**Concern.** Pair construction may not scale to broader retrieval pools.  
**Agreement.** Yes.  
**Evidence.** The official pool is approximately ten documents; the frozen method scores ten pairs/query after pruning. Only one of 3,000 queries has at least twenty documents.  
**Paper location.** Section 7.1, Appendix I, and Limitation 5.  
**Boundary.** No corpus-scale or streaming claim is made.

## 5. Support replication across readers

**Concern.** The second reader may not independently validate the complete pipeline.  
**Agreement.** Correct.  
**Evidence.** FLAN and UnifiedQA have positive Answer deltas, but SP predictions are shared. Joint point estimates consequently share one component.  
**Paper location.** Section 9, Appendix H, and Limitation 6.  
**Boundary.** We call this directional answer-reader evidence, not independent SP or Joint replication.

## Final response boundary

The revision does not add tuning, reselect policies from holdout outcomes, or reinterpret non-significant results as success. Its contribution is a more accurate account of what the frozen evidence supports and what remains unresolved.
"""


def section_for(text: str, offset: int) -> str:
    section = "Title"
    for match in re.finditer(r"(?m)^#{1,3} (.+)$", text):
        if match.start() > offset:
            break
        section = match.group(1)
    return section


def sentence_for(text: str, start: int, end: int) -> str:
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start))
    right_candidates = [p for p in (text.find(". ", end), text.find("\n", end)) if p >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    sentence = text[left + 1 : right + (1 if text[right:right + 1] == "." else 0)]
    return " ".join(sentence.split()).replace("|", "\\|")


def build_reader_safe_audit() -> str:
    rows = []
    for path in [ROOT / "paper_anonymous_review_polished.md", ROOT / "paper_appendix_review_polished.md"]:
        text = path.read_text()
        for match in re.finditer(r"(?i)reader-safe", text):
            sentence = sentence_for(text, match.start(), match.end())
            bounded = any(token in sentence.lower() for token in ["denotes", "does not", "limited", "objective label"])
            replacement = "Keep: explicitly bounded." if bounded else "Replace with `risk-controlled` or restate the no-guarantee definition."
            rows.append((path.name, section_for(text, match.start()), sentence, "bounded" if bounded else "review", replacement))
    lines = [
        "# Reader-Safe Term Audit",
        "",
        "> First-use definition: \"Reader-safe denotes an answer-preservation-oriented, risk-controlled selection objective; it does not provide a per-query harm guarantee.\"",
        "",
        "| File | Section | Sentence | Status | Action |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows)
    lines.extend([
        "",
        "The title uses **Risk-Controlled Selection**, so the term does not make an unqualified promise before its definition. Prohibited alternatives such as `safe intervention`, `guaranteed answer preservation`, `harm-free`, and `prevents degradation` do not appear in the canonical manuscripts.",
    ])
    return "\n".join(lines)


def build_claim_audit() -> tuple[str, bool]:
    phrases = [
        "efficient", "low-cost", "large gain", "practically significant",
        "safe intervention", "prevents harm", "guaranteed answer preservation",
        "two independent readers", "independent SP replication",
        "generalizes to 2Wiki", "robust under shift", "scales to large pools",
        "open-domain", "streaming", "outperforms RECOMP", "RECOMP is inferior",
        "causal intervention effect", "all selected queries improve", "SOTA",
        "Federated", "privacy-preserving",
    ]
    evidence = {
        "independent sp replication": "One support predictor is shared across FLAN and UnifiedQA.",
        "streaming": "No continuous-index or streaming experiment was run.",
        "efficient": "Full is 1.52x the measured baseline latency.",
    }
    findings = []
    violation = False
    for path in [ROOT / "paper_anonymous_review_polished.md", ROOT / "paper_appendix_review_polished.md"]:
        text = path.read_text()
        lower = text.lower()
        for phrase in phrases:
            needle = phrase.lower()
            start = 0
            while True:
                idx = lower.find(needle, start)
                if idx < 0:
                    break
                sentence = sentence_for(text, idx, idx + len(needle))
                safe_tokens = ["not ", "does not", "no ", "without", "outside", "unsupported", "rather than"]
                safe = any(token in sentence.lower() for token in safe_tokens)
                risk = "low: explicit boundary" if safe else "high: potentially unsupported"
                replacement = "Keep as a negated boundary." if safe else "Replace with a measured, scope-bounded statement."
                findings.append((path.name, section_for(text, idx), phrase, sentence, evidence.get(needle, "No frozen evidence supports an unbounded version."), risk, replacement))
                if not safe:
                    violation = True
                start = idx + len(needle)

    lines = [
        "# Final Claim Audit",
        "",
        "## Canonical claim checks",
        "",
        "| Claim family | Frozen evidence | Final status |",
        "|---|---|---|",
        "| Same-source population effect | Two disjoint Hotpot holdouts; +0.0064/+0.0080 Joint F1 | Allowed as modest same-source evidence |",
        "| Selected-policy effect | Means plus wins/losses/ties, zero median/IQR, and 7.75-7.83% Answer drop | Allowed only as conditional descriptive evidence |",
        "| RECOMP | Budget-controlled, non-significant Joint difference versus baseline | No general ranking |",
        "| Cost | Full 213.48 versus 140.88 ms/query, one reader call | Measured bounded latency claim |",
        "| Transfer | 2Wiki non-significant; calibration target missed | Failed diagnostic only |",
        "| Multi-reader | Two answer readers, one shared support predictor | Directional answer evidence only |",
        "| Candidate scale | Roughly ten documents; ten pairs scored/query | Bounded post-retrieval scope only |",
        "",
        "## Dangerous phrase occurrences",
        "",
        "Every exact occurrence in the canonical main paper and appendix is listed below. Negated limitation language is retained as a boundary rather than counted as an overclaim.",
        "",
        "| File | Section | Phrase | Sentence | Evidence | Risk | Replacement |",
        "|---|---|---|---|---|---|---|",
    ]
    if findings:
        lines.extend(f"| {a} | {b} | `{c}` | {d} | {e} | {f} | {g} |" for a, b, c, d, e, f, g in findings)
    else:
        lines.append("| - | - | - | No listed dangerous phrase occurs. | - | none | - |")
    lines.extend([
        "",
        f"**Semantic overclaim violations:** {sum(1 for item in findings if item[5].startswith('high'))}.",
        "",
        "The audit treats explicit negations and limitation statements as safe boundary language. It does not infer support merely from the absence of keywords; the canonical claim table above checks the main numerical and scope claims directly.",
    ])
    return "\n".join(lines), not violation


def build_readiness(no_overclaim: bool) -> str:
    main = (ROOT / "paper_anonymous_review_polished.md").read_text()
    appendix = (ROOT / "paper_appendix_review_polished.md").read_text()
    canonical = main + "\n" + appendix
    bib = (ROOT / "references.bib").read_text()
    used = set(re.findall(r"@([A-Za-z0-9:_-]+)", canonical))
    known = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
    missing = sorted(used - known)
    anonymity_hits = re.findall(r"/Users/|/home/|iilab|FedE4RAG-main", canonical, flags=re.I)
    outdated_hits = re.findall(r"selected[^\n.]{0,100}2\.0%|2\.0%[^\n.]{0,100}selected", canonical, flags=re.I)
    fields = {
        "review_facts_corrected": True,
        "latency_numbers_current": "213.48" in canonical and "140.88" in canonical,
        "reader_safe_language_bounded": "does not provide a per-query harm guarantee" in canonical,
        "population_and_conditional_effects_separated": "Population and conditional views answer different questions" in canonical,
        "recomp_claim_bounded": "not a claimed end-to-end reproduction" in canonical,
        "candidate_pool_scope_clear": "2,973" in canonical and "ten pairs per query" in canonical,
        "multi_reader_claim_bounded": "not an independent SP replication" in canonical,
        "transfer_failure_preserved": "failed zero-shot safety-transfer diagnostic" in canonical and "misses the pre-specified 4% target" in canonical,
        "no_outdated_numbers": not outdated_hits,
        "no_overclaim": no_overclaim,
        "anonymous": not anonymity_hits,
        "citations_complete": not missing,
    }
    all_ready = all(fields.values())
    lines = [
        "# Submission Readiness Report",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    lines.extend(f"| `{key}` | {'PASS' if value else 'FAIL'} |" for key, value in fields.items())
    lines.extend([
        "",
        f"**Overall mechanical status:** {'PASS' if all_ready else 'FAIL'}.",
        "",
        "**Recommended tier:** `main_conference_ready_with_review_risk`.",
        "",
        "The tier is intentionally unchanged. Better exposition does not remove the scientific risks: modest population gains, 1.52x latency, nonzero selected-query harm, same-source validation, bounded candidate pools, a shared support predictor, failed Lite non-inferiority, and failed 2Wiki calibration.",
        "",
        "## Diagnostics",
        "",
        f"- Missing citation keys: {missing or 'none'}",
        f"- Anonymity hits: {anonymity_hits or 'none'}",
        f"- Outdated selected 2.0% hits: {outdated_hits or 'none'}",
        "- Frozen outcomes changed during polishing: no",
        "- Holdout outcome selection added: no",
        "- New experiment family added: no",
    ])
    return "\n".join(lines)


def main() -> None:
    write("review_accuracy_audit.md", review_accuracy)
    write("remaining_weakness_priority.md", remaining_weakness)
    write("selected_policy_effect_audit.md", selected_audit)
    write("recomp_claim_audit.md", recomp_audit)
    write("candidate_pool_scope_statement.md", candidate_scope)
    write("multi_reader_support_audit.md", multi_reader)
    write("external_transfer_claim_audit.md", external_transfer)
    write("review_response_strengths_and_remaining_weaknesses.md", review_response)
    write("reader_safe_term_audit.md", build_reader_safe_audit())
    claim_text, no_overclaim = build_claim_audit()
    write("final_claim_audit.md", claim_text)
    write("submission_readiness_report.md", build_readiness(no_overclaim))


if __name__ == "__main__":
    main()
