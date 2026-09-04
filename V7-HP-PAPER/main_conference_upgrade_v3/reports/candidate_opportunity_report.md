# Candidate Opportunity Report

## Scope and leakage boundary

This is a **diagnostic-only** analysis of the frozen 797 no-positive queries. It uses gold support and reader outcomes to explain failures. None of its labels or target-derived values is exported to the v3 candidate generator or held-out selector features.

## Primary taxonomy

| Primary diagnostic category | Queries | Share of 797 |
| --- | --- | --- |
| strong_baseline_with_no_improvable_context | 386 | 48.4% |
| answer_safe_but_evidence_neutral | 206 | 25.8% |
| candidate_pool_lacks_gold_support_title | 163 | 20.5% |
| candidate_exists_but_wrong_insertion_slot | 27 | 3.4% |
| document_order_failure | 5 | 0.6% |
| redundancy_or_duplicate_occupation | 4 | 0.5% |
| candidate_exists_but_displaces_answer_anchor | 4 | 0.5% |
| reader_prediction_variance_unmeasured | 1 | 0.1% |
| single_document_action_insufficient | 1 | 0.1% |

## Required answers

1. Full source pool lacks a gold-support title: **0 / 797**.
2. The older exposed action pool lacks a gold-support title: **163 / 797**.
3. A suitable source document exists but the current templates do not place it safely: **195 / 797**.
4. Baseline misses both support titles and therefore needs a bounded two-document action: **10 / 797**.
5. Evidence-positive edits displace an answer anchor: **5 / 797**.
6. Baseline already has perfect answer F1 and title recall: **386 / 797**.

## Generator priorities

The three most defensible families are anchor-preserving tail replacement, bridge-aware complementary insertion, and a newly bounded two-document chain. They directly target safe placement, cross-document complementarity, and the single-action expressivity limit while keeping the search space finite.
