# Review Weakness Audit

- Task: `V7-HP-PAPER-review-driven-revision-v5`
- Protocol frozen: `2026-07-14`
- Frozen V4 directories modified: `false`
- Untouched revision holdout eligible: `true` (expected slice 4000:7405; identity audit still required)

## Source Inventory

| Artifact | Exists | Rows/bytes | Path |
|---|---:|---:|---|
| overleaf_pdf | true | 344008 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_final_submission_refinement/overleaf_v4_final/main.pdf` |
| paper_full | true | 38802 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/paper_full_clean_v4_submission.md` |
| paper_main | true | 17846 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/paper_main_conference_v4_submission.md` |
| paper_appendix | true | 14404 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/paper_appendix_v4_submission.md` |
| v4_actions | true | 8934 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/generated_actions/v4_outer_test_actions.jsonl` |
| v4_action_outcomes | true | 8934 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/action_outcomes/v4_action_outputs.jsonl` |
| v4_nested_selector | true | 1000 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/nested_selector/v4_nested_per_query.jsonl` |
| v4_official_per_query | true | 2000 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/official_metrics/official_hotpotqa_per_query.jsonl` |
| holdout_3000_contexts | true | 3000 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_baseline_contexts_3000.jsonl` |
| holdout_3000_reader | true | 6000 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/official_metrics/flan_per_query.jsonl` |
| holdout_source_audit | true | 2140 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/same_source_context_audit.json` |
| two_wiki_summary | true | 2463 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/outputs/external_2wiki_frozen/external_validation_results.json` |
| recomp_top1_summary | true | 2867 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/outputs/faithful_baseline/faithful_baseline_results.json` |
| generator_ablation_summary | true | 6052 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v4_submission_completion/outputs/generator_ablation/generator_ablation_results.json` |
| reader_manifest | true | 836 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/action_outcomes/reader_environment_manifest.json` |
| fold_manifest | true | 44313 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/semantic_generator/foldwise_generator_models.json` |
| support_outputs | true | 1515 | `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/official_metrics/official_hotpotqa_summary.json` |

## 1. Marginal absolute gains (P1)

- **Reviewer concern:** Marginal absolute gains
- **is_valid:** `true`
- **Current paper evidence:** The frozen 3,000-query result reports modest population-level gains (Answer +0.0088, SP +0.0056, Joint +0.0064).
- **Scientific risk:** A reviewer may consider the average effect too small without conditional intervention effects and deployment cost.
- **Required experiment:** Compute exact selected-query and fallback-query effects from paired per-query outputs; report gain per 100 interventions and online overhead.
- **Required writing change:** Lead with selective intervention scope, report both population and conditional effects, and avoid practical-impact inflation.
- **Blocks submission:** `partially`

## 2. Limited domain transfer (P1)

- **Reviewer concern:** Limited domain transfer
- **is_valid:** `true`
- **Current paper evidence:** Frozen 2Wiki transfer is non-significant, support is flat, and selected answer-drop rises from 2.0% in-domain to 6.92%.
- **Scientific risk:** The current draft cannot claim robust cross-dataset generalization.
- **Required experiment:** Run nested K-shot safety calibration on 2Wiki train only and preserve the existing 1,000-query dev evaluation set.
- **Required writing change:** Separate zero-shot failure from few-shot calibration and retain distribution shift as a limitation.
- **Blocks submission:** `no, if claims are narrowed`

## 3. Mixed semantic component ablations (P0/P1)

- **Reviewer concern:** Mixed semantic component ablations
- **is_valid:** `true`
- **Current paper evidence:** Generator ablations show the most stable mechanism-level signal for pair complementarity and bounded two-document construction; other semantic features are mixed.
- **Scientific risk:** Presenting every semantic feature as an independent contribution overstates the evidence and leaves an unnecessarily complex method.
- **Required experiment:** Evaluate fully nested Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain-Ablation; freeze a 0.002 Joint-F1 non-inferiority margin before holdout access.
- **Required writing change:** Center pair complementarity, bounded chains, anchor preservation, and reader-safe selection; move full semantic implementation details to the appendix.
- **Blocks submission:** `yes unless method claims are simplified`

## 4. Unfair RECOMP comparison (P0)

- **Reviewer concern:** Unfair RECOMP comparison
- **is_valid:** `true`
- **Current paper evidence:** The original RECOMP context averages 47.13 tokens versus 660.57 for V4 and 668.18 for the baseline.
- **Scientific risk:** Top-1 sentence compression cannot support a general superiority claim against near-full Top-5 contexts.
- **Required experiment:** Run official RECOMP sentence scoring at 64/128/256/384/512/660 tokens, Baseline-Truncated controls, the same reader, support predictor, and paired protocol; freeze 660 before the 3,000-query holdout.
- **Required writing change:** Remove the old superiority statement. Retain a main-table comparison only if the matched-budget experiment completes.
- **Blocks submission:** `yes`

## 5. High operational complexity (P1)

- **Reviewer concern:** High operational complexity
- **is_valid:** `partially_true`
- **Current paper evidence:** The current method includes MPNet, a cross-encoder, multiple generator heads, nested selection, and expensive offline action labeling; online cost has not been isolated from offline development cost.
- **Scientific risk:** Readers may infer that every candidate action requires an online reader call or that the small gain cannot justify deployment overhead.
- **Required experiment:** Benchmark baseline, Full, Lite, RECOMP Top-1, and budget-matched RECOMP; separately measure offline labels and online single-reader inference.
- **Required writing change:** State prominently that deployment runs the reader once on the selected final context, while candidate reader outcomes are offline supervision only.
- **Blocks submission:** `partially`

## Revision Order

1. Budget-match RECOMP or remove its numeric superiority comparison.
2. Test the pre-registered Lite variants under fully nested evaluation.
3. Measure online and offline cost separately and compute exact selected-query effects.
4. Calibrate 2Wiki safety with target-train examples only.
5. Rewrite the paper around complementary pairs, bounded chains, anchor preservation, and reader-safe selective intervention.
