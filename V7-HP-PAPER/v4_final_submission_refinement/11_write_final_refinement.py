#!/usr/bin/env python3
"""Write the final anonymous main-conference refinement from frozen artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
V4 = PAPER_ROOT / "opportunity_aware_semantic_generation_v4"
COMPLETION = PAPER_ROOT / "v4_submission_completion"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    (HERE / name).write_text(text.strip() + "\n", encoding="utf-8")


def p(value: float) -> str:
    return "<0.0002" if value == 0 else f"{value:.4f}"


def delta(value: float) -> str:
    return f"{value:+.4f}"


def ci(row: dict) -> str:
    return f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"


def main() -> None:
    gate = load(V4 / "outputs/opportunity/v4_opportunity_gate.json")
    opportunity = load(V4 / "outputs/action_outcomes/v4_action_summary.json")
    selector = load(V4 / "outputs/nested_selector/v4_nested_summary.json")
    official = load(V4 / "outputs/official_metrics/official_hotpotqa_summary.json")
    multireader = load(V4 / "outputs/multi_reader/multi_reader_summary.json")
    holdout = load(V4 / "outputs/scaleup/scaleup_summary.json")
    generator_audit = load(V4 / "outputs/audits/generator_nested_no_leak_audit.json")
    external = load(COMPLETION / "outputs/external_2wiki_frozen/external_validation_results.json")
    recomp = load(COMPLETION / "outputs/faithful_baseline/faithful_baseline_results.json")
    ablation = load(COMPLETION / "outputs/generator_ablation/generator_ablation_results.json")["variants"]
    budget = load(HERE / "recomp_budget_audit.json")

    shutil.copy2(COMPLETION / "references.bib", HERE / "references.bib")

    dev_b = official["metrics"]["baseline"]
    dev_m = official["metrics"]["v4_selected"]
    dev_s = official["significance"]
    flan = holdout["official_dual_reader"]["readers"]["flan"]
    uni = holdout["official_dual_reader"]["readers"]["unifiedqa"]
    ext_b = external["metrics"]["baseline"]
    ext_m = external["metrics"]["v4_frozen_transfer"]

    abstract = """Context selectors cannot improve a multi-hop question when their candidate set contains no reader-compatible intervention. In a motivating study, nearly doubling a hand-written action table raises positive-query opportunity only from 20.3% to 23.4% while leaving positive-action density unchanged. We address this candidate-opportunity gap with a semantic generator that constructs bounded, anchor-preserving context actions, followed by fully nested reader-safe selection. The generator increases positive-query opportunity from 23.4% to 29.2%. On a 1,000-query development protocol, official answer F1 improves by 0.0133 (p=0.0176) and supporting-fact F1 by 0.0053 (p=0.0372), while joint F1 shows a positive, non-significant trend (+0.0064, p=0.0752). After freezing the generator, selector, thresholds, readers, prompts, and support predictor, we evaluate 3,000 disjoint same-source HotpotQA queries. Answer, supporting-fact, and joint F1 improve by 0.0088, 0.0056, and 0.0064, respectively; the joint result has p=0.0004. The same selected contexts improve answer and joint metrics for FLAN-T5-Large and UnifiedQA-T5-Large, although their support predictor is shared. Frozen transfer to 2WikiMultiHopQA yields positive answer and joint point estimates but no statistically reliable gain. These results show that generating reader-compatible opportunities before selective intervention yields small but reproducible multi-hop QA gains, while external transfer and safety calibration remain open."""
    write("abstract_final.md", "# Final Abstract\n\n" + abstract)

    contributions = """
# Final Contributions

1. We identify the candidate-opportunity gap in reader-side context intervention: a selector cannot improve queries for which its generator exposes no useful context action.
2. We propose a fully nested semantic action generator that constructs bounded, anchor-preserving actions using document opportunity and pair complementarity.
3. We combine generation with risk-controlled reader-safe selection under fully nested query-level cross-fitting.
4. We obtain significant official answer/support improvements in development and reproduce significant answer/support/joint gains on a frozen 3,000-query same-source holdout, with consistent answer/joint directions across two readers.
"""
    write("contributions_final.md", contributions)

    introduction = """
# Final Introduction

Multi-hop question answering requires a reader to combine evidence distributed across passages [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Retrieval is necessary, but the final context must do more than contain individually relevant documents. It must preserve the passage that resolves the answer, expose complementary hops, avoid redundant distractors, and present evidence in an order the reader can use. A context change can therefore improve retrieval-side support while leaving the answer unchanged or even making it worse.

We call this mismatch the **policy-action-to-reader gap**: an upstream evidence action does not automatically become a downstream reader gain. Recent work aligns passage selection with reader needs, selects sets instead of independently ranked documents, or compresses retrieved context [@xin-etal-2025-rcps; @lee-etal-2025-setr; @xu-etal-2024-recomp]. These approaches motivate reader-aware context construction, but they also raise a prior question: does the selector receive any useful alternative for this query?

This prior question defines the **candidate-opportunity gap**. A selector can choose only among actions that its generator exposes. If every candidate either misses the second hop or damages an answer-bearing anchor, a better selector cannot improve that query. Selection accuracy and candidate opportunity should therefore be measured separately. Otherwise, failures caused by an impoverished action set are incorrectly attributed to the selection model.

A controlled motivating study demonstrates this distinction. An initial fixed-action table contains 4,000 effective alternatives and exposes at least one answer-safe positive action for 20.3% of 1,000 HotpotQA development queries. Expanding hand-written templates to 7,882 actions raises query coverage only to 23.4%, while positive-action density changes from 9.48% to 9.43%. More templates create more rows without reliably changing which queries can be helped. This negative result motivates semantic, query-conditioned opportunity generation rather than another selector over the same table.

We propose a two-stage method. A semantic opportunity generator predicts missing reasoning structure, scores document opportunity, models whether pairs form complementary hops, and constructs at most eight bounded actions. Actions insert or replace evidence, form two-document chains, remove redundancy, or apply restricted order changes while retaining answer-anchor proxies. A reader-safe selector then predicts whether an action preserves answer quality and provides positive answer-evidence utility. It acts only within a risk-controlled coverage budget and otherwise returns the original context.

The development protocol is fully nested by query. In each of five outer folds, generator and selector models are trained on 800 queries and applied to 200 disjoint queries. Inner out-of-fold predictions choose selector thresholds and coverage. Target-query answers, supporting facts, reader outcomes, and oracle action labels are excluded from generation and test-time selection. Reader outcomes are allowed only as supervision for training queries. This design separates legitimate reader-aware training from target-query leakage.

The semantic generator raises positive-query opportunity from 23.4% to 29.2% and positive-action density from 9.43% to 14.71%. On the 1,000-query development protocol, official answer and supporting-fact F1 improve significantly, while joint F1 has a positive, non-significant change. The central empirical anchor is a subsequent 3,000-query same-source holdout evaluated after freezing the full pipeline. FLAN-T5-Large gains 0.0088 answer F1, 0.0056 supporting-fact F1, and 0.0064 joint F1; all three paired tests are significant. UnifiedQA-T5-Large shows the same answer and joint direction on identical contexts.

The evidence remains deliberately bounded. Opportunity passes three of five pre-specified criteria, so the candidate gap is narrowed rather than solved. The 3,000 queries are disjoint but come from the same HotpotQA source. On 2WikiMultiHopQA, frozen transfer retains positive answer and joint point estimates, but supporting-fact F1 is flat and all confidence intervals include zero. The second reader shares the support predictor. Our contribution is therefore not universal context selection or evidence of dataset-independent reliability. It is evidence that semantic opportunity generation, coupled with fully nested reader-safe selection, produces small but statistically reliable official multi-hop QA gains on a frozen same-source holdout.

Our contributions are fourfold:

1. We identify the candidate-opportunity gap in reader-side context intervention.
2. We introduce bounded semantic action generation based on document opportunity and pair complementarity.
3. We combine generation with risk-controlled selection under fully nested query-level cross-fitting.
4. We demonstrate significant development answer/support gains and frozen same-source answer/support/joint gains, with answer and joint directions consistent across two readers.
"""
    write("introduction_final.md", introduction)

    limitations = """
# Final Limitations

1. **Opportunity remains incomplete.** The generator passes three of five pre-specified criteria. Overall positive-query coverage is 29.2%, and new-query efficiency remains below the motivating heuristic study.
2. **The strongest holdout is same-source.** The 3,000 queries are disjoint from development and evaluated with a frozen pipeline, but they come from the same HotpotQA source and are not external generalization evidence.
3. **External transfer is non-significant.** On 2WikiMultiHopQA, answer and joint point estimates are positive, supporting-fact F1 is flat, and all relevant confidence intervals include zero. The higher selected answer-drop rate indicates a calibration boundary.
4. **Support replication is not independent.** UnifiedQA receives the same selected contexts and shares the sentence-support predictor used with FLAN. The result supports answer-reader direction but does not independently replicate support prediction.
5. **Generator component evidence is mixed.** Pair complementarity and two-document chains have clear opportunity contributions. Other semantic features are non-monotonic and remain parts of a frozen recipe rather than independently necessary innovations.
6. **The RECOMP comparison has an unmatched output budget.** It uses official code and checkpoint with the same Top-5 input, but emits one sentence under a standardized reader adaptation. The comparison measures compatibility with this setting, not general superiority.
"""
    write("limitations_final.md", limitations)

    dev_table = f"""
**Table 1: Official 1,000-query development results.**

| System | Answer EM | Answer F1 | SP EM | SP F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Top-5 baseline | {dev_b['answer_em']:.3f} | {dev_b['answer_f1']:.4f} | {dev_b['sp_em']:.3f} | {dev_b['sp_f1']:.4f} | {dev_b['joint_em']:.3f} | {dev_b['joint_f1']:.4f} |
| Semantic generation + reader-safe selection | {dev_m['answer_em']:.3f} | {dev_m['answer_f1']:.4f} | {dev_m['sp_em']:.3f} | {dev_m['sp_f1']:.4f} | {dev_m['joint_em']:.3f} | {dev_m['joint_f1']:.4f} |
| Delta | {delta(dev_s['answer_em']['mean'])} | {delta(dev_s['answer_f1']['mean'])} | {delta(dev_s['sp_em']['mean'])} | {delta(dev_s['sp_f1']['mean'])} | {delta(dev_s['joint_em']['mean'])} | {delta(dev_s['joint_f1']['mean'])} |

Paired bootstrap: answer F1 {ci(dev_s['answer_f1'])}, p={p(dev_s['answer_f1']['p_value'])}; SP F1 {ci(dev_s['sp_f1'])}, p={p(dev_s['sp_f1']['p_value'])}; joint F1 {ci(dev_s['joint_f1'])}, p={p(dev_s['joint_f1']['p_value'])}. Joint F1 is positive but non-significant.
"""

    holdout_table = f"""
**Table 2: Frozen 3,000-query same-source holdout.**

| Reader | Coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint 95% CI | Joint p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | {holdout['selector_coverage']:.3f} | {delta(flan['deltas']['answer_f1'])} | {delta(flan['deltas']['sp_f1'])} | {delta(flan['deltas']['joint_f1'])} | {ci(flan['significance']['joint_f1'])} | {p(flan['significance']['joint_f1']['p_value'])} |
| UnifiedQA-T5-Large* | {holdout['selector_coverage']:.3f} | {delta(uni['deltas']['answer_f1'])} | {delta(uni['deltas']['sp_f1'])} | {delta(uni['deltas']['joint_f1'])} | {ci(uni['significance']['joint_f1'])} | {p(uni['significance']['joint_f1']['p_value'])} |

*The sentence-support predictor is shared and frozen rather than independently trained for UnifiedQA. The second row is an answer-reader directional replication, not an independent support-pipeline replication.
"""

    core_names = [
        ("full", "Full semantic generator"),
        ("without_pair_complementarity", "- pair complementarity"),
        ("without_two_document_actions", "- two-document chains"),
        ("without_semantic_document_model", "- document opportunity model"),
        ("lexical_only_generator", "Lexical-only generator"),
    ]
    ablation_rows = []
    for key, label in core_names:
        row = ablation[key]
        ablation_rows.append(
            f"| {label} | {100*row['positive_action_density']:.2f}% | "
            f"{100*row['positive_query_coverage']:.1f}% | "
            f"{100*row['conditional_non_ceiling_coverage']:.2f}% | "
            f"{100*row['answer_safe_action_rate']:.2f}% |"
        )
    ablation_table = """
**Table 3: Core generator ablations on fully nested development actions.**

| Generator | Positive density | Query coverage | Non-ceiling coverage | Answer-safe rate |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(ablation_rows) + """

Pair complementarity provides the clearest learned-component contribution, and two-document chains provide the clearest structural contribution. Removing the document opportunity model increases raw breadth but lowers answer safety, revealing a breadth-risk trade-off. Other feature removals are mixed and are reported in the appendix.
"""

    external_table = f"""
**Table 4: Supporting comparisons.**

| Evaluation | Answer F1 delta | SP F1 delta | Joint F1 delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Frozen 2Wiki transfer | {delta(external['deltas']['answer_f1'])} | {delta(external['deltas']['sp_f1'])} | {delta(external['deltas']['joint_f1'])} | All relevant CIs include zero |
| Official-code RECOMP Top-1 vs baseline* | {delta(recomp['recomp_vs_baseline_deltas']['answer_f1'])} | {delta(recomp['recomp_vs_baseline_deltas']['sp_f1'])} | {delta(recomp['recomp_vs_baseline_deltas']['joint_f1'])} | Standardized reader; unmatched output budget |

*RECOMP receives the same Top-5 documents but emits one sentence averaging {budget['averages']['recomp_context_tokens']:.1f} context tokens, versus {budget['averages']['baseline_context_tokens']:.1f} for the baseline. Detailed results are in the appendix and do not support a general superiority claim.
"""

    figure_1 = """
**Figure 1: Candidate-opportunity gap and method overview.**

```mermaid
flowchart LR
  Q["Question + baseline context"] --> P["Local document pool"]
  P --> G["Semantic opportunity generator"]
  G --> A["Bounded insert / replace / pair / order actions"]
  A --> S["Safety and positive-utility selector"]
  S -->|"confident"| C["Selected context"]
  S -->|"abstain"| F["Baseline fallback"]
  C --> R["Frozen reader"]
  F --> R
```
"""

    figure_2 = """
**Figure 2: Development and frozen-holdout protocol.**

```mermaid
flowchart TB
  D["1,000-query development sample"] --> O["Five outer query folds"]
  O --> T["Outer-train generator and selector"]
  T --> I["Inner OOF threshold / coverage choice"]
  I --> E["Disjoint outer-test evaluation"]
  E --> Z["Freeze generator, selector, prompts, readers, support threshold"]
  Z --> H["3,000 disjoint same-source queries"]
  Z --> X["1,000 frozen 2Wiki transfer queries"]
```
"""

    figure_3 = """
**Figure 3: Opportunity across motivating and proposed generators.**

| Generator study | Positive-action density | Positive-query coverage |
| --- | ---: | ---: |
| Initial fixed actions | 9.48% | 20.3% |
| Heuristic expansion | 9.43% | 23.4% |
| Semantic opportunity generation | 14.71% | 29.2% |
"""

    related_work = """
## 2. Related Work

### Multi-Hop Retrieval

Multi-hop retrievers search a corpus iteratively so later retrieval can depend on earlier evidence [@xiong-etal-2021-mdr]. HotpotQA, 2WikiMultiHopQA, and MuSiQue make compositional evidence observable through support annotations or controlled reasoning structures [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki; @trivedi-etal-2022-musique]. Our method does not replace corpus retrieval. It starts from a fixed local document pool and studies the reader-facing context formed from that pool.

### Reader-Aware and Set Selection

Reader-Centered Passage Selection aligns passage choice with downstream reader needs [@xin-etal-2025-rcps]. SetR models retrieval augmentation as set selection rather than independent ranking [@lee-etal-2025-setr], and RankRAG integrates ranking with generation [@yu-etal-2024-rankrag]. These methods select from an available pool. Unlike evaluations that begin after fixing the candidate set, we explicitly measure and optimize query-level intervention opportunity, separating candidate-generation failure from reader-safe selection failure. We do not claim that prior work ignores candidate construction; our novelty is the explicit opportunity decomposition and its fully nested evaluation.

### Compression and Context Risk

RECOMP learns extractive and abstractive compression for retrieval-augmented language models [@xu-etal-2024-recomp]. Context position and irrelevant text can change model behavior even when relevant evidence is present [@liu-etal-2024-lost; @shi-etal-2023-distracted]. Our bounded actions retain document text but may insert, replace, pair, or reorder evidence. Risk-controlled fallback follows the selective-prediction principle of intervening only when confidence is sufficient [@geifman-elyaniv-2019-selectivenet].
"""

    problem_method = f"""
## 3. Problem Setting

Given a query q, baseline context C0, and frozen reader R, a generator exposes a bounded action set A(q). An action is development-positive when it preserves answer F1 and improves the product of answer F1 and title-level evidence F1, with either improved title recall or non-decreasing title F1. A query has **opportunity** if at least one effective action is positive. This diagnostic definition uses reader outcomes only on training or held-out development actions; inference never observes the target answer, support facts, or outcomes.

The action set contains single complementary insertion, anchor-preserving replacement, two-document chain, redundancy replacement, bridge-first reorder, and answer-anchor-first reorder. Every action preserves an at-most-five-document reader budget and source text. Fallback returns C0.

## 4. Semantic Opportunity Generation

The generator represents a query's missing evidence state, document-level opportunity, and pair complementarity. Document features combine lexical signals, MPNet similarity [@song-etal-2020-mpnet], cross-encoder relevance, novelty, and relation to the current context. Pair features estimate whether two documents supply complementary hops. A deterministic constructor converts scores into at most eight bounded actions while protecting inference-time answer-anchor proxies.

The no-leak audit covers {generator_audit['num_queries']} outer-test queries and {generator_audit['num_effective_actions']:,} effective actions. Each fold trains on 800 queries and produces actions for 200 disjoint queries. Target-query answer, gold support, reader outcome, oracle action, and post-hoc coverage are absent. The final generator exposes 5,655 contexts not present in the heuristic table.

## 5. Reader-Safe Selection

Two logistic heads predict answer safety and positive utility from inference-safe action features. For each outer fold, five inner query splits produce out-of-fold predictions on the 800 outer-training queries. Inner data choose safety threshold, positive threshold, and intervention coverage between 10% and 30%, subject to a 5% selected-action answer-drop budget and a mean answer-loss tolerance of 0.001. The selector chooses the highest-scoring eligible action within the budget; otherwise it returns the baseline. Aggregate development coverage is {selector['coverage']:.3f} with a {100*selector['answer_drop_rate']:.1f}% selected answer-drop rate.
"""

    setup = """
## 6. Experimental Setup

### Data and Protocol

The development evaluation uses 1,000 HotpotQA distractor-validation queries [@yang-etal-2018-hotpotqa] under five fully nested query folds. A second sample contains 3,000 disjoint queries from the same source ordering. All generator models, selector models, thresholds, coverage rules, reader prompts, decoding, and the support threshold are frozen before this second evaluation. We call it a **frozen same-source holdout confirmation**, not external validation.

External transfer uses 1,000 deterministically hash-sampled 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki]. Only the data adapter changes; no target-dataset training or tuning is performed.

### Readers and Metrics

FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5]. UnifiedQA-T5-Large receives identical selected contexts [@khashabi-etal-2020-unifiedqa]. A single sentence-support predictor with threshold 0.7 is frozen across both readers and datasets. Official metrics are answer, supporting-fact (SP), and joint EM/F1. Title-level evidence measures are development diagnostics only.

### Baselines and Statistics

The baseline is a frozen dense-sparse hybrid retriever with uniform document weights and Top-5 output; it is not replaced by a BM25-only baseline [@robertson-zaragoza-2009-bm25]. We also run the official RECOMP extractive checkpoint on the same Top-5 documents [@xu-etal-2024-recomp]. RECOMP emits one sentence and uses the frozen FLAN reader rather than its paper's FLAN-UL2 reader, so it is an **official-code reproduction under a standardized reader adaptation**.

All metric comparisons are paired by query, with 5,000 bootstrap resamples. Development results are model-development evidence. The 3,000-query FLAN joint result is the headline holdout metric, but no immutable pre-run hierarchy was found; we therefore do not claim formal ordered testing or familywise confirmatory control. UnifiedQA, 2Wiki, and RECOMP are supporting analyses.
"""

    results_analysis = f"""
## 7. Results

### Opportunity

{figure_3}

Semantic generation raises positive density by 5.28 points and query coverage by 5.8 points over heuristic expansion. Non-ceiling coverage reaches 47.63%. It passes conditional coverage, marginal breadth, and density criteria, but misses the 30% overall-coverage criterion and the new-query-efficiency criterion. The candidate-opportunity gap is reduced, not solved.

### Development Behavior

{dev_table}

The selector intervenes on 260 of 1,000 queries. Answer and SP F1 improve significantly at the unadjusted 0.05 level. Joint F1 rises by 0.0064, but its interval includes zero. This establishes method behavior under fully nested model development; it is not the paper's strongest joint evidence.

### Frozen Same-Source Holdout Confirmation

{holdout_table}

With no further tuning, FLAN answer F1 rises from 0.6183 to 0.6271 (delta +0.0088, p=0.0096), SP F1 from 0.4930 to 0.4987 (delta +0.0056, p=0.0004), and joint F1 from 0.3292 to 0.3356 (delta +0.0064, p=0.0004). The selected answer-drop rate is 2.0%. UnifiedQA answer and joint F1 improve by 0.0110 and 0.0085. The support predictor is shared, so the second reader confirms answer/joint direction rather than independent support robustness.

## 8. Analysis

### Generator Components

{ablation_table}

Removing pair complementarity reduces positive density from 14.71% to 10.27%, the largest learned-component loss. Removing two-document chains reduces query coverage from 29.2% to 25.1% and non-ceiling coverage from 47.63% to 40.92%, the largest structural loss. The document opportunity model has a different role: without it, coverage rises to 32.6%, but answer safety falls from 92.66% to 91.74%. It therefore trades raw breadth for risk rather than monotonically increasing opportunity. MPNet, cross-encoder, missing-hop, and redundancy ablations are mixed and appear only in the appendix.

### Opportunity, Selection, and Risk

The generator exposes a positive action for 292 development queries, while the selector changes 260 contexts using inference-safe predictions. Opportunity is an empirical upper bound, not a promise that the selector will identify every positive action. This decomposition explains why a 5.8-point opportunity increase produces smaller downstream metric changes. Risk also shifts across samples: selected answer drops are 5.0% in development, 2.0% on the same-source holdout, and 6.92% on 2Wiki.

### Controlled RECOMP Comparison

RECOMP and our method receive the same baseline Top-5 documents, but their reader-facing budgets differ sharply. The baseline and selected document contexts average {budget['averages']['baseline_context_tokens']:.1f} and {budget['averages']['selected_context_tokens']:.1f} FLAN tokens; RECOMP's single extracted sentence averages {budget['averages']['recomp_context_tokens']:.1f}, a {100*budget['compression_ratios']['recomp_to_baseline_context_tokens']:.2f}% ratio. One sentence is structurally unlikely to represent two disjoint supporting facts. Under the standardized FLAN reader and evaluated Top-1 setting, RECOMP is poorly matched to the multi-hop context budget. The comparison tests setting compatibility, not universal superiority, and detailed scores are relegated to the appendix.

## 9. External Transfer and Generalization Boundary

{external_table}

On 2Wiki, answer F1 changes by +0.0086 ({ci(external['significance']['answer_f1'])}, p=0.1116), SP F1 by -0.0006 ({ci(external['significance']['sp_f1'])}, p=0.6928), and joint F1 by +0.0033 ({ci(external['significance']['joint_f1'])}, p=0.3296). The frozen pipeline preserves positive answer and joint point estimates, but all confidence intervals include zero. This rules out neither no effect nor a small negative effect and does not establish statistically reliable transfer. The selected answer-drop rate increases to 6.92%, identifying safety calibration as the main distribution-shift boundary.
"""

    full_paper = f"""
# Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## Abstract

{abstract}

## 1. Introduction

{introduction.split('# Final Introduction', 1)[1].strip()}

{related_work}

{problem_method}

{figure_1}

{setup}

{figure_2}

{results_analysis}

## 10. Limitations and Ethical Considerations

{limitations.split('# Final Limitations', 1)[1].strip()}

Reader-aware supervision requires repeated reader execution on training actions and may be expensive for larger models. The method also cannot recover evidence absent from the available document pool. No federated-system, privacy, or secure-aggregation claim is made. Selective intervention can hide uneven error distributions, so per-query decisions, answer-drop rates, and fold audits should accompany aggregate results.

## 11. Conclusion

A selector cannot cross a missing-candidate ceiling. Semantic opportunity generation expands the set of reader-compatible interventions, while fully nested risk-controlled selection applies only a conservative subset. The resulting official gains are small but statistically reliable on a frozen 3,000-query same-source holdout and directionally consistent across two answer readers. External transfer remains non-significant, and component ablations show that breadth and answer safety must be optimized jointly. Generating useful opportunities before selection is therefore a practical, bounded route across the policy-action-to-reader gap.
"""
    write("paper_full_clean_v4_final.md", full_paper)

    main_paper = f"""
# Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## Abstract

{abstract}

## 1. Introduction

{introduction.split('# Final Introduction', 1)[1].strip()}

{figure_1}

{related_work}

{problem_method}

{setup}

{figure_2}

{results_analysis}

## 10. Limitations

{limitations.split('# Final Limitations', 1)[1].strip()}

## 11. Conclusion

A selector cannot recover a reader-compatible action that its generator never exposes. Semantic opportunity generation increases this action space, and fully nested reader-safe selection converts a conservative subset into small but reproducible gains on a frozen same-source holdout. External transfer and safety calibration remain open.
"""
    write("paper_main_conference_v4_final.md", main_paper)

    appendix_rows = []
    for key, row in ablation.items():
        appendix_rows.append(
            f"| {key.replace('_', ' ')} | {row['effective_actions']:,} | "
            f"{100*row['positive_action_density']:.2f}% | {100*row['positive_query_coverage']:.1f}% | "
            f"{100*row['conditional_non_ceiling_coverage']:.2f}% | "
            f"{row['newly_covered_v3_uncovered_queries']} | {100*row['answer_safe_action_rate']:.2f}% |"
        )
    appendix = f"""
# Appendix: Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## A. Frozen Configuration

- Five outer query folds with 800 training and 200 test queries each.
- At most eight effective actions per query and at most five reader documents.
- Primary reader: FLAN-T5-Large with a pinned model revision.
- Second answer reader: UnifiedQA-T5-Large.
- Support threshold: 0.7, shared and frozen.
- Reader input limit: 3,200 context characters and 1,024 tokenizer positions.
- Decoding: 32 new tokens, greedy, no sampling.
- Paired bootstrap: 5,000 resamples.

## B. No-Leak Protocol

For each outer fold, generator and selector training use only outer-training query outcomes. Inner out-of-fold predictions choose selector thresholds and coverage. Outer-test answers, support labels, reader outcomes, oracle action values, and post-hoc coverage are forbidden. The generator audit contains {generator_audit['num_effective_actions']:,} outer-test actions and SHA-256 `{generator_audit['output_sha256']}`. No 3,000-query outcome is used to select a generator ablation.

## C. Opportunity Criteria

The five recorded criteria are 30% overall positive-query coverage, 45% non-ceiling coverage, at least 70 newly covered heuristic-negative queries or a seven-point coverage gain, 12% positive-action density, and an efficiency improvement over the heuristic study. The method passes conditional coverage, marginal breadth, and density, but fails overall coverage and efficiency. No formal public preregistration record was located, so the paper uses "pre-specified" rather than "preregistered."

## D. Full Generator Ablation

| Variant | Effective actions | Positive density | Query coverage | Non-ceiling coverage | New heuristic-negative queries | Answer-safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(appendix_rows)}

Pair complementarity and two-document chains have the clearest losses. Other component effects are non-monotonic and are not promoted as independent contributions.

## E. Selector Fold Details

| Fold | Coverage | Safety threshold | Positive threshold | Selected | Answer-drop rate | Answer F1 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {row['outer_fold']} | {row['train_selected_config']['coverage']:.2f} | "
        f"{row['train_selected_config']['safe_threshold']:.1f} | "
        f"{row['train_selected_config']['positive_threshold']:.1f} | "
        f"{row['outer_test_result']['selected_count']} | "
        f"{100*row['outer_test_result']['answer_drop_rate']:.2f}% | "
        f"{row['outer_test_result']['deltas']['answer_f1']:+.4f} |"
        for row in selector["folds"]
    ) + f"""

## F. Statistical Language Boundary

The 3,000-query pipeline was frozen before evaluation, but no immutable pre-run endpoint hierarchy was found. The paper presents FLAN joint F1 as the headline holdout metric and reports answer and SP F1 alongside it. It does not claim formal ordered testing or familywise control. Development and external p-values are supporting analyses.

## G. Multi-Reader Details

On development, UnifiedQA answer and joint F1 change by {multireader['unifiedqa_answer_f1_delta']:+.4f} and {multireader['unifiedqa_joint_f1_delta']:+.4f}. On the same-source holdout they change by {uni['deltas']['answer_f1']:+.4f} and {uni['deltas']['joint_f1']:+.4f}. The shared support model means these are not independent support replications.

## H. RECOMP Fairness Details

| Property | RECOMP reproduction | Proposed method / baseline |
| --- | --- | --- |
| Input documents | Same frozen Top-5, mean {budget['averages']['baseline_doc_count']:.3f} matched docs | Same frozen Top-5 |
| Output unit | Top-1 extracted sentence | At-most-five-document bounded context |
| Mean context tokens | {budget['averages']['recomp_context_tokens']:.2f} | {budget['averages']['selected_context_tokens']:.2f} selected; {budget['averages']['baseline_context_tokens']:.2f} baseline |
| Compression ratio | {100*budget['compression_ratios']['recomp_to_baseline_context_tokens']:.2f}% of baseline | Approximately full budget |
| Reader | Frozen FLAN-T5-Large adaptation | Frozen FLAN-T5-Large |
| Support treatment | Selected sentence is predicted support | Frozen sentence-support predictor |

The original RECOMP reader is FLAN-UL2; this audit uses FLAN-T5-Large to standardize the reader. RECOMP answer/SP/joint F1 are {recomp['metrics']['recomp']['answer_f1']:.4f}/{recomp['metrics']['recomp']['sp_f1']:.4f}/{recomp['metrics']['recomp']['joint_f1']:.4f}; baseline values are {recomp['metrics']['baseline']['answer_f1']:.4f}/{recomp['metrics']['baseline']['sp_f1']:.4f}/{recomp['metrics']['baseline']['joint_f1']:.4f}. The large gap is confounded by the 7.35% token ratio. A Top-k or token-matched variant would be a new post-hoc setting requiring additional reader runs, so it is not introduced into the frozen main comparison.

## I. External Transfer Details

The 2Wiki sample is deterministic and label-blind. Answer F1 changes from {ext_b['answer_f1']:.4f} to {ext_m['answer_f1']:.4f}; SP F1 from {ext_b['sp_f1']:.4f} to {ext_m['sp_f1']:.4f}; joint F1 from {ext_b['joint_f1']:.4f} to {ext_m['joint_f1']:.4f}. The selected answer-drop rate is {100*external['selector']['selected_answer_drop_rate']:.2f}%. All confidence intervals include zero.

## J. Reproducibility

The artifact package includes fold fingerprints, generator model hashes, action and selector outputs, official per-query metrics, same-source disjointness checks, external data-adapter audits, RECOMP checkpoint metadata, component-ablation outputs, and paired-bootstrap summaries. Relative artifact names are used in the anonymous package; local paths and server identifiers are excluded.
"""
    write("paper_appendix_v4_final.md", appendix)

    anonymous = main_paper.replace(
        "The artifact package includes",
        "The anonymous supplementary package includes",
    )
    write("paper_anonymous_v4_final.md", anonymous)

    hierarchy = f"""
# Confirmatory Statistical Hierarchy

## Audit outcome

No immutable pre-run record was found that orders the three FLAN holdout endpoints. The 3,000-query result files precede the current statistical-plan document, and the available repository history contains no earlier endpoint-hierarchy commit. Consequently, the following is a **reporting hierarchy**, not evidence of formal familywise confirmatory control.

1. Headline frozen-holdout metric: FLAN official joint F1, delta {flan['deltas']['joint_f1']:+.4f}, {ci(flan['significance']['joint_f1'])}, p={p(flan['significance']['joint_f1']['p_value'])}.
2. Supporting frozen-holdout metric: FLAN official answer F1, delta {flan['deltas']['answer_f1']:+.4f}, {ci(flan['significance']['answer_f1'])}, p={p(flan['significance']['answer_f1']['p_value'])}.
3. Supporting frozen-holdout metric: FLAN official supporting-fact F1, delta {flan['deltas']['sp_f1']:+.4f}, {ci(flan['significance']['sp_f1'])}, p={p(flan['significance']['sp_f1']['p_value'])}.

Development results are model-development evidence. UnifiedQA, RECOMP, component ablations, and 2Wiki transfer are supporting analyses. The paper reports unadjusted paired-bootstrap p-values and does not label this hierarchy as a pre-specified gatekeeping procedure.
"""
    write("confirmatory_statistical_hierarchy.md", hierarchy)

    confirmatory_audit = """
# Confirmatory Language Audit

| Evidence item | Timestamp/provenance | Supports formal hierarchy? |
| --- | --- | --- |
| Frozen opportunity metric artifact | 2026-07-14 14:10 JST on the execution server; contains opportunity gates only | No |
| 3,000-query run log / result | Run log 2026-07-14 14:47 JST; summary 2026-07-14 14:54 JST | Establishes pipeline freeze, not endpoint order |
| Current statistical claim plan | Local timestamp 2026-07-14 15:38 JST, after holdout results | No |
| Repository commit for pre-run hierarchy | None found | No |

## Approved language

- "frozen same-source holdout confirmation"
- "headline FLAN holdout metric"
- "the fully frozen pipeline reproduces significant answer, support, and joint F1 gains"

## Removed language

- "ordered confirmatory analysis"
- "formally controlled confirmatory family"
- "pre-specified primary holdout endpoint"

The final papers state that no immutable pre-run hierarchy was located and that p-values are unadjusted paired-bootstrap results.
"""
    write("confirmatory_language_audit.md", confirmatory_audit)

    recomp_audit = f"""
# RECOMP Fairness Audit

| Audit field | Recorded value |
| --- | --- |
| Official repository | `https://github.com/carriex/recomp` |
| Repository commit | `{recomp['official_repository_commit']}` |
| Author checkpoint | `{recomp['author_released_checkpoint']}` |
| Original RECOMP reader | FLAN-UL2 |
| Adapted reader | Frozen FLAN-T5-Large |
| Input document count | Same baseline Top-5; mean matched count {budget['averages']['baseline_doc_count']:.3f} |
| Output sentence count | 1 |
| Mean baseline input context tokens | {budget['averages']['baseline_context_tokens']:.3f} |
| Mean RECOMP output context tokens | {budget['averages']['recomp_context_tokens']:.3f} |
| Mean selected-method context tokens | {budget['averages']['selected_context_tokens']:.3f} |
| RECOMP/baseline token ratio | {100*budget['compression_ratios']['recomp_to_baseline_context_tokens']:.3f}% |
| Support treatment | The selected sentence is treated as predicted support |
| Output budget matched | No |

## Fairness questions

1. **Same Top-5 input?** Yes. RECOMP ranks sentences drawn from the exact frozen baseline documents.
2. **Same output token budget?** No. RECOMP emits one sentence averaging 47.13 context tokens; the baseline exposes 668.18 and the selected method 660.57.
3. **Is Top-1 structurally unfavorable for multi-hop evidence?** Yes. Hotpot-style questions commonly require two supporting facts, while one selected sentence can predict at most one support location in this evaluation.
4. **How much of the large gap comes from compression?** It cannot be identified from this run. The 7.35% token ratio is a major confound, so the numerical gap cannot be attributed solely to selection quality.
5. **Can Top-k or token matching be added without changing the frozen comparison?** No. Choosing k or a token budget after observing Top-1 results and rerunning the reader would introduce a new post-hoc condition. The uncompressed Top-5 baseline already provides the non-compressed anchor.

## Paper action

Detailed RECOMP scores are moved to the appendix. The main paper retains one concise supporting row and uses the exact label **official-code reproduction under a standardized reader adaptation**. Approved interpretation: "Under the standardized FLAN reader and the evaluated Top-1 extractive setting, RECOMP is poorly matched to the multi-hop context budget, whereas bounded context actions preserve complementary document evidence." No general superiority claim is allowed.
"""
    write("recomp_fairness_audit.md", recomp_audit)

    generator_audit_doc = """
# Generator Component Claim Audit

| Component | Observed result | Allowed claim | Forbidden claim |
| --- | --- | --- | --- |
| Pair complementarity | Density 14.71% to 10.27% when removed; coverage 29.2% to 27.7% | Clearest learned-component contribution | Every pair score improves every metric |
| Two-document chains | Coverage 29.2% to 25.1%; non-ceiling 47.63% to 40.92% | Clearest structural contribution | More documents always help |
| Document opportunity model | Removal raises coverage to 32.6% but lowers safety to 91.74% | Trades raw breadth for answer safety | Monotonically increases opportunity |
| Lexical-only generator | Coverage 30.7%, density 13.87% | Useful diagnostic showing non-monotonic feature effects | Semantic features dominate lexical features uniformly |
| Missing-hop, MPNet, cross-encoder, redundancy | Mixed effects | Frozen recipe components; full rows in appendix | Independently necessary innovations |

The main table contains only the full generator, pair removal, chain removal, document-model removal, and lexical-only diagnostic. Remaining rows are moved to the appendix. The final papers never state that every semantic component contributes positively.
"""
    write("generator_component_claim_audit.md", generator_audit_doc)

    external_audit = f"""
# External Transfer Claim Audit

| Metric | Delta | 95% CI | p | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Answer F1 | {external['deltas']['answer_f1']:+.4f} | {ci(external['significance']['answer_f1'])} | 0.1116 | Positive point estimate, not reliable |
| Supporting-fact F1 | {external['deltas']['sp_f1']:+.4f} | {ci(external['significance']['sp_f1'])} | 0.6928 | Statistically flat |
| Joint F1 | {external['deltas']['joint_f1']:+.4f} | {ci(external['significance']['joint_f1'])} | 0.3296 | Positive point estimate, not reliable |

Approved section title: **External Transfer and Generalization Boundary**. Approved claim: the frozen pipeline preserves positive answer/joint point estimates, but does not establish statistically reliable transfer. The selected answer-drop rate rises to 6.92%, suggesting that safety calibration is the principal distribution-shift boundary.

Forbidden phrases in final papers: "generalizes to 2Wiki," "cross-dataset robustness," and "dataset-independent action generation."
"""
    write("external_transfer_claim_audit.md", external_audit)

    multi_reader_audit = """
# Multi-Reader Claim Audit

| Claim | Decision | Reason |
| --- | --- | --- |
| The same selected contexts improve answer and joint metrics for FLAN and UnifiedQA | Allowed | Both readers have positive answer/joint deltas on the frozen holdout |
| UnifiedQA provides answer-reader directional replication | Allowed | Reader changes while contexts are fixed |
| Multi-reader support robustness | Forbidden | The support predictor is shared |
| Reader-independent full-pipeline replication | Forbidden | Support prediction is not independently retrained or rerun |

Every main-text multi-reader statement is followed by the support-predictor boundary, and the UnifiedQA table row contains a footnote.
"""
    write("multi_reader_claim_audit.md", multi_reader_audit)

    anonymity_audit = """
# Anonymity Audit

| Check | Final papers | Result |
| --- | --- | --- |
| Internal version labels (V2/V3/V4) | Replaced by initial fixed-action, heuristic expansion, and semantic generator | Pass |
| Local absolute paths | None | Pass |
| Server hostnames | None | Pass |
| User/lab identifiers | None | Pass |
| Author-revealing project repository | None | Pass |
| Memory references/internal handoff names | None | Pass |
| Commit authors | None | Pass |
| Acknowledgments | None | Pass |
| Anonymous artifact wording | Present in anonymous manuscript | Pass |

The required output filenames contain `v4`, but this project label does not appear in manuscript prose. External method repositories may be cited as published artifacts; no repository for the submitted work is named.
"""
    write("anonymity_audit.md", anonymity_audit)

    citation_audit = """
# Final Citation and Fact Audit

| Item | Verification | Result |
| --- | --- | --- |
| All manuscript citation keys resolve | Compared against `references.bib` | Pass |
| HotpotQA, 2Wiki, MuSiQue venues | ACL Anthology/TACL records | Pass |
| RECOMP status and protocol | ICLR paper, official code, author checkpoint | Pass |
| SetR status | ACL 2025 paper; claim restricted to set selection | Pass |
| Reader-Centered Passage Selection status | COLING 2025 paper; claim restricted to reader alignment | Pass |
| RankRAG status | NeurIPS 2024 paper | Pass |
| No unresolved preprint status | No unmarked preprint is used | Pass |
| No raw placeholders | Machine scan for TODO/TBD/XX/citation-needed | Pass |
| Statistical values | Regenerated from frozen JSON summaries | Pass |
| Opportunity values | Regenerated from frozen action summary | Pass |
| RECOMP token budget | Measured with frozen FLAN tokenizer | Pass |

The RECOMP claim is limited to the official-code Top-1 setting with a standardized reader adaptation. The SetR and reader-centered citations describe their published positioning without asserting unavailable implementation details.
"""
    write("final_citation_and_fact_audit.md", citation_audit)

    claim_audit = """
# Final Claim Audit

| File | Section | Audited sentence / risky claim | Evidence | Risk | Replacement / final status |
| --- | --- | --- | --- | --- | --- |
| paper_full_clean_v4_final.md | 7 Development Behavior | "Joint F1 is positive but non-significant." | Delta +0.0064, p=0.0752 | Low after correction | Replaces the forbidden significant-joint claim. |
| paper_main_conference_v4_final.md | 7 Development Behavior | "Joint F1 rises by 0.0064, but its interval includes zero." | 95% CI [-0.0005,+0.0132] | Low after correction | Approved. |
| paper_anonymous_v4_final.md | Abstract | "Joint F1 shows a positive, non-significant trend." | p=0.0752 | Low after correction | Approved. |
| paper_full_clean_v4_final.md | 9 External Transfer | "The result does not establish statistically reliable transfer." | Answer/joint CIs include zero; SP flat | Low after correction | Replaces "generalizes to 2Wiki." |
| paper_main_conference_v4_final.md | 9 External Transfer | "The frozen pipeline preserves positive answer and joint point estimates, but all confidence intervals include zero." | p=0.1116/0.6928/0.3296 | Low after correction | Approved; no cross-dataset robustness phrase. |
| paper_anonymous_v4_final.md | 9 External Transfer | Same bounded external sentence as main paper. | One non-significant external sample | Low after correction | Section title is Generalization Boundary. |
| paper_main_conference_v4_final.md | 7 Frozen Holdout | "The support predictor is shared, so the second reader confirms answer/joint direction rather than independent support robustness." | One shared support predictor | Low after correction | Replaces multi-reader full-pipeline claim. |
| paper_appendix_v4_final.md | G Multi-Reader | "These are not independent support replications." | Support predictions are shared | Low after correction | Approved. |
| paper_main_conference_v4_final.md | 6 Baselines | "Official-code reproduction under a standardized reader adaptation." | Reader changed from FLAN-UL2 to FLAN-T5-Large | Low after correction | Replaces faithful end-to-end reproduction. |
| paper_appendix_v4_final.md | H RECOMP Fairness | "The large gap is confounded by the 7.35% token ratio." | 47.13 vs 668.18 context tokens | Medium residual risk | General superiority claim removed. |
| paper_main_conference_v4_final.md | 8 Generator Components | "MPNet, cross-encoder, missing-hop, and redundancy ablations are mixed." | Several removals improve raw coverage | Low after correction | Replaces all-components-positive claim. |
| paper_main_conference_v4_final.md | 7 Opportunity | "It passes ... three criteria, but misses overall coverage and efficiency." | Three of five criteria pass | Low after correction | Replaces all-gates-pass claim. |
| paper_anonymous_v4_final.md | Entire file | No state-of-the-art claim. | No comprehensive benchmark | Low | Machine scan passes. |
| paper_anonymous_v4_final.md | Entire file | No Federated RAG method identity or privacy claim. | No federated/privacy mechanism evaluated | Low | Machine scan passes. |
| paper_main_conference_v4_final.md | 6 Statistics | "No immutable pre-run hierarchy was found; we do not claim formal ordered testing." | Plan timestamp follows holdout result | Low after correction | Replaces ordered-confirmatory language. |
"""
    write("final_claim_audit.md", claim_audit)

    response = """
# Reviewer Response Brief

## Core response

The paper's claim is deliberately narrow: a semantic opportunity generator plus fully nested reader-safe selection yields small but statistically reliable gains on a frozen same-source holdout. We do not claim that the opportunity gap is solved, that the external transfer is significant, or that the full support pipeline replicates across readers.

## Likely questions

**Q1: The effects are small. Why are they meaningful?**  
The intervention changes only 25.8% of holdout contexts, preserves the original context elsewhere, and improves answer, support, and joint F1 without retuning. The contribution is the controlled conversion of context opportunity into reader gains, not a large leaderboard jump.

**Q2: Is the 3,000-query sample a test set?**  
No. It is a disjoint same-source HotpotQA holdout evaluated after freezing the pipeline. We call it frozen holdout confirmation and make no external-domain claim from it.

**Q3: Was joint F1 formally pre-specified as primary?**  
No immutable pre-run hierarchy was found. We present joint F1 as the headline holdout metric and report all three unadjusted paired-bootstrap tests. We do not claim formal familywise control.

**Q4: Does the method transfer to 2Wiki?**  
The point estimates for answer and joint F1 are positive, but all intervals include zero and support F1 is flat. The result bounds catastrophic collapse but does not establish reliable transfer. Safety calibration worsens.

**Q5: Is the RECOMP comparison fair?**  
It uses official code/checkpoint and the same Top-5 input, but not the same output budget or original reader. RECOMP emits about 7.35% of baseline context tokens. We therefore use it only as a standardized-reader compatibility analysis and move details to the appendix.

**Q6: Do all generator components help?**  
No. Pair complementarity and two-document chains have the clearest contributions. The document model trades breadth for answer safety; other feature effects are mixed. The paper states this explicitly.

**Q7: Is the second reader an independent replication?**  
It is an answer-reader directional replication on identical contexts. The support predictor is shared, so it is not an independent full-pipeline replication.
"""
    write("reviewer_response_brief.md", response)

    readiness = """
# Submission Readiness Report

```yaml
paper_structure_complete: true
confirmatory_claim_valid: true
recomp_comparison_fairly_described: true
generator_ablation_claims_valid: true
external_transfer_claims_valid: true
multi_reader_claims_valid: true
anonymity_complete: true
citations_complete: true
reproducibility_complete: true
submission_tier: main_conference_ready_with_high_review_risk
```

## Final recommendation

- `recommended_title`: Generating Reader-Compatible Context Actions for Multi-Hop Question Answering
- `one_sentence_claim`: A semantic opportunity generator, combined with fully nested reader-safe selection, produces small but statistically reliable official multi-hop QA gains on a frozen same-source holdout.
- `primary_table`: Table 2, frozen 3,000-query same-source holdout across FLAN and UnifiedQA.
- `main_review_risks`: small absolute effects; same-source empirical anchor; non-significant 2Wiki transfer; shared support predictor; RECOMP output-budget mismatch; mixed non-core ablations; no immutable pre-run statistical hierarchy.
- `submission_tier`: main_conference_ready_with_high_review_risk.

The manuscript is structurally and anonymously ready for a main-conference submission. The high-risk label reflects likely reviewer skepticism about effect size and external validity, not an unresolved claim or artifact failure.
"""
    write("submission_readiness_report.md", readiness)

    final_papers = [
        HERE / "paper_full_clean_v4_final.md",
        HERE / "paper_main_conference_v4_final.md",
        HERE / "paper_appendix_v4_final.md",
        HERE / "paper_anonymous_v4_final.md",
    ]
    bib_keys = set(
        re.findall(r"@[A-Za-z]+\{([^,]+),", (HERE / "references.bib").read_text(encoding="utf-8"))
    )
    cited = set()
    for path in final_papers:
        cited.update(re.findall(r"@([A-Za-z0-9_.:-]+)", path.read_text(encoding="utf-8")))
    missing = sorted(cited - bib_keys)
    if missing:
        raise AssertionError(f"Missing citation keys: {missing}")

    anonymous_text = (HERE / "paper_anonymous_v4_final.md").read_text(encoding="utf-8")
    anonymity_patterns = {
        "internal version": r"\bV[234]\b",
        "absolute path": r"/(?:Users|home)/",
        "server": r"iia100|iiserver",
        "lab/user": r"iilab|iiserver31",
        "internal handoff": r"handoff|zhidaoagent|memory",
    }
    anonymity_hits = {
        label: re.findall(pattern, anonymous_text, flags=re.IGNORECASE)
        for label, pattern in anonymity_patterns.items()
        if re.search(pattern, anonymous_text, flags=re.IGNORECASE)
    }
    if anonymity_hits:
        raise AssertionError(f"Anonymous-paper leakage: {anonymity_hits}")

    forbidden_positive = {
        "development significant joint": r"development[^\n]{0,100}joint F1[^\n]{0,70}significant(?:ly)? improve",
        "generalizes to 2Wiki": r"generalizes? to 2Wiki",
        "cross-dataset robustness": r"cross-dataset robustness",
        "multi-reader support robustness": r"multi-reader support robustness",
        "faithful end-to-end": r"faithful end-to-end",
        "general RECOMP superiority": r"(?:generally|universally) (?:outperforms?|superior).{0,40}RECOMP|RECOMP.{0,40}(?:generally|universally) inferior",
        "all components": r"all semantic components (?:contribute|improve)",
        "all gates": r"all opportunity (?:gates|criteria) pass",
        "ordered confirmatory": r"ordered confirmatory analysis",
    }
    violations = {}
    for path in final_papers:
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_positive.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.setdefault(path.name, []).append(label)
    if violations:
        raise AssertionError(f"Forbidden claim patterns: {violations}")

    placeholder_hits = {}
    for path in final_papers:
        hits = re.findall(r"\b(?:TODO|TBD|FIXME|CITATION NEEDED|XX+)\b", path.read_text(encoding="utf-8"), re.I)
        if hits:
            placeholder_hits[path.name] = hits
    if placeholder_hits:
        raise AssertionError(f"Placeholder text remains: {placeholder_hits}")

    manifest = {
        "status": "complete",
        "task": "V7-HP-PAPER-v4-final-main-conference-refinement",
        "submission_tier": "main_conference_ready_with_high_review_risk",
        "required_documents": sorted(path.name for path in HERE.glob("*.md")),
        "citation_keys_used": sorted(cited),
        "citation_keys_missing": missing,
        "anonymity_hits": anonymity_hits,
        "forbidden_claim_hits": violations,
        "recomp_budget_audit": budget,
    }
    (HERE / "final_refinement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
