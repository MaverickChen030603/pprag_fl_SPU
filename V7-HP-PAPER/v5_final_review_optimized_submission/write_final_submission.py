#!/usr/bin/env python3
"""Write the final review-optimized paper and response from frozen evidence."""

from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
V4 = PROJECT / "V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
V4_REFINED = PROJECT / "V7-HP-PAPER/v4_final_submission_refinement"
V4_COMPLETE = PROJECT / "V7-HP-PAPER/v4_submission_completion"
V5 = PROJECT / "V7-HP-PAPER/review_driven_revision_v5"
OUT = HERE / "outputs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def clean(value: str) -> str:
    dedented = textwrap.dedent(value).strip()
    return "\n".join(line[4:] if line.startswith("    ") else line for line in dedented.splitlines())


def fmt_p(value: float) -> str:
    return "<.0002" if value == 0 else f"{value:.4f}"


def ci(row: dict[str, Any], v4: bool = False) -> str:
    low = row["ci95_low"] if v4 else row["ci_low"]
    high = row["ci95_high"] if v4 else row["ci_high"]
    return f"[{low:+.4f}, {high:+.4f}]"


def metric_value(row: dict[str, Any], v4: bool) -> float:
    return float(row["mean"] if v4 else row["delta"])


def main() -> None:
    holdouts = read_json(OUT / "tables/two_frozen_same_source_holdouts.json")
    selected = read_json(OUT / "selected_effect/selected_effect_distribution.json")
    cost = read_json(OUT / "cost/frozen_end_to_end_latency.json")
    recomp = read_json(OUT / "audits/recomp_fairness.json")
    transfer = read_json(OUT / "audits/external_transfer.json")
    lite = read_json(V5 / "outputs/lite_model/lite_holdout_metrics.json")
    original_summary = read_json(V4 / "outputs/scaleup/scaleup_summary.json")
    external_zero = read_json(V4_COMPLETE / "outputs/external_2wiki_frozen/external_validation_results.json")

    costs = cost["systems"]
    full_cost = costs["full_v4"]
    base_cost = costs["frozen_top5_baseline"]
    lite_cost = costs["lite_lexical_pair"]
    full_ms = 1000 * full_cost["end_to_end_post_retrieval_latency"]["mean_seconds"]
    base_ms = 1000 * base_cost["end_to_end_post_retrieval_latency"]["mean_seconds"]
    lite_ms = 1000 * lite_cost["end_to_end_post_retrieval_latency"]["mean_seconds"]
    truncated_ms = 1000 * costs["baseline_truncated_660"]["end_to_end_post_retrieval_latency"]["mean_seconds"]
    recomp_660_ms = 1000 * costs["recomp_budgetmatched"]["end_to_end_post_retrieval_latency"]["mean_seconds"]
    full_p95 = 1000 * full_cost["end_to_end_post_retrieval_latency"]["p95_seconds"]
    base_p95 = 1000 * base_cost["end_to_end_post_retrieval_latency"]["p95_seconds"]
    full_overhead = full_ms - base_ms
    full_ratio = full_ms / base_ms
    title = "Pair-Complementary Context Construction with Reader-Safe Selection for Multi-Hop QA"
    claim = "Pair-complementary action generation and fully nested reader-safe selection yield modest but reproducible same-source QA gains, with larger descriptive effects on selected interventions and unresolved cost and transfer boundaries."

    abstract = clean(f"""
    Multi-hop question answering depends not only on retrieving relevant passages but also on constructing a context that exposes complementary evidence to a fixed reader. We study the **candidate-opportunity gap**: a selector cannot recover a reader-compatible context when its candidate actions omit the needed evidence combination. Our Full method scores pair complementarity, constructs bounded two-document chains, preserves answer-bearing anchors, and applies an action only through a fully nested two-head reader-safe selector. On a frozen 3,000-query HotpotQA holdout, Full improves Answer, supporting-fact (SP), and Joint F1 by +0.0088, +0.0056, and +0.0064. An untouched 3,405-query same-source holdout confirms gains of +0.0116, +0.0061, and +0.0080. On policy-selected interventions, the corresponding descriptive gains are +0.0340/+0.0219/+0.0250 on the first holdout; these conditional values are not causal effects and accompany, rather than replace, the modest population results. A frozen Lite simplification fails a 0.002 Joint-F1 non-inferiority criterion, so Full remains primary. Under an approximately matched 660-token budget, adapted RECOMP sentence packing does not improve the frozen baseline. Measured post-retrieval inference is {full_ms:.1f} ms/query for Full versus {base_ms:.1f} ms/query for the baseline, with one final reader call in both. Finally, 2Wiki calibration lowers selected answer-drop from 6.92% to 5.10% but misses the pre-specified 4% target. The evidence supports bounded same-source context construction, not broad efficiency or transfer claims.
    """)
    word_count = len(re.findall(r"\b[\w+-]+\b", abstract))
    if not 180 <= word_count <= 230:
        raise AssertionError(f"Abstract word count {word_count} is outside 180-230")

    introduction = clean(f"""
    ## 1. Introduction

    Multi-hop question answering (QA) is often framed as finding relevant documents, but a reader consumes an ordered and budget-limited context rather than an unordered relevance set. To answer correctly, that context must expose complementary hops, retain the passage that gives the answer its lexical form, and place the evidence where the reader can use it. More retrieval is therefore not automatically better: adding an individually relevant document can displace an answer anchor or leave two necessary facts disconnected.

    This creates a structural limit for post-retrieval selection. A selector can choose only among the contexts proposed by its action generator. If no proposal contains a reader-compatible repair, improved selection cannot help. We call this mismatch the **candidate-opportunity gap**. It is observable as a low density of actions that improve reader outcomes without damaging answer quality.

    Fixed expansion templates do not reliably close this gap. Independent insertions and replacements often select documents that are relevant to the question but redundant with each other. Unrestricted replacement can also remove the passage that supplies the answer wording. In our frozen development analyses, increasing action count alone yielded little additional safe positive-query coverage.

    We instead organize context construction around **pair complementarity**. The Full Pair-Complementary Action Generator models whether two documents supply different parts of a multi-hop chain, constructs bounded two-document actions, and preserves high-value baseline anchors. Its implementation combines lexical, MPNet, and cross-encoder signals with missing-hop, document-opportunity, and pair-complementarity models. These modules form the empirically stronger Full recipe; our conceptual claim is narrower than a claim that each feature helps monotonically.

    A separate reader-safe selector decides whether any generated action should reach the reader. One head predicts answer safety and another predicts positive utility. Frozen thresholds and a coverage budget allow selective intervention; otherwise the system returns the original Top-5 context exactly. The online reader runs once, after this decision, rather than once per candidate action.

    We use a fully nested five-fold protocol. Generator and selector training occurs on outer-training queries, thresholds are set using inner out-of-fold predictions, and each outer-test query is processed only by frozen components. We then evaluate the frozen Full system on two disjoint same-source holdouts: 3,000 original holdout queries and an untouched 3,405-query revision holdout. Full improves Answer, SP, and Joint F1 on both. The absolute population changes are modest, while direct paired accounting shows larger descriptive changes on the 25-26% of queries where the policy intervenes and exactly zero change on fallbacks.

    The revision experiments also define what we do not claim. A development-frozen Lite-Lexical-Pair simplification fails the independent non-inferiority test, so it does not replace Full. A budget-matched compression comparison is reported as a difference between context-construction objectives, not as a universal rank ordering. Frozen transfer to 2Wiki is non-significant and its few-shot gate calibration misses the safety target. Finally, Full is {full_ratio:.2f}x the measured baseline post-retrieval latency, so the result is a quality-risk trade-off over a bounded candidate pool rather than an unqualified deployment claim.

    Our contributions are:

    1. We formulate the candidate-opportunity gap for reader-aware multi-hop context construction.
    2. We introduce pair-complementary, anchor-preserving action generation with bounded two-document chains.
    3. We combine the generator with fully nested reader-safe selective intervention and exact fallback.
    4. We provide two frozen same-source confirmations plus conditional-effect, non-inferiority, budget-matched compression, online cost, and transfer-boundary analyses.
    """)

    related = clean("""
    ## 2. Related Work

    **Multi-hop retrieval and QA.** HotpotQA and 2WikiMultiHopQA pair answers with supporting facts, making it possible to distinguish answer generation from evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Multi-step retrievers such as MDR acquire evidence across retrieval steps [@xiong-etal-2021-mdr]. Our setting starts later: a frozen retriever has already produced a bounded candidate pool, and the method reorganizes that pool for a fixed reader.

    **Reader-aware context construction.** Reader-aware retrieval and reranking account for downstream behavior beyond independent query-document relevance [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Prior analyses show that distractors and evidence position can alter reader output [@shi-etal-2023-distracted; @liu-etal-2024-lost]. We distinguish action opportunity from action selection: a reader-safe selector remains limited by the combinations exposed by its generator.

    **Compression and selective prediction.** RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Because its released Hotpot configuration and our near-full contexts have different budgets and objectives, we use the author-released compressor under a common FLAN reader and include a 660-token condition plus a source-order truncation control. Selective prediction motivates fallback when estimated risk is high [@geifman-elyaniv-2019-selectivenet]. Here fallback means preserving the frozen retrieval baseline, and offline reader outcomes supervise the gate without adding online candidate-reader calls.
    """)

    method = clean("""
    ## 3. Method

    ### 3.1 Problem and Opportunity

    For a question $q$, a frozen retriever returns a bounded document pool $D_q$ and ordered Top-5 baseline $C_0(q)$. A context action maps $C_0$ to another five-document sequence using only documents in $D_q$ and without editing source text. The generator exposes a finite action set $A(q)$; the selector either chooses one action or returns $C_0$.

    During training only, an action is answer-safe when the frozen reader's Answer F1 is no lower than on $C_0$ and positive when it improves the reader-side joint answer-evidence utility. Query opportunity is the existence of at least one safe positive action in $A(q)$. It is an empirical upper bound for any selector restricted to that set, not an inference-time label.

    ### 3.2 Full Pair-Complementary Action Generator

    #### 3.2.1 Candidate document signals

    The Full generator computes normalized BM25, question-title and question-text overlap, named-entity overlap, bridge overlap with baseline documents, novelty, and redundancy. It augments these lexical signals with cached MPNet query-document similarities [@song-etal-2020-mpnet] and cross-encoder relevance. At test time these features use only the question, candidate text, baseline ordering, and learned parameters; answers, support annotations, and reader outcomes are absent.

    #### 3.2.2 Missing-hop and document-opportunity modules

    A missing-hop estimator summarizes which query and baseline signals remain weakly represented. A document-opportunity model then scores whether a candidate can fill that estimated gap while adding nonredundant information. Both models are trained inside each outer fold from training-query outcomes. They are components of the empirically stronger Full implementation, not independently established monotonic contributions.

    #### 3.2.3 Pair complementarity

    Individual relevance cannot determine whether two documents jointly supply different hops. For each pair among the top candidate set, the generator constructs features from their individual scores, entity-chain overlap, combined novelty, redundancy, and relation to the missing-hop state. A balanced pair classifier estimates complementarity. With at most $L$ candidate documents, pair scoring is bounded by $L(L-1)/2$; the frozen deployment uses ten pair scores per query.

    #### 3.2.4 Bounded two-document chain construction

    High-scoring complementary pairs form bounded two-document actions. The pair is inserted into weak tail positions of the five-document baseline, producing a compact chain rather than an unconstrained search over permutations. The generator also retains a small single-complementary-insertion family. Duplicate contexts are removed, and the candidate action count is capped before selection.

    #### 3.2.5 Anchor preservation and action pruning

    The constructor protects the strongest early baseline anchors whenever the budget permits. This prevents an apparently useful support insertion from deleting the passage that supplies answer wording. Actions that duplicate a context, violate the five-document budget, or rank below the frozen pruning rule are discarded. Fallback is always present.

    ### 3.3 Reader-Safe Selector

    The selector has two balanced logistic heads. The safety head estimates whether an action preserves baseline answer quality; the utility head estimates whether it yields a positive reader-side change. Inner out-of-fold predictions determine both thresholds and a 10-30% coverage budget. On an outer-test query, an action is eligible only if it passes both heads. The highest-ranked eligible action is applied while the fold-level budget remains; all other queries receive the exact baseline. The reader then evaluates one final context.

    ### 3.4 Fully Nested Training and Evaluation

    Five outer folds separate training from evaluation. Generator modules and selector heads fit only outer-training queries. Inner folds tune thresholds and coverage without reading outer-test outcomes. Each outer-test query is processed by fold-specific frozen models. The 3,000-query and 3,405-query holdouts are disjoint from the 1,000 development queries, and no holdout outcome selects an architecture or threshold.

    ### 3.5 Review-Driven Lite Simplification

    Lite-Lexical-Pair removes MPNet, cross-encoder, missing-hop, and document-opportunity computation while retaining lexical pair complementarity, bounded chains, anchors, and the two-head selector. Its architecture is selected on development and frozen before the revision holdout is opened. The pre-specified Joint-F1 non-inferiority margin is 0.002. Lite fails this independent test, so Full remains the primary method. The experiment narrows the conceptual explanation but does not show that Full's removed semantic modules are dispensable.
    """)

    setup = clean("""
    ## 4. Experimental Setup

    We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. A fixed 1,000-query development slice supports nested training and threshold selection. The next disjoint 3,000 queries form the original confirmatory holdout; the remaining 3,405 form an untouched revision holdout. All retain the same source distribution and are not external-domain tests.

    The frozen upstream baseline is HybridSoftRetriever with alpha 0.55, uniform document weights, and Top-5 output. FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5], using greedy decoding, at most 32 generated tokens, a 1,024-token input limit, and context capped at 3,200 characters. A Hotpot-development support predictor uses a frozen 0.7 threshold. We report official Answer, supporting-fact (SP), and Joint EM/F1. Paired 95% intervals and two-sided p-values use 5,000 query-level bootstrap resamples.

    For RECOMP, the author-released HotpotQA compressor scores sentences in the same Top-5 input [@xu-etal-2024-recomp]. Development budgets are 64, 128, 256, 384, 512, and 660 FLAN tokens; 660 is frozen before holdout evaluation. Baseline-Truncated retains source sentence order at the same budget. All systems share reader, prompt, support predictor, and metric code.

    For online cost, all systems run on one GPU with batch size one over the same ordered queries. We use 50 warmup queries and measure the next 500, synchronizing CUDA around every component. Model loading is excluded. Online features/actions are recomputed and their final context must exactly match the frozen artifact. Candidate outcome labeling and training are offline.
    """)

    holdout_lines = [
        "| Split | N | Coverage | Baseline A-F1 | Full A-F1 | A delta (95% CI; p) | SP delta (95% CI; p) | Joint delta (95% CI; p) | Selected A-drop |",
        "|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for row in holdouts["rows"]:
        v4 = row["statistics_schema"] == "v4"
        s = row["statistics"]
        holdout_lines.append(
            f"| {row['split']} | {row['n']} | {row['coverage']:.1%} | {row['baseline']['answer_f1']:.4f} | {row['full']['answer_f1']:.4f} | {metric_value(s['answer_f1'], v4):+.4f} ({ci(s['answer_f1'], v4)}; {fmt_p(s['answer_f1']['p_value'])}) | {metric_value(s['sp_f1'], v4):+.4f} ({ci(s['sp_f1'], v4)}; {fmt_p(s['sp_f1']['p_value'])}) | {metric_value(s['joint_f1'], v4):+.4f} ({ci(s['joint_f1'], v4)}; {fmt_p(s['joint_f1']['p_value'])}) | {row['answer_drop_rate_selected']:.2%} |"
        )
    original_effect = selected["original_holdout_3000"]
    revision_effect = selected["revision_holdout_3405"]
    results = clean(f"""
    ## 5. Main Results

    ### 5.1 Two Frozen Same-Source Holdouts

    {chr(10).join(holdout_lines)}

    Full improves all three F1 measures on both holdouts. On the original 3,000-query holdout, the paired deltas are +0.0088 Answer, +0.0056 SP, and +0.0064 Joint F1. The untouched 3,405-query holdout confirms +0.0116, +0.0061, and +0.0080. The latter simultaneously serves as the independent Lite non-inferiority test and a replication of Full. Both sets are disjoint from development, Full was frozen before both runs, and revision outcomes were unread when the Lite architecture was fixed. Because both are HotpotQA same-source samples, we do not pool them for a new significance claim.

    ### 5.2 Descriptive Effects on Policy-Selected Interventions

    Population and conditional views answer different questions. In the original holdout, Full edits {original_effect['selected_n']}/{original_effect['n_queries']} contexts ({original_effect['coverage']:.1%}); Answer/SP/Joint population deltas are {original_effect['metrics']['answer_f1']['population_mean_delta']:+.4f}/{original_effect['metrics']['sp_f1']['population_mean_delta']:+.4f}/{original_effect['metrics']['joint_f1']['population_mean_delta']:+.4f}. Conditional on these policy-selected interventions, the descriptive means are {original_effect['metrics']['answer_f1']['selected_mean_delta']:+.4f}/{original_effect['metrics']['sp_f1']['selected_mean_delta']:+.4f}/{original_effect['metrics']['joint_f1']['selected_mean_delta']:+.4f}. Answer has {original_effect['metrics']['answer_f1']['selected_wins']} wins, {original_effect['metrics']['answer_f1']['selected_losses']} losses, and {original_effect['metrics']['answer_f1']['selected_ties']} ties; Joint has {original_effect['metrics']['joint_f1']['selected_wins']}/{original_effect['metrics']['joint_f1']['selected_losses']}/{original_effect['metrics']['joint_f1']['selected_ties']}. The selected Answer- and Joint-drop rates are {original_effect['metrics']['answer_f1']['selected_drop_rate']:.2%} and {original_effect['metrics']['joint_f1']['selected_drop_rate']:.2%}. Medians and both interquartile endpoints are zero because most selected contexts tie the baseline.

    The revision holdout shows the same concentration pattern: {revision_effect['selected_n']}/{revision_effect['n_queries']} interventions, with descriptive selected Answer/SP/Joint means of {revision_effect['metrics']['answer_f1']['selected_mean_delta']:+.4f}/{revision_effect['metrics']['sp_f1']['selected_mean_delta']:+.4f}/{revision_effect['metrics']['joint_f1']['selected_mean_delta']:+.4f}. These values are descriptive gains conditional on policy-selected interventions. They are not causal treatment effects, expected gains for arbitrary queries, or effects on all improvable queries. In both holdouts, fallback contexts and metrics are exactly unchanged.

    ### 5.3 Full-to-Lite Non-Inferiority

    | Revision-holdout system | Answer F1 | SP F1 | Joint F1 |
    |---|---:|---:|---:|
    | Frozen Top-5 | {lite['metrics']['frozen_top5_baseline']['answer_f1']:.4f} | {lite['metrics']['frozen_top5_baseline']['sp_f1']:.4f} | {lite['metrics']['frozen_top5_baseline']['joint_f1']:.4f} |
    | Full | {lite['metrics']['full_v4']['answer_f1']:.4f} | {lite['metrics']['full_v4']['sp_f1']:.4f} | {lite['metrics']['full_v4']['joint_f1']:.4f} |
    | Lite-Lexical-Pair | {lite['metrics']['lite_lexical_pair']['answer_f1']:.4f} | {lite['metrics']['lite_lexical_pair']['sp_f1']:.4f} | {lite['metrics']['lite_lexical_pair']['joint_f1']:.4f} |

    Lite minus Full Joint F1 is {lite['lite_noninferiority']['joint_f1_delta']['delta']:+.4f} (95% CI {ci(lite['lite_noninferiority']['joint_f1_delta'])}, p={fmt_p(lite['lite_noninferiority']['joint_f1_delta']['p_value'])}). It misses both the point and interval versions of the frozen 0.002 margin. Lite reduces computation, but the independent quality criterion fails; it is therefore a simplification diagnostic rather than a replacement for Full.
    """)

    rmetrics = recomp["metrics"]
    rsig = recomp["recomp_660_vs_baseline"]
    recomp_section = clean(f"""
    ## 6. Budget-Matched Compression

    | System (3,000 holdout) | Tokens | Documents | Answer F1 | SP F1 | Joint F1 | E2E ms/query |
    |---|---:|---:|---:|---:|---:|---:|
    | Frozen Top-5 | {rmetrics['frozen_top5_baseline']['context_tokens']:.1f} | {rmetrics['frozen_top5_baseline']['represented_documents']:.3f} | {rmetrics['frozen_top5_baseline']['answer_f1']:.4f} | {rmetrics['frozen_top5_baseline']['sp_f1']:.4f} | {rmetrics['frozen_top5_baseline']['joint_f1']:.4f} | {base_ms:.2f} |
    | Baseline-Truncated-660 | {rmetrics['baseline_truncated_660']['context_tokens']:.1f} | {rmetrics['baseline_truncated_660']['represented_documents']:.3f} | {rmetrics['baseline_truncated_660']['answer_f1']:.4f} | {rmetrics['baseline_truncated_660']['sp_f1']:.4f} | {rmetrics['baseline_truncated_660']['joint_f1']:.4f} | {truncated_ms:.2f} |
    | RECOMP-660 | {rmetrics['recomp_budget_660']['context_tokens']:.1f} | {rmetrics['recomp_budget_660']['represented_documents']:.3f} | {rmetrics['recomp_budget_660']['answer_f1']:.4f} | {rmetrics['recomp_budget_660']['sp_f1']:.4f} | {rmetrics['recomp_budget_660']['joint_f1']:.4f} | {recomp_660_ms:.2f} |
    | Full | {rmetrics['full_v4']['context_tokens']:.1f} | {rmetrics['full_v4']['represented_documents']:.3f} | {rmetrics['full_v4']['answer_f1']:.4f} | {rmetrics['full_v4']['sp_f1']:.4f} | {rmetrics['full_v4']['joint_f1']:.4f} | {full_ms:.2f} |

    RECOMP-660 changes Answer/SP/Joint F1 relative to Frozen Top-5 by {rsig['answer_f1']['delta']:+.4f}/{rsig['sp_f1']['delta']:+.4f}/{rsig['joint_f1']['delta']:+.4f}. The Joint interval is {ci(rsig['joint_f1'])}, p={fmt_p(rsig['joint_f1']['p_value'])}; the difference is not significant. Under an approximately matched context budget and a standardized FLAN reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline, whereas Full context actions retain a positive same-source effect. This is an official-compressor implementation under reader and budget adaptation, not a claimed end-to-end reproduction. Matched tokens also do not create identical structural action spaces: sentence compression and pair-complementary five-document actions optimize different objectives. The approximately 47-token Top-1 condition is retained only as a compatibility diagnostic in the appendix.
    """)

    cost_rows = []
    cost_labels = {"frozen_top5_baseline": "Frozen Top-5", "full_v4": "Full", "lite_lexical_pair": "Lite", "baseline_truncated_660": "Baseline-Truncated-660", "recomp_top1": "RECOMP Top1", "recomp_budgetmatched": "RECOMP 660"}
    for system in ("frozen_top5_baseline", "full_v4", "lite_lexical_pair", "baseline_truncated_660", "recomp_top1", "recomp_budgetmatched"):
        row = costs[system]
        cost_rows.append(
            f"| {cost_labels[system]} | {1000*row['generator_only_latency']['mean_seconds']:.2f} | {1000*row['selector_only_latency']['mean_seconds']:.2f} | {1000*row['reader_only_latency']['mean_seconds']:.2f} | {1000*row['end_to_end_post_retrieval_latency']['mean_seconds']:.2f} | {1000*row['end_to_end_post_retrieval_latency']['p95_seconds']:.2f} | {row['cross_encoder_document_scores_per_query']} | {row['final_reader_calls_per_query']} | {row['peak_gpu_memory_bytes']/2**30:.2f} |"
        )
    cost_section = clean(f"""
    ## 7. Computational Cost and Deployment Boundary

    | System | Generator ms | Selector ms | Reader ms | Total ms | P95 total | Cross scores | Reader calls | Peak GiB |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|
    {chr(10).join(cost_rows)}

    These are measured end-to-end **post-retrieval** times, not reader-only proxies. The shared protocol uses one GPU, batch size one, 50 warmup queries, 500 measured queries, CUDA synchronization, and the same query fingerprint. All online components are recomputed; every final context matches its frozen artifact. Model loading is excluded.

    Full's mean is {full_ms:.2f} ms/query versus {base_ms:.2f} ms/query for Frozen Top-5, an overhead of {full_overhead:.2f} ms and a {full_ratio:.2f}x ratio. Lite lowers the mean to {lite_ms:.2f} ms by removing semantic encoders, but its independent non-inferiority failure prevents promotion to the main method. Every system invokes the answer reader exactly once on the final context. Candidate reader outcomes are offline labels and do not add online reader calls.

    Offline work includes action-outcome labeling and fold-specific generator and selector training. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and production streaming are outside the evaluation.
    """)

    zero = external_zero
    best = transfer["best_calibrated"]
    external = clean(f"""
    ## 8. External Transfer and Calibration

    ### 8.1 Zero-Shot Frozen Transfer

    On 1,000 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki], the unchanged Hotpot gate covers {zero['selector']['coverage']:.1%}. Baseline Answer/SP/Joint F1 are {zero['metrics']['baseline']['answer_f1']:.4f}/{zero['metrics']['baseline']['sp_f1']:.4f}/{zero['metrics']['baseline']['joint_f1']:.4f}; frozen transfer yields {zero['metrics']['v4_frozen_transfer']['answer_f1']:.4f}/{zero['metrics']['v4_frozen_transfer']['sp_f1']:.4f}/{zero['metrics']['v4_frozen_transfer']['joint_f1']:.4f}. The deltas are {zero['deltas']['answer_f1']:+.4f}/{zero['deltas']['sp_f1']:+.4f}/{zero['deltas']['joint_f1']:+.4f}. Answer has 95% CI [{zero['significance']['answer_f1']['ci95_low']:+.4f}, {zero['significance']['answer_f1']['ci95_high']:+.4f}], p={zero['significance']['answer_f1']['p_value']:.4f}; SP has [{zero['significance']['sp_f1']['ci95_low']:+.4f}, {zero['significance']['sp_f1']['ci95_high']:+.4f}], p={zero['significance']['sp_f1']['p_value']:.4f}; Joint has [{zero['significance']['joint_f1']['ci95_low']:+.4f}, {zero['significance']['joint_f1']['ci95_high']:+.4f}], p={zero['significance']['joint_f1']['p_value']:.4f}. None is significant, support is effectively flat, and selected answer-drop is {zero['selector']['selected_answer_drop_rate']:.2%}. This is a failed zero-shot safety-transfer diagnostic.

    ### 8.2 Few-Shot Gate Calibration

    We calibrate only the safety gate using nested K in {{16, 32, 64, 128}} target-train examples under five fixed seeds. Threshold-only, temperature, Platt, and risk-constrained variants are evaluated with frozen generator, action families, reader, prompt, and evaluation set. The best mean answer-drop is {best['answer_drop_rate_mean']:.2%} at K={best['k']} with {best['method']}, {best['coverage_mean']:.2%} coverage, Answer/SP/Joint F1 {best['answer_f1_mean']:.4f}/{best['sp_f1_mean']:.4f}/{best['joint_f1_mean']:.4f}, ECE {best['ece_mean']:.4f}, and Brier {best['brier_mean']:.4f}. It misses the pre-specified 4% target. Few-shot calibration partially reduces answer-drop risk but does not recover the in-domain safety level. We do not continue tuning K, seed, temperature, or threshold after observing this failure.
    """)

    analysis = clean("""
    ## 9. Analysis

    **Opportunity before selection.** The action generator determines whether repair is possible at all. Pair complementarity raises the chance that a proposal contains both hops, while bounded construction prevents opportunity from becoming an uncontrolled permutation search. Selection then trades coverage for answer risk. This explains why conditional gains can exceed population gains without implying a broad treatment effect.

    **What the Lite failure means.** Pair complementarity, chains, anchors, and selective safety are the most interpretable mechanisms. Yet the untouched holdout shows that lexical pair features alone do not preserve Full quality within the chosen margin. Missing-hop, MPNet, cross-encoder, and document-opportunity components therefore remain in the stronger implementation. Their mixed individual ablations support neither a claim that each is necessary nor a claim that each always helps.

    **Compression versus structured action.** Equalizing token budget removes the most obvious information-volume confound, but it does not equalize objectives. Sentence packing chooses text spans; Full selects a small structural intervention while retaining five-document coverage. The comparison bounds interpretation rather than identifying one universally better constructor.

    **Transfer as a gate boundary.** 2Wiki retains some positive candidate opportunity, but the Hotpot safety probabilities are misaligned with target-domain harm. Target-train calibration lowers risk only partially. The current evidence therefore separates reusable action construction from unresolved risk calibration under shift.
    """)

    limitations = clean("""
    ## 10. Limitations and Ethical Considerations

    The population effects are below one and two F1 points, respectively, on the two same-source holdouts. Selected-query means are conditioned on the policy's choices and cannot be extrapolated to arbitrary or generally improvable queries. Most interventions tie the baseline, and 7.75-7.83% lower Answer F1 among selected interventions.

    Both confirmatory sets come from HotpotQA validation. The 2Wiki experiment is non-significant and violates the answer-drop target even after few-shot calibration, so cross-dataset safety remains unresolved. Calibration also requires labeled target-train reader outcomes and is not zero-shot behavior.

    Full adds measured online latency relative to Frozen Top-5. Lite reduces this overhead but fails the independent quality rule. Historical offline GPU-hour totals are unavailable. The benchmark begins after retrieval and does not include corpus indexing, network transfer, or retriever execution.

    The bounded pool normally contains about ten distractor documents. Larger corpus-scale pools and changing indexes are not evaluated. The fixed reader and support predictor can have entity-, language-, or question-type-specific errors. A selector lowers average risk but offers no correctness guarantee. The method rearranges supplied passages and does not synthesize evidence; this helps auditability but cannot recover facts absent from the pool.
    """)

    conclusion = clean(f"""
    ## 11. Conclusion

    Reader-aware selection is limited first by the contexts its generator makes possible. Full pair-complementary generation creates bounded two-document alternatives, preserves answer anchors, and exposes them to a fully nested reader-safe selector. Two frozen same-source holdouts show modest positive Answer, SP, and Joint F1 changes, with larger descriptive means on the quarter of queries selected for intervention and exact fallback elsewhere. The Lite failure keeps the full semantic recipe in the primary method while narrowing the conceptual contribution. Equal-budget compression, measured latency, and unsuccessful transfer calibration define clear comparison and deployment boundaries. In short, {claim[0].lower() + claim[1:]}
    """)

    appendix = clean(f"""
    # Appendix

    ## A. Frozen Protocol

    - Hotpot source ordering seed: 44.
    - Development: 1,000 queries.
    - Original confirmatory holdout: 3,000 disjoint queries.
    - Untouched revision holdout: 3,405 remaining disjoint queries.
    - Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
    - Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
    - Support predictor threshold: 0.7.
    - Paired bootstrap samples: 5,000.
    - Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

    ## B. Full Component and Training Details

    Full combines lexical and entity features, MPNet similarities, cross-encoder relevance, a missing-hop estimator, a document-opportunity model, pair complementarity, anchor-preserving single and two-document actions, and two selector heads. Each learned component is fold-specific. Outcome labels are computed offline from the frozen reader; inference sees no answer, support label, candidate outcome, or oracle action score.

    The central concepts are pair complementarity, bounded two-document chains, anchor preservation, and selective safety. Other modules are included because the joint Full implementation is empirically stronger than Lite, not because every module has been independently validated as a monotonic contribution.

    ## C. Selected-Policy Distribution

    | Holdout | Selected / N | Metric | Mean | Median | Q25 | Q75 | Wins | Losses | Ties | Drop rate |
    |---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
    | Original | {original_effect['selected_n']}/{original_effect['n_queries']} | Answer | {original_effect['metrics']['answer_f1']['selected_mean_delta']:+.4f} | {original_effect['metrics']['answer_f1']['selected_median_delta']:+.4f} | {original_effect['metrics']['answer_f1']['selected_q25_delta']:+.4f} | {original_effect['metrics']['answer_f1']['selected_q75_delta']:+.4f} | {original_effect['metrics']['answer_f1']['selected_wins']} | {original_effect['metrics']['answer_f1']['selected_losses']} | {original_effect['metrics']['answer_f1']['selected_ties']} | {original_effect['metrics']['answer_f1']['selected_drop_rate']:.2%} |
    | Original | {original_effect['selected_n']}/{original_effect['n_queries']} | Joint | {original_effect['metrics']['joint_f1']['selected_mean_delta']:+.4f} | {original_effect['metrics']['joint_f1']['selected_median_delta']:+.4f} | {original_effect['metrics']['joint_f1']['selected_q25_delta']:+.4f} | {original_effect['metrics']['joint_f1']['selected_q75_delta']:+.4f} | {original_effect['metrics']['joint_f1']['selected_wins']} | {original_effect['metrics']['joint_f1']['selected_losses']} | {original_effect['metrics']['joint_f1']['selected_ties']} | {original_effect['metrics']['joint_f1']['selected_drop_rate']:.2%} |
    | Revision | {revision_effect['selected_n']}/{revision_effect['n_queries']} | Answer | {revision_effect['metrics']['answer_f1']['selected_mean_delta']:+.4f} | {revision_effect['metrics']['answer_f1']['selected_median_delta']:+.4f} | {revision_effect['metrics']['answer_f1']['selected_q25_delta']:+.4f} | {revision_effect['metrics']['answer_f1']['selected_q75_delta']:+.4f} | {revision_effect['metrics']['answer_f1']['selected_wins']} | {revision_effect['metrics']['answer_f1']['selected_losses']} | {revision_effect['metrics']['answer_f1']['selected_ties']} | {revision_effect['metrics']['answer_f1']['selected_drop_rate']:.2%} |
    | Revision | {revision_effect['selected_n']}/{revision_effect['n_queries']} | Joint | {revision_effect['metrics']['joint_f1']['selected_mean_delta']:+.4f} | {revision_effect['metrics']['joint_f1']['selected_median_delta']:+.4f} | {revision_effect['metrics']['joint_f1']['selected_q25_delta']:+.4f} | {revision_effect['metrics']['joint_f1']['selected_q75_delta']:+.4f} | {revision_effect['metrics']['joint_f1']['selected_wins']} | {revision_effect['metrics']['joint_f1']['selected_losses']} | {revision_effect['metrics']['joint_f1']['selected_ties']} | {revision_effect['metrics']['joint_f1']['selected_drop_rate']:.2%} |

    Gain per 100 original-holdout interventions is +{original_effect['metrics']['answer_f1']['gain_per_100_interventions']:.2f} Answer-F1 points and +{original_effect['metrics']['joint_f1']['gain_per_100_interventions']:.2f} Joint-F1 points. These are descriptive accounting quantities, not policy values for unselected queries.

    ## D. RECOMP Development Curve and Top-1 Diagnostic

    The official compressor is evaluated at 64, 128, 256, 384, 512, and 660 token targets on development. The 660-token protocol is frozen for the holdout. The Top-1 compatibility condition averages approximately 47 tokens and one represented document; it is not used for a broad superiority claim. Baseline-Truncated packs source-order sentences to the same targets.

    ## E. 2Wiki Calibration Grid

    The full K-by-method table is stored with the submission artifacts. Each cell averages five fixed seeds and reports coverage, selected answer-drop, Answer/SP/Joint F1, ECE, and Brier score. The minimum mean answer-drop is {best['answer_drop_rate_mean']:.2%}, above the 4% target.

    ## F. End-to-End Timing Protocol

    The timing harness recomputes every online stage while enforcing frozen final contexts. Full is split into document preprocessing, lexical features, MPNet encoding, cross-encoder scoring, missing-hop prediction, document-opportunity scoring, pair-feature construction, pair-complementarity scoring, action construction, safety, positive utility, serialization, and reader. Full mean/P95 total latency is {full_ms:.2f}/{full_p95:.2f} ms, compared with {base_ms:.2f}/{base_p95:.2f} ms for Frozen Top-5. Online reader calls per query equal one for every system.

    ## G. Reproducibility Boundary

    Frozen method predictions, holdout outcomes, and source artifacts are not overwritten. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The timing result covers post-retrieval execution on one fixed machine and should not be read as a retriever or index benchmark.
    """)

    header = f"# {title}"
    main_paper = "\n\n".join([header, "## Abstract\n\n" + abstract, introduction, related, method, setup, results, recomp_section, cost_section, external, analysis, limitations, conclusion])
    full_paper = main_paper + "\n\n" + appendix
    anonymous = main_paper

    write(HERE / "abstract_v5_final.md", "# Abstract\n\n" + abstract)
    write(HERE / "introduction_v5_final.md", introduction)
    write(HERE / "method_v5_final.md", method)
    write(HERE / "results_v5_final.md", "\n\n".join([results, recomp_section, analysis]))
    write(HERE / "cost_section_v5_final.md", cost_section)
    write(HERE / "external_transfer_v5_final.md", external)
    write(HERE / "limitations_v5_final.md", limitations)
    write(HERE / "paper_full_clean_v5_final.md", full_paper)
    write(HERE / "paper_main_conference_v5_final.md", main_paper)
    write(HERE / "paper_anonymous_v5_final.md", anonymous)
    write(HERE / "paper_appendix_v5_final.md", appendix)

    review_response = clean(f"""
    # Final Response to Review Concerns

    We thank the reviewers for identifying weaknesses that required both additional controls and narrower claims. All new evaluations preserve the frozen Full system; neither the 3,000-query nor the 3,405-query holdout outcomes were used to select methods or thresholds.

    ## Weakness 1: Marginal Absolute Gains

    **Response.** We agree that the population gains are modest and now state this in the Abstract, Introduction, Results, Limitations, and Conclusion. We add a second untouched 3,405-query confirmation and exact intervals and p-values for both holdouts. We also add direct policy-selected accounting: on the original 774 interventions, Answer/SP/Joint F1 change by +0.0340/+0.0219/+0.0250, with Answer wins/losses/ties of 89/60/625 and Joint 141/115/518. These are labeled descriptive conditional effects, not causal or arbitrary-query effects. The population deltas remain primary, and the new end-to-end cost table prevents a practical-impact claim beyond the evidence.

    ## Weakness 2: Limited Transfer

    **Response.** We retain the failed frozen 2Wiki result: coverage is 26.0%, selected answer-drop is 6.92%, Answer and Joint changes are non-significant, and SP is flat. We add target-train gate calibration for K={{16,32,64,128}} under five seeds, with frozen generator, reader, prompt, action space, and evaluation outcomes. The best mean answer-drop is 5.10%, an improvement from 6.92% but above the pre-specified 4% success criterion. We therefore keep distribution-shift safety as an explicit limitation.

    ## Weakness 3: Mixed Semantic Ablations

    **Response.** We agree that mixed component results do not support treating every semantic signal as a separate novelty claim. The paper now centers pair complementarity, bounded two-document chains, anchor preservation, and selective safety. We evaluate a development-frozen Lite simplification, but on the untouched revision holdout Lite minus Full Joint F1 is -0.0063 and fails the 0.002 non-inferiority rule. Full remains empirically stronger. Missing-hop, MPNet, cross-encoder, and document-opportunity components are described as the Full implementation recipe, not as independently proven monotonic contributions.

    ## Weakness 4: Unfair RECOMP Comparison

    **Response.** We agree that the approximately 47-token Top-1 condition cannot support a general ranking against near-full contexts. We add an official-compressor 64-660-token development curve, a source-order Baseline-Truncated control, and a frozen 660-token protocol evaluated on the 3,000-query holdout with the same Top-5 input, FLAN reader, support predictor, and metric code. RECOMP-660 changes Joint F1 by {rsig['joint_f1']['delta']:+.4f} ({ci(rsig['joint_f1'])}, p={fmt_p(rsig['joint_f1']['p_value'])}). We remove broad superiority language and describe this as an official-compressor implementation under reader and budget adaptation. Matched tokens do not equate the structural action spaces.

    ## Weakness 5: Complexity

    **Response.** We now separate offline outcome labeling and training from online inference. Online inference calls the answer reader once on the final context, not once per action. Under a shared 50-warmup/500-measurement protocol, Full measures {full_ms:.2f} ms/query end to end after retrieval, including {1000*full_cost['generator_only_latency']['mean_seconds']:.2f} ms generation, {1000*full_cost['selector_only_latency']['mean_seconds']:.2f} ms selection, and {1000*full_cost['reader_only_latency']['mean_seconds']:.2f} ms reading; Frozen Top-5 measures {base_ms:.2f} ms. Lite reduces this to {lite_ms:.2f} ms but fails non-inferiority. We therefore report an explicit quality-cost trade-off and limit scope to bounded post-retrieval pools. Historical offline GPU-hour totals were not recorded and are therefore unavailable.
    """)
    write(HERE / "review_response_final.md", review_response)

    revision_summary = clean(f"""
    # Final Revision Summary

    - Reordered Method so Full is primary; Lite is a review-driven simplification experiment.
    - Added the untouched 3,405-query holdout to the main evidence with exact paired statistics.
    - Recomputed selected-policy wins, losses, ties, quantiles, harm rates, and exact fallback behavior.
    - Replaced the Top-1-centered RECOMP discussion with a 660-token holdout comparison and fair objective boundary.
    - Added a synchronized 50-warmup/500-query end-to-end benchmark for five frozen systems. Full is {full_ms:.2f} ms/query versus {base_ms:.2f} ms/query for Frozen Top-5.
    - Preserved Lite non-inferiority failure and 2Wiki calibration failure.
    - Removed unresolved measurement placeholders and narrowed claims to bounded same-source context construction.
    """)
    write(HERE / "revision_summary.md", revision_summary)

    shutil.copyfile(V4_REFINED / "references.bib", HERE / "references.bib")
    write(OUT / "audits/paper_build_manifest.json", json.dumps({
        "status": "complete", "title": title, "one_sentence_claim": claim, "abstract_word_count": word_count,
        "full_mean_ms": full_ms, "baseline_mean_ms": base_ms, "full_overhead_ms": full_overhead,
        "files": ["paper_full_clean_v5_final.md", "paper_main_conference_v5_final.md", "paper_anonymous_v5_final.md", "paper_appendix_v5_final.md"],
    }, indent=2))
    print(json.dumps({"status": "complete", "abstract_words": word_count, "full_ms": full_ms, "baseline_ms": base_ms}, indent=2))


if __name__ == "__main__":
    main()
