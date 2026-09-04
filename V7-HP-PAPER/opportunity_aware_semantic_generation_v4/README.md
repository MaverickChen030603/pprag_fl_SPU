# V7-HP-PAPER Opportunity-Aware Semantic Generation V4

This independent branch freezes v2 as the submission fallback and v3 as a negative opportunity study. It tests a fully nested semantic candidate generator on the same 1,000-query HotpotQA development protocol.

The development execution order is `00` through `12`. Stage `06` applies opportunity gates frozen before v4 reader outcomes. Selector and downstream evaluations are skipped unless the gate passes. Target-query generation never uses the gold answer, gold supporting titles, reader outcomes, or oracle actions.

The same-source scale-up is implemented by stages `13` through `18` and can be run with:

```bash
bash V7-HP-PAPER/opportunity_aware_semantic_generation_v4/run_same_source_scaleup_3000.sh
```

It reconstructs the original `hotpot_qa/distractor/validation` seed-44 ordering, verifies exact source and baseline title-order reproduction on all 1,000 development queries, and freezes the disjoint slice `[1000:4000]`. The baseline remains `HybridSoftRetriever(alpha=0.55, uniform weights, top_k<=5)`; BM25-only top-5 is not substituted. Generator models, selector thresholds and coverage, reader prompts and decoding, and the sentence-support threshold are unchanged during the 3,000-query evaluation.
