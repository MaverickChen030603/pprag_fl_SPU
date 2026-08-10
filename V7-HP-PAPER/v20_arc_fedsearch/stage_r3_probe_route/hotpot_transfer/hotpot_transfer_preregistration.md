# R3-T/R3-C Frozen Hotpot Transfer Contract

- Hotpot model: the same 18-float probe schema, P0 Top-8 candidates, class-balanced Logistic Regression (`C=1`, `liblinear`), P0 Top-8 non-support hard negatives, independent Top-3, local depth 10 and 15-document main budget.
- Training: the existing 5,000-query Hotpot train split only. No 2Wiki/MuSiQue weights, no retriever tuning and no Hotpot holdout selection.
- Fresh holdout: source-order tail 300 of Hotpot development after excluding every ID already materialized in M0 N=100/N=300 and V19 development confirmation.
- Label-free transfer: reuse the frozen 2Wiki P5 rule, normalized `0.25 * static_score + 0.75 * dense_top3_mean`; no Hotpot rule search.
- H0/H1: inherited Bc=3 route with raw/percentile merge. H2/H3: label-free probe with raw/percentile. H4/H5: Logistic probe with raw/percentile.
- Cost baselines: C0 static Top-3 x5 docs, C1 static Top-4 x5 docs, C2 static Top-3 x5 plus P0 ranks 4-8 x1. Their actual document bytes, client compute and document counts are reported rather than byte-matched by assertion.
- Reader and final test remain prohibited. Hotpot reader is considered only after the retrieval and cost reports are complete and the reader gate is checked jointly with 2Wiki/MuSiQue.
