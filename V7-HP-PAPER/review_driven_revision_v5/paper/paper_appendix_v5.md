# Appendix

## A. Frozen Protocol

- Hotpot source ordering seed: 44.
- Development: indices 0--999.
- Original confirmatory holdout: indices 1,000--3,999.
- Revision holdout: indices 4,000--7,404.
- Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
- Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
- Support predictor threshold: 0.7.
- Bootstrap samples: 5,000.
- Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

## B. Generator Ablation Interpretation

The main text reports Full, Lite-Lexical-Pair, Lite-Semantic-Pair, PairChain-Ablation, removal of pair complementarity, removal of two-document chains, anchor preservation, and the safety selector. Missing-hop, MPNet, cross-encoder, and document-opportunity ablations are implementation diagnostics. Their mixed behavior is not interpreted as consistent independent benefit.

## C. RECOMP Protocol

The author-released checkpoint `fangyuan/hotpotqa_extractive_compressor` scores every sentence in the same frozen Top-5 input. Whole sentences are added in score order to the nearest target context budget. The fixed holdout protocol is 660 tokens; the 64--660 curve is development-only. Baseline-Truncated uses the same sentence packing budget in source order. The answer reader and support predictor are shared.

## D. Candidate-Pool Boundary

Pool sensitivity status: `scope_limited`. In the frozen 3,000 artifact, a common 10/20/50/100-document subset is unavailable. Top-L pruning fixes pair scoring at at most 45 pairs for L=10 even when a larger upstream pool exists. This is a complexity bound, not an open-domain retrieval experiment.

## E. Reproducibility and Missing Measurements

All V5 outputs are generated under `review_driven_revision_v5/` without changing the frozen V4 paper or result directories. Values absent from source manifests are marked `[NEEDS MEASUREMENT]`, `[NEEDS SOURCE FILE]`, or `[NOT AVAILABLE]`; no elapsed time or call count is inferred from file modification times.
