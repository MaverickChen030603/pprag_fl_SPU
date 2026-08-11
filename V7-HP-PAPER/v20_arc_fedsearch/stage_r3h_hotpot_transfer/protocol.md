# R3-H Frozen Hotpot Transfer and Routing-Merge Composition

This is an isolated successor stage. It does not modify R3 2Wiki/MuSiQue,
M0, or the stopped REM-P/CTD-CSR line.

## Frozen contract

- Train: first 5,000 Hotpot train query IDs after sorting by
  `sha256("r3h-hotpot-train-v1:" + query_id)`.
- Holdout: first 300 eligible Hotpot development IDs after sorting by
  `sha256("r3h-hotpot-holdout-v1:" + query_id)`.
- Exclusions: all M0 N=100/N=300 records, V19 reader-development records,
  and the prior uncompleted R3 Hotpot transfer holdout.
- Candidate protocol: P0 Top-8, 18 fixed float32 no-body features, 592-byte
  packet, independent Top-3, local depth 10, five documents per client,
  15 documents total, global top-10.
- Routing: B1 static P0, B3 unified frozen P5 (`0.25 * static + 0.75 *
  dense-top3-mean` after per-query min-max), B4 class-balanced Logistic
  Regression using static score plus the 18 fixed features.
- Merge: raw dense score and the M0 rank-percentile rule only.
- Gold is prohibited in packet materialization and inference. The sealed
  holdout label file may be read only by the offline evaluator after canonical
  routing and document-output files are frozen.
- Reader and final test are prohibited.

## Gates

Gate A compares H5 (B4/raw) with H1 (B1/raw): coverage >= +0.08, local and
raw merged complete@10 >= +0.05, positive bootstrap lower bounds, rescue >
harm, no official retrieval metric below -0.02, and the same result for all
three deterministic seeds.

Gate B compares H6 (B4/percentile) with H5 and H2. It requires no more than
one-point degradation, a +0.03 final complete@10 gain over the strongest
single component, rescue/harm > 2, and a positive bootstrap lower bound.
