# R3-H Maintenance Recovery

This stage is a completed, immutable HotpotQA ranker-training checkpoint. The
three class-balanced logistic-regression seed models live in `train/models/`.
The 5,000-query packet file is intentionally not versioned: it is 157 MB and is
fully regenerable from the frozen source data and index below.

## Required server state

The restored machine must provide the same project root and read-only source
assets used by the original run:

- `/home/iiserver31/projects/FedE4RAG-main`
- `V7-HP-PAPER/v17_fedaction_rag/data/hotpotqa/{train,development}.jsonl`
- `V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes/hotpotqa/topic_silo`
- `V7-HP-PAPER/v17_fedaction_rag/partitions/{centroids/hotpotqa/topic_silo_m20.npy,assignments/hotpotqa/topic_silo_m20.jsonl}`
- the historical exclusion paths enumerated in `run_r3h_smoke.sh`
- Python environment `/home/iiserver31/anaconda3/envs/supv2`

## Recovery Procedure

1. Restore the repository and switch to the saved commit.
2. Confirm the paths above are mounted and choose an available CUDA device.
3. Run `bash V7-HP-PAPER/v20_arc_fedsearch/stage_r3h_hotpot_transfer/resume_r3h_after_maintenance.sh <cuda-device>`.

The recovery script reconstructs only the sealed split/profile and five-query
label-free smoke state when absent, then SHA-256 verifies all three frozen model
objects against `frozen_model_manifest.json`. It never launches reader/final
evaluation and never calls `run_r3h_train.sh`.

`run_r3h_train.sh` is deliberately write-once and refuses to overwrite
`model_results.csv`. Use it only for a wholly new, separately authorized R3-H
attempt, never to resume this completed checkpoint.
