# B3 / Multi-seed / Ablation Sequence Launch Report

Date: 2026-06-29 13:57:09 JST

## Launch Summary

- Sequence stamp: `20260629_135708`
- Queue log: `/home/iiserver31/projects/FedE4RAG-main/experiments/v6_hp_hyper_next/logs/b3_multiseed_ablation_sequence_20260629_135708.log`
- GPU free-memory threshold: `38000 MiB`
- Order: B3 hard_500 seed 42 -> multi-seed baseline -> ablation groups

## Planned Stages

1. B3: `hotpot_hard_500`, seed `42`, same-payload V3/V4/V5/V6.
2. Multi-seed: `hotpot_all_1000`, `hotpot_hard_1000`, `hotpot_hard_500` with seeds `43` and `44`.
3. Ablation groups: `pooler`, `layerwise`, `score_mode`, `hard_weight`, `adaptive`.

## Expected Output Families

- B3 raw CSV: `experiments/v6_hp_hyper_next/results/same_payload_b3_hard500_raw.csv`
- Multi-seed raw CSV: `experiments/v6_hp_hyper_next/results/same_payload_multiseed_raw.csv`
- Ablation raw/summary/report files under `experiments/v6_hp_hyper_next/results/` and `experiments/v6_hp_hyper_next/reports/`.

## Notes

The script waits for a safe GPU window before each stage and does not preempt other users' GPU jobs.
