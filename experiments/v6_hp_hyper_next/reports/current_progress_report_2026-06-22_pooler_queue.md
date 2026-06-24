# V6-HP-hyper Pooler Ablation Progress Report

Date: 2026-06-22 JST

## Status

The pooler dominance ablation is prepared and queued on the server. It has not started yet because GPU memory is not safely available.

Resource check:

- `/home` usage: 91%, acceptable.
- `experiments/v6_hp_hyper_next`: 3.3G.
- All GPUs are occupied by other workloads.
- Free GPU memory is around 17-19GB, below the safe threshold of 38GB.

Queue:

- PID: `583103`
- Log: `experiments/v6_hp_hyper_next/logs/pooler_ablation_queue_20260622_021439.log`
- Command to be launched when safe: `GROUP=pooler bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh`

## Completed This Turn

- Added `summarize_selection_ablation.py`.
- Updated `run_selection_diversity_ablation.sh` to write group-specific outputs.
- Updated after-ablation audit CLI compatibility.
- Synced updated scripts to the server.
- Passed server-side compile and shell syntax checks.
- Started safe GPU queue for pooler ablation.

## Current Decision

B3 hard_500 remains deferred.

Seeds 43/44 remain deferred.

Layerwise and score_mode ablations are not started yet because the pooler group must complete first.

## Next Check

Check the queue log later:

```bash
tail -120 experiments/v6_hp_hyper_next/logs/pooler_ablation_queue_20260622_021439.log
```

If it completes, inspect:

```bash
experiments/v6_hp_hyper_next/results/pooler_ablation_raw.csv
experiments/v6_hp_hyper_next/results/pooler_ablation_summary.csv
experiments/v6_hp_hyper_next/reports/pooler_ablation_report.md
```
