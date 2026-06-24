# V6-HP-hyper Selection Diagnosis Progress Report

Date: 2026-06-19 JST

## Current Status

This round shifted from benchmark expansion to mechanism diagnosis. The server resource check was completed and appended to `experiments/v6_hp_hyper_next/logs/all_commands.log`.

Resource status:

- `/home` usage is 91%, so storage is acceptable.
- All A100 GPUs are currently occupied by other workloads.
- No active V6-HP-hyper B3, hard500, ablation, or score-logging process is running.
- Because the GPUs are busy, no new V6-HP-hyper training job was launched in this step.

## Implemented

- Added score-distribution logging hooks to V4, V5, and V6-HP1.
- Added V6-HP1 selection-diversity controls:
  - `score_mode=delta`
  - `score_mode=grad_norm`
  - `pooler_cap_ratio`
  - `exclude_pooler`
  - `hard_weight_alpha`
  - `budget_mode=adaptive_realloc` as a same-payload pilot mode.
- Added score-log collection script.
- Added anchor scorelog rerun script.
- Added selection-diversity ablation script.
- Added after-ablation identity audit script.
- Generated current score logging and after-ablation reports in pending state.
- Generated paper landing diagnosis report.

## Not Yet Run

The following require a safe GPU window:

- S1 scorelog anchor rerun on `hotpot_hard_1000`, seed 42, methods V4/V5/V6.
- S2 selection-diversity ablation groups.
- S3 after-ablation audit using real ablation score logs.

## Current Decision

B3 `hotpot_hard_500` remains deferred. Multi-seed expansion also remains deferred.

Reason:

- Existing B2 improvement is too small.
- Existing method identity audit shows V4/V5/V6 selection collapse.
- Score-distribution logging has just been added, but no scorelog rerun has completed yet.

## Next Action

When GPU becomes safely available, run:

```bash
bash experiments/v6_hp_hyper_next/run_scorelog_anchor_hard1000.sh
```

Then run ablation groups selectively:

```bash
GROUP=layerwise bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh
GROUP=score_mode bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh
GROUP=pooler bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh
```

Only after a group creates meaningful selection diversity should B3 or seed expansion be reconsidered.
