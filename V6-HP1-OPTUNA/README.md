# V6-HP1-OPTUNA

Small-scale Optuna search controller for the HotpotQA V6-HP1 pipeline.

This folder is intentionally a controller layer. It does not replace the
`V6-HP1` experiment code; each trial calls the existing upstream and
downstream scripts with a sampled configuration, then records the combined
communication/downstream score in an Optuna study.

## MVP Search

- Dataset: HotpotQA validation split
- Method: `hypernet_v6`
- Trials: 20 by default
- Seed: 0 by default
- Rounds: 10 by default
- Downstream examples: 300 by default
- Objective: maximize downstream quality with a payload penalty

The default objective is:

```text
0.30 * MRR + 0.25 * NDCG + 0.20 * F1 + 0.15 * EM + 0.10 * recall_3
  - payload_penalty * overall_payload_ratio
```

## Run

```bash
bash run_v6_hp1_optuna.sh
```

By default the launcher waits for the main `run_v6_hp1_all.sh` chain to
finish, so it will not compete for `cuda:0`. To run immediately on another
GPU:

```bash
WAIT_FOR_V6_HP1=0 GPU_ID=1 bash run_v6_hp1_optuna.sh
```

## Outputs

- `V6-HP1-OPTUNA/outputs/optuna/v6hp1_optuna.db`
- `V6-HP1-OPTUNA/outputs/trial_results/trial_*.json`
- `V6-HP1-OPTUNA/outputs/optuna_summary.json`
- `V6-HP1-OPTUNA/outputs/optuna_summary.csv`
- `V6-HP1-OPTUNA/outputs/optuna_report.md`

