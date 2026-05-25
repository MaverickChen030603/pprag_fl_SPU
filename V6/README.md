# V6 Experiment Pipeline

`V6/` is a parallel experiment line built on top of `V5/`, focused on a stricter research question:

- can we improve downstream RAG utility under the **same communication budget**
- can we keep budget expansion rare and targeted
- can we make downstream-aware selection sharper instead of simply more conservative

## Key changes

- `hypernet_v6`: downstream-aware scorer used under fixed-budget-first settings
- `adaptive_v6`: stricter budget expansion logic; expansion only happens under very hard and very high-utility cases
- `v6_budget_aligned`: explicit same-budget comparison suite
- `v6_hardquery`: harder downstream setting
- `v6_ablation_budget`: fixed budget vs adaptive_v6 vs fixed-layerwise replacement

## Suites

- `v6_main`
- `v6_budget_aligned`
- `v6_heterogeneity`
- `v6_hardquery`
- `v6_ablation_signal`
- `v6_ablation_budget`
- `v6_explain`
- `all_v6`

## Quick start

Single suite:

```bash
python V6/run_experiment_suite.py --suite v6_main --gpu 0 --batch-size 1 --seed-list 0,1,2
```

Full automation:

```bash
bash run_v6_all.sh
```

## Outputs

- upstream outputs: `V6/outputs/pprag_fl_v6/...`
- downstream outputs: `V6/outputs/rag_eval_all_v6/...`
- reports: `实验分析报告/V6/...`
