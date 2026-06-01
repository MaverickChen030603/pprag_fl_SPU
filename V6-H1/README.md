# V6-H1 Hard-Query Experiment Pipeline

`V6-H1/` is a parallel branch of `V6/` focused on one sharper question:

Can selective upload show clearer downstream advantages when evaluation is restricted to stable hard queries rather than the full, relatively easy query set?

## Key Changes

- Keeps V6 same-budget and fixed-budget-first comparisons.
- Adds per-query downstream logging in `RAGTest/main_100_test.py`.
- Builds `V6-H1/hard_queries/stable_hard_queries.json` from baseline failures.
- Re-runs downstream evaluation on the stable hard-query subset.
- Generates complete experiment record and analysis documents after automation finishes.

## Main Scripts

- `run_v6_h1_all.sh`: full automation entrypoint.
- `V6-H1/run_experiment_suite.py`: upstream suite runner.
- `V6-H1/finalize_pipeline.py`: upstream summary, downstream RAG evaluation, and full-pipeline report.
- `V6-H1/build_hard_query_subset.py`: stable hard-query subset builder.
- `V6-H1/write_experiment_docs.py`: final Chinese record and analysis writer.

## Suites

- `v6h1_main`
- `v6h1_budget_aligned`
- `v6h1_heterogeneity`
- `v6h1_hardquery`
- `v6h1_ablation_signal`
- `v6h1_ablation_budget`
- `v6h1_explain`

## Quick Start

```bash
bash run_v6_h1_all.sh
```

Useful overrides:

```bash
GPU=0 BATCH_SIZE=1 SEED_LIST=0,1,2 bash run_v6_h1_all.sh
```

## Outputs

- Upstream outputs: `V6-H1/outputs/pprag_fl_v6_h1/...`
- Full-set downstream outputs: `V6-H1/outputs/rag_eval_all_v6_h1/...`
- Hard-query downstream outputs: `V6-H1/outputs/rag_eval_hard_v6_h1/...`
- Hard-query subset: `V6-H1/hard_queries/stable_hard_queries.json`
- Reports: `实验分析报告/V6-H1/...`
- Final docs: `V6-H1/v6_h1_complete_experiment_record_cn.md` and `V6-H1/v6_h1_complete_experiment_analysis_cn.md`
