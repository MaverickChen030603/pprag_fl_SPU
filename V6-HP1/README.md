# V6-HP1 HotpotQA Experiment Pipeline

`V6-HP1/` is a parallel branch of `V6/` that replaces the previous low-sensitivity downstream dataset with **HotpotQA fullwiki**.

## Goal

Use a harder, multi-hop QA dataset to test whether same-budget selective upload can produce clearer downstream differences than the earlier dataset.

## Key Changes

- Upstream FedE training data is rebuilt from **HotpotQA train split** supporting-facts contexts.
- Downstream RAG evaluation switches to **HotpotQA validation split**.
- `RAGTest` is controlled by environment variables:
  - `RAGTEST_DATASET=hotpot_qa`
  - `HOTPOT_SPLIT=validation`
  - `HOTPOT_MAX_EXAMPLES=<N>`
- Full automation now includes Hotpot data preparation, upstream suites, downstream RAG evaluation, suite reports, full-pipeline reports, and final Chinese record/analysis documents.

## Main Scripts

- `run_v6_hp1_all.sh`: full automation entrypoint.
- `V6-HP1/prepare_hotpot_data.py`: converts HotpotQA into FedE upstream training JSON.
- `V6-HP1/run_experiment_suite.py`: upstream suite runner.
- `V6-HP1/finalize_pipeline.py`: upstream summary, downstream RAG evaluation, and full-pipeline report generation.
- `V6-HP1/write_experiment_docs.py`: final Chinese record and analysis writer.

## Suites

- `v6hp1_main`
- `v6hp1_budget_aligned`
- `v6hp1_heterogeneity`
- `v6hp1_hardquery`
- `v6hp1_ablation_signal`
- `v6hp1_ablation_budget`
- `v6hp1_explain`

## Quick Start

```bash
bash run_v6_hp1_all.sh
```

Useful overrides:

```bash
GPU_ID=0 BATCH_SIZE=1 SEED_LIST=0,1,2 HOTPOT_TRAIN_EXAMPLES=5000 HOTPOT_EVAL_EXAMPLES=1000 bash run_v6_hp1_all.sh
```

## Outputs

- Upstream outputs: `V6-HP1/outputs/pprag_fl_v6_hp1/...`
- Downstream outputs: `V6-HP1/outputs/rag_eval_all_v6_hp1/...`
- Reports: `实验分析报告/V6-HP1/...`
- Final docs:
  - `V6-HP1/v6_hp1_complete_experiment_record_cn.md`
  - `V6-HP1/v6_hp1_complete_experiment_analysis_cn.md`
