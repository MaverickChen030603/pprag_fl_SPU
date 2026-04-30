# SUP_v2: Complete Experiment Code for PPRAG-FL Selective Upload

`SUP_v2` implements the experimental design in `SUP_V1/complete_rp_cn.md`.
It focuses on the upstream FL communication bottleneck and connects the trained
retriever checkpoint to the downstream `RAGTest` workflow.

## Implemented Methods

- `full`: FedE4RAG-style full upload baseline.
- `random`: randomly upload `K` parameter blocks.
- `static_top`: always upload the last `K` BERT/BGE blocks.
- `delta_norm`: upload blocks with the largest previous-round delta norm.
- `hypernet`: PPRAG-FL, using a hypernetwork to predict block importance.

## Files

- `fedrag_selective_upload.py`: FLGo-compatible `Server` and `Client`.
- `hypernetwork.py`: block grouping, delta statistics, and importance hypernetwork.
- `upload_selectors.py`: comparison-method upload selectors.
- `experiment_config.py`: default configs and ablation-suite builders.
- `run_upstream.py`: run one upstream FL experiment.
- `run_experiment_suite.py`: run comparison/top-k/warmup/encryption suites.
- `run_rag_eval.py`: invoke downstream `RAGTest` scripts for a trained retriever.
- `run_all_rag_eval.py`: batch downstream `RAGTest` for every upstream run with a saved HF retriever.
- `summarize_results.py`: aggregate communication logs into CSV/JSON.
- `metrics.py`: logging and payload-estimation helpers.
- `report_generator.py`: automatically generate experiment analysis reports in Markdown/JSON.

## Upstream Single Run

Run from the repository root:

```bash
python3 SUP_v2/run_upstream.py --strategy hypernet --topk 3 --warmup 2
```

For baselines:

```bash
python3 SUP_v2/run_upstream.py --strategy full --topk 0
python3 SUP_v2/run_upstream.py --strategy random --topk 3
python3 SUP_v2/run_upstream.py --strategy static_top --topk 3
python3 SUP_v2/run_upstream.py --strategy delta_norm --topk 3
```

## Experiment Suites

```bash
python3 SUP_v2/run_experiment_suite.py --suite comparison
python3 SUP_v2/run_experiment_suite.py --suite topk
python3 SUP_v2/run_experiment_suite.py --suite warmup
python3 SUP_v2/run_experiment_suite.py --suite encryption
```

Use `--dry-run` to generate and inspect the suite manifest without launching training:

```bash
python3 SUP_v2/run_experiment_suite.py --suite all --dry-run
```

Results are written under:

```text
SUP_v2/outputs/<experiment_name>/<strategy>_k<topk>_w<warmup>_s<seed>_enc<0|1>/
```

Each run stores:

- `upstream_config.json`
- `run_metadata.json`
- `round_logs.jsonl`
- `round_logs.json`
- `round_logs.csv`
- `final_artifacts.json`
- `retriever_state_<timestamp>.bin`
- `retriever_hf_<timestamp>/` for downstream `RAGTest --model`
- `hypernet_<timestamp>.pt` for the hypernetwork strategy

After each completed upstream run, an analysis report is also generated under:

```text
实验分析报告/
```

The report includes:

- configuration snapshot
- communication statistics
- automatic textual analysis
- downstream RAG handoff path

## Summarize Communication Results

```bash
python3 SUP_v2/summarize_results.py
```

This produces:

```text
SUP_v2/outputs/pprag_fl_sup_v2/summary.json
SUP_v2/outputs/pprag_fl_sup_v2/summary.csv
```

In addition, `run_experiment_suite.py` automatically writes a suite-level analysis
report into `实验分析报告/`, summarizing all completed runs in the suite.

## Downstream RAG Evaluation

After upstream training, use the saved retriever path with `RAGTest`:

```bash
python3 SUP_v2/run_rag_eval.py \
  --model SUP_v2/outputs/pprag_fl_sup_v2/hypernet_k3_w2_s0_enc0/retriever_hf_<timestamp> \
  --script main_100_test.py \
  --output-dir SUP_v2/outputs/rag_eval/hypernet_k3
```

Use `--dry-run` to inspect the command without executing it.

To batch-evaluate every upstream run that produced `retriever_hf_*`:

```bash
python3 SUP_v2/run_all_rag_eval.py
```

## Required Project Setup

The original FedE model placeholder has been replaced with:

```text
BAAI/bge-base-en-v1.5
```

Override it with a local HuggingFace model directory when needed:

```bash
export FEDE_EMBEDDING_MODEL=/path/to/local/bge-base-en-v1.5
```

The training data is prepared as:

```text
FedE/select_data.json -> FedE/train_data/data_1000_random.json
```

Before full training, still verify project-specific data/model availability:

- `FedE/flgo/benchmark/fedrag_classification/config.py`
- `FedE/flgo/benchmark/fedrag_classification/core.py`
- `FedE/select_data.json`

## Notes

This code logs estimated communication reduction. If `--estimate-encryption` is
enabled, encrypted payload size is estimated using an expansion factor; it does
not perform real CKKS encryption inside SUP_v2.
