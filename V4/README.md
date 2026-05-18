# V4: Downstream-Aware Value Hypernetwork for Federated RAG

`V4/` 在 `V3/` 基础上继续推进，目标是不只降低上游通信量，还要更明确地让“更省下来的通信”对应到更稳定的下游 RAG 效果。

V4 的关键增强包括：

- `downstream-aware` 打分：在重要性/价值密度之外，引入下游收益代理
- `hard-query / hard-client` 感知：对更难的查询与客户端分配更高优先级
- `utility memory`：记录 block 的收益历史，而不只是被选频率
- `adaptive_v4` 预算：根据预算预测与客户端 hardness 联合调整上传块数

## 目录说明

- `run_upstream.py`：单次上游实验入口
- `run_experiment_suite.py`：V4 实验套件入口
- `fedrag_selective_upload.py`：上游联邦训练主流程
- `hypernetwork.py`：V4 超网络与下游代理打分
- `history_memory.py`：收益历史记忆
- `utility_proxy.py`：下游收益代理估计
- `hardness_estimator.py`：hard-query / hard-client 估计
- `downstream_sampler.py`：轻量下游稳定性代理
- `budget_allocator.py`：V4 自适应预算
- `run_all_rag_eval.py`：批量下游评测
- `finalize_pipeline.py`：上下游汇总与总报告

## V4 的关键方法开关

- `selection_strategy`
  - `full`
  - `random`
  - `static_top`
  - `delta_norm`
  - `hypernet_v2`
  - `hypernet_v3`
  - `hypernet_v4`
- `score_mode`
  - `importance`
  - `value`
  - `downstream_value`
- `budget_mode`
  - `fixed`
  - `adaptive`
  - `adaptive_v4`
- `use_hard_query_weighting`
- `use_utility_memory`
- `layerwise_budget`

## 快速开始

### 1. dry-run 查看单次配置

```bash
python V4/run_upstream.py \
  --strategy hypernet_v4 \
  --topk 3 \
  --warmup 1 \
  --score-mode downstream_value \
  --budget-mode adaptive_v4 \
  --dry-run
```

### 2. smoke test

```bash
python V4/run_experiment_suite.py --suite smoke
```

### 3. 主实验

```bash
python V4/run_experiment_suite.py --suite v4_main
```

### 4. 预算实验

```bash
python V4/run_experiment_suite.py --suite v4_budget
```

### 5. 异构性实验

```bash
python V4/run_experiment_suite.py --suite v4_heterogeneity
```

### 6. hard-query 实验

```bash
python V4/run_experiment_suite.py --suite v4_hardquery
```

### 7. 消融实验

```bash
python V4/run_experiment_suite.py --suite v4_ablation_signal
python V4/run_experiment_suite.py --suite v4_ablation_budget
```

### 8. explain 实验

```bash
python V4/run_experiment_suite.py --suite v4_explain
```

### 9. 全量汇总

```bash
python V4/finalize_pipeline.py \
  --suite-name all_v4 \
  --upstream-root V4/outputs/pprag_fl_v4 \
  --downstream-root V4/outputs/rag_eval_all_v4 \
  --force-rag
```

### 10. 无人值守总控脚本

```bash
bash run_v4_all.sh
```

这个脚本会自动：

1. 按顺序运行 `v4_main -> v4_budget -> v4_heterogeneity -> v4_hardquery -> v4_ablation_signal -> v4_ablation_budget -> v4_explain`
2. 每套 suite 完成后自动执行 `finalize_pipeline.py`
3. 自动补下游 `RAGTest`
4. 最后统一执行 `all_v4 finalize`

常用环境变量：

```bash
GPU_ID=0 BATCH_SIZE=1 SEED_LIST=0,1,2 bash run_v4_all.sh
```

如果要强制重跑下游评测：

```bash
FORCE_RAG=1 bash run_v4_all.sh
```

## 输出位置

- 上游输出：`V4/outputs/pprag_fl_v4/...`
- 下游输出：`V4/outputs/rag_eval_all_v4/...`
- 报告归档：`实验分析报告/V4/...`

## 说明

V4 不只是延续 V3 的价值密度选择，而是进一步强调：

1. 参数块是否值得上传，要同时考虑成本与下游代理收益
2. 困难客户端和困难查询应该获得更高预算优先级
3. 历史记忆应记录“收益历史”，而不仅是“被选历史”
