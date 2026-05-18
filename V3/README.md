# V3: Heterogeneity-Aware Value-Guided Selective Upload

`V3/` 是在现有 `SUP_v3` 基础上独立整理出来的新版实验目录，目标是把“超网络选择性上传”从 `V2` 的可行性验证，推进到：

- 面向客户端异构性的条件化超网络
- 面向通信性价比的 `value-aware` 打分
- 面向动态预算的自适应选择
- 保持上游 FL、下游 RAG 与报告归档的完整闭环

## 目录说明

- `run_upstream.py`：单次上游实验入口
- `run_experiment_suite.py`：V3 实验套件入口
- `fedrag_selective_upload.py`：上游联邦训练主流程
- `hypernetwork.py`：V3 超网络与特征编码
- `history_memory.py`：客户端历史记忆
- `budget_allocator.py`：自适应预算与价值密度分配
- `explain_analyzer.py`：解释性分析
- `run_all_rag_eval.py`：批量下游评测
- `finalize_pipeline.py`：上下游汇总与总报告
- `report_generator.py`：单实验 / suite / full pipeline 报告

## V3 的关键方法开关

- `selection_strategy`
  - `full`
  - `random`
  - `static_top`
  - `delta_norm`
  - `hypernet_v2`
  - `hypernet_v3`
- `score_mode`
  - `importance`
  - `value`
- `budget_mode`
  - `fixed`
  - `adaptive`
- `use_client_embedding`
- `use_history_features`
- `layerwise_budget`

## 快速开始

### 1. dry-run 查看单次配置

```bash
python V3/run_upstream.py \
  --strategy hypernet_v3 \
  --topk 3 \
  --warmup 1 \
  --score-mode value \
  --budget-mode adaptive \
  --dry-run
```

### 2. smoke test

```bash
python V3/run_experiment_suite.py --suite smoke
```

### 3. 主实验

```bash
python V3/run_experiment_suite.py --suite v3_main
```

### 4. 预算实验

```bash
python V3/run_experiment_suite.py --suite v3_budget
```

### 5. 异构性实验

```bash
python V3/run_experiment_suite.py --suite v3_heterogeneity
```

### 6. 消融实验

```bash
python V3/run_experiment_suite.py --suite v3_ablation_feature
python V3/run_experiment_suite.py --suite v3_ablation_budget
```

### 7. 解释性实验

```bash
python V3/run_experiment_suite.py --suite v3_explain
```

### 8. 全量汇总

```bash
python V3/finalize_pipeline.py \
  --suite-name all_v3 \
  --upstream-root V3/outputs/pprag_fl_v3 \
  --downstream-root V3/outputs/rag_eval_all_v3 \
  --force-rag
```

### 9. 无人值守总控脚本

如果希望整个 V3 流程自动串行执行，而不是每个 suite 手动接续，可以直接运行：

```bash
bash run_v3_all.sh
```

这个脚本会自动：

1. 按顺序运行 `v3_main -> v3_budget -> v3_heterogeneity -> v3_ablation_feature -> v3_ablation_budget -> v3_explain`
2. 每套 suite 完成后自动执行 `finalize_pipeline.py`
3. 自动补下游 `RAGTest`
4. 最后统一执行 `all_v3 finalize`

常用环境变量：

```bash
GPU_ID=0 BATCH_SIZE=1 SEED_LIST=0,1,2 bash run_v3_all.sh
```

如果要强制重跑下游评测：

```bash
FORCE_RAG=1 bash run_v3_all.sh
```

## 输出位置

- 上游输出：`V3/outputs/pprag_fl_v3/...`
- 下游输出：`V3/outputs/rag_eval_all_v3/...`
- 报告归档：`实验分析报告/V3/...`

## 说明

V3 默认保留 V2 的完整流程能力，但把方法核心升级成：

1. 客户端条件特征驱动的 block 打分  
2. 历史记忆辅助的重要性/价值判断  
3. 按 `value density` 进行上传排序  
4. 支持固定预算与自适应预算两种模式  
5. 记录解释性分析文件，便于后续论文绘图与分析
