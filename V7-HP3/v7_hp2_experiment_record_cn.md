# V7-HP3 实验记录

## 目标

V7-HP3 是 V7-HP1 的平行分支，目标是验证 reader-aware agent reward 是否能把上游 block selection 的优势进一步对齐到 HotpotQA 下游生成式 QA 指标。

## 与 V7-HP1 的差异

1. 新增 HotpotQA validation/dev 分层随机 300 条评估集，由 `prepare_hotpot_dev_stratified_sample.py` 生成。
2. 新增 reader feedback proxy，将 reader 侧偏好的 evidence-supporting block prior、client hardness 与 block delta utility 混合进 `downstream_utility_ema`。
3. 主套件 `hp2_reader_aligned` 同时比较 no-reader memory/tail agent 与 reader-aware memory/tail agent。
4. 下游自动评估包含 official fullwiki supporting-fact 指标与 T5-small reader answer EM/F1。

## 主套件

- suite: `hp2_reader_aligned`
- experiment: `pprag_fl_v7_hp3`
- seeds: `0,1,2`
- methods per seed:
  - `hypernet_v6`
  - `adaptive_v6`
  - `agent_tail_v7hp2`
  - `agent_memory_v7hp2`
  - `agent_tail_reader_v7hp2`
  - `agent_memory_reader_v7hp2`

## 执行命令

```bash
nohup env ROUNDS=12 SEEDS=0,1,2 GPU=1 READER_GPU=2 BATCH_SIZE=1 MAX_EVAL=300 \
  V7-HP3/run_v7_hp3_full_pipeline.sh \
  > V7-HP3/logs/v7_hp3_full_pipeline.nohup.log 2>&1 &
```

## 输出

- training outputs: `V7-HP3/outputs/pprag_fl_v7_hp3/hp2_reader_aligned`
- official eval: `V7-HP3/outputs/hotpot_official_fullwiki_dev300`
- reader eval: `V7-HP3/outputs/hotpot_reader_fullwiki_t5small_dev300`
- final report: `实验分析报告/V7-HP3/v7_hp3_reader_alignment_latest.md`

## 当前状态

2026-06-08 已完成 smoke，正式 full pipeline 已在服务器后台启动。
