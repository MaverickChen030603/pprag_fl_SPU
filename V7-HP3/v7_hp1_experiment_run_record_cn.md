# V7-HP3 实验运行记录

## 2026-06-02

- 创建 `V7-HP3` 平行目录，基于 `V7-H1` agent 框架，排除历史 outputs。
- 引入 HotpotQA fullwiki 派生数据：`FedE/select_data_hotpot_train_5000.json`。
- 合并 `rawdata_path`、`rag_dataset=hotpot_qa`、`rag_hotpot_split=validation` 配置。
- 新增 HP3 suite、strict eval、自动中文分析报告、root 级执行/状态/同步脚本。
- 待执行：smoke 起跑检查；smoke 通过后后台启动 full pipeline。

## Smoke 起跑检查

- 时间：2026-06-02 15:46 JST
- 结果：通过
- 完成：1 run，Hotpot task 生成成功，summary/HP3 strict eval/自动中文报告均成功生成
- smoke 报告：`实验分析报告/V7-HP3/v7_hp3_auto_analysis_20260602_154645.md`

## Full pipeline

- 时间：2026-06-02 15:47 JST
- 状态：已后台启动
- 命令：`HP3_GPU=0 HP3_BATCH_SIZE=1 HP3_ROUNDS=12 HP3_SEED_LIST=0,1,2 ./run_v7_hp3_all.sh full`
- 日志：`v7_hp3_full.nohup.log` 与 `v7_hp3_all.log`

## V7 agent core 优化

- 时间：2026-06-02
- 新增 `agent_core.py`：实现 `AgentMemory`、`AgentReward`、`AgentScorer`。
- `AgentMemory` 支持 utility EMA、hard-query EMA、rarity EMA、selection EMA 与 mask instability penalty。
- `AgentScorer` 支持 `agent_rule_v7` 与 `agent_bandit_v7`，并提供 `stability_focused`、`hard_query_focused`、`diversity_focused` 三种评分模式。
- `UploadSelector` 已接入 agent scorer；agent 只改变 block 排序，最终仍由同一个 `budget_topk` 强制裁剪，符合 Same-Budget Protocol。
- `run_upstream.py` 已修正：`agent_rule_v7` / `agent_bandit_v7` 不再伪装成 `hypernet_v6`，而是真正进入 agent selection strategy。
- 验证：单元级 scorer/selector 通过，`agent_rule_v7 --dry-run` 显示 `selection_strategy=agent_rule_v7`。

## 2026-06-02 rare suite crash fix

- 现象：`hp2_rare_bridge_tail` 第一个 run 在 `_log_round()` 触发 `NameError: selection_metadata is not defined`。
- 原因：client update 中已有 `selection_metadata`，但日志汇总函数没有读取该字段。
- 修复：在 `V7-HP3/fedrag_selective_upload.py` 的 `_log_round()` 中读取并兜底 `selection_metadata`。
- 操作：修复后重启 full pipeline；已完成的 `hp2_multihop_hard` 会由 suite runner 自动跳过。
