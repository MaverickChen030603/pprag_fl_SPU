# V7-HP1 实验运行记录

## 2026-06-02

- 创建 `V7-HP1` 平行目录，基于 `V7-H1` agent 框架，排除历史 outputs。
- 引入 HotpotQA fullwiki 派生数据：`FedE/select_data_hotpot_train_5000.json`。
- 合并 `rawdata_path`、`rag_dataset=hotpot_qa`、`rag_hotpot_split=validation` 配置。
- 新增 HP1 suite、strict eval、自动中文分析报告、root 级执行/状态/同步脚本。
- 待执行：smoke 起跑检查；smoke 通过后后台启动 full pipeline。

## Smoke 起跑检查

- 时间：2026-06-02 15:46 JST
- 结果：通过
- 完成：1 run，Hotpot task 生成成功，summary/HP1 strict eval/自动中文报告均成功生成
- smoke 报告：`实验分析报告/V7-HP1/v7_hp1_auto_analysis_20260602_154645.md`

## Full pipeline

- 时间：2026-06-02 15:47 JST
- 状态：已后台启动
- 命令：`HP1_GPU=0 HP1_BATCH_SIZE=1 HP1_ROUNDS=12 HP1_SEED_LIST=0,1,2 ./run_v7_hp1_all.sh full`
- 日志：`v7_hp1_full.nohup.log` 与 `v7_hp1_all.log`
