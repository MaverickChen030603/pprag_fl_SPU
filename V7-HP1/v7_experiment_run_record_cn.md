# V7 实验数据记录

更新日期：2026-05-27

## 1. 实验路径

- 服务器项目根目录：`/home/iiserver31/projects/FedE4RAG-main`
- V7 平行路径：`/home/iiserver31/projects/FedE4RAG-main/V7`
- V7 输出目录：`/home/iiserver31/projects/FedE4RAG-main/V7/outputs`
- V7 报告目录：`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V7`
- 主日志：`/home/iiserver31/projects/FedE4RAG-main/v7_all.log`
- nohup 日志：`/home/iiserver31/projects/FedE4RAG-main/v7_nohup.log`

## 2. 已完成配置

已在服务器上从 V6 bootstrap 出 V7 平行路径，并完成以下 V7 化配置：

1. 新增 `pprag_fl_v7` 实验命名空间。
2. 新增 `num5_dir_a03_imb00_ts0_v7` task 命名。
3. 保留 V6 same-budget pipeline，加入 V7 agent 方法标签。
4. 支持 `agent_rule_v7`、`agent_bandit_v7`、`agent_policy_v7`、`agent_llm_planner_v7` 方法名记录。
5. 新增 `method_name` 和 `agent_profile` 字段，用于区分论文方法名与底层可执行 selector。
6. 清理从 V6 复制来的历史 outputs，避免混入 V7 结果。

## 3. 当前运行状态

已通过 smoke test：

- suite：`smoke`
- 方法：`agent_rule_v7`
- 轮数：`1`
- 结果：成功完成 upstream，并生成 `suite_smoke_2026-05-27_15-24-28/report.md`

已起跑 first-pass：

```bash
cd /home/iiserver31/projects/FedE4RAG-main
nohup ./run_v7_all.sh first_pass > v7_nohup.log 2>&1 &
```

first-pass 包含：

- `v7_main`
- `v7_budget_aligned`
- `v7_hardquery`

当前矩阵：

- `v7_main`: 21 upstream runs
- `v7_budget_aligned`: 21 upstream runs
- `v7_hardquery`: 15 upstream runs

合计：

- 57 upstream runs
- downstream RAG evaluation 由 `run_v7_all.sh` 在每个 upstream suite 后自动触发

## 4. 检查命令

```bash
cd /home/iiserver31/projects/FedE4RAG-main
./check_v7_status.sh
```

重点检查：

- `run_metadata.json` 数量
- `rag_eval_stdout.log` 数量
- `v7_all.log` 最新轮次
- `实验分析报告/V7` 新增报告目录

## 5. 结果采集命令

first-pass 或 full-pass 完成后运行：

```bash
cd /home/iiserver31/projects/FedE4RAG-main
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/collect_v7_results.py
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/write_v7_analysis.py
```

也可以直接运行：

```bash
./run_v7_all.sh collect
```

## 6. GitHub 同步

已完成同步：

- commit：`e5bb880`
- message：`Add V7 agentic federated RAG automation`
- branch：`main`

