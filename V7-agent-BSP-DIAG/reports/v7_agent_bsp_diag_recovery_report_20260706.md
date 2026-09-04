# V7-agent-BSP-DIAG 恢复记录

时间：2026-07-06 13:09 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 状态检查

原主流程 PID 文件仍指向：

- `runs/v7bspdiag_all.pid`: `97504`

但实际检查时：

- `PID 97504` 不存在
- `PID 155046` 不存在
- 未发现活跃的 `run_hotpot_official_eval.py` / `run_v7bspdiag_official_fid.sh` 子进程

说明原 official pipeline 已中断。日志未显示明确 `Traceback` / OOM / tokenizer error，最后停在：

- `[29/40] agent_pm_bandit_slot seed=3`

## 已完成计数

恢复前计数：

- `official_metrics = 28`
- `per_query_official = 28`
- `predictions = 28`
- `reader_inputs = 0`
- `reader_verification = 0`

## 恢复动作

未重跑 training，仅恢复 official + 后续 export / analysis / report 链：

```bash
bash scripts/run_v7bspdiag_official_fid.sh &&
python scripts/export_reader_inputs_diag.py --methods hypernet_v6,agent_rule_v7_dynamic,agent_pm_bandit_slot,agent_bsp_memory_bandit_retrieval,agent_bsp_hf_bandit_strict,agent_bsp_hf_bandit_retrieval --seeds 0,1,2,3,4 --sample-size 50 --orderings retrieval_score,agent_priority,gold_oracle_debug --device cpu &&
python scripts/analyze_v7bspdiag.py &&
python scripts/generate_v7bspdiag_report.py
```

新恢复进程：

- resume PID：`247136`
- official launcher：`247139`
- 当前 official eval：`247337`
- resume log：`runs/v7bspdiag_resume_20260706_130930.nohup.log`
- resume pidfile：`runs/v7bspdiag_resume_20260706_130930.pid`

## 恢复验证

恢复日志显示已正确跳过完成的 1-28 个 run：

- `[1/40]` 到 `[28/40]`: `SKIP`
- `[29/40]`: `RUN agent_pm_bandit_slot seed=3`

当前恢复状态正常。

