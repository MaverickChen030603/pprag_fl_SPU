# V7-agent-BSP-DIAG 恢复记录

时间：2026-07-07 13:38 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 状态检查

昨日恢复进程已不在：

- `runs/v7bspdiag_resume_20260706_130930.pid`: `247136`
- 检查时 `247136` / `247139` / `247337` 均不存在
- 未发现活跃的 DIAG official 子进程

恢复前已完成：

- `official_metrics = 29`
- `per_query_official = 29`
- `predictions = 29`
- `reader_inputs = 0`
- `reader_verification = 0`

日志显示昨日恢复已补完 `[29/40] agent_pm_bandit_slot seed=3`，并停在 `[30/40] agent_pm_bandit_slot seed=4`。

## 恢复动作

未重跑 training，仅恢复 official + export / analysis / report 链。新恢复进程：

- resume PID：`563416`
- official launcher：`563419`
- 当前 official eval：`563653`
- 当前 run：`[30/40] agent_pm_bandit_slot seed=4`
- resume log：`runs/v7bspdiag_resume_20260707_133728.nohup.log`
- resume pidfile：`runs/v7bspdiag_resume_20260707_133728.pid`

## 恢复验证

恢复日志显示：

- `[1/40]` 到 `[29/40]`: `SKIP`
- `[30/40]`: `RUN agent_pm_bandit_slot seed=4`

当前恢复状态正常。

