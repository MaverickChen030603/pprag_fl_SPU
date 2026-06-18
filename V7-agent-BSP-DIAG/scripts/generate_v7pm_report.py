from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE / "analysis"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


def csv_block(path: Path) -> str:
    if not path.exists():
        return "_待生成_"
    try:
        df = pd.read_csv(path)
    except Exception:
        return "_待生成_"
    if df.empty:
        return "_待生成_"
    return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    out = REPORTS / "v7_agent_pm_complete_report_20260616.md"
    text = f"""# V7-agent-PM 完整实验报告

生成时间：{now}

## 1. 实验目的

本轮实验在 V7-agent-2 基础上验证 Planning-Memory Agent 是否能在 fixed same-budget top-k=3 下超过 `agent_rule_v7_dynamic`，并检查 memory/failure/rarity/instability 是否真实改变 upload block selection。

## 2. 与 V7-agent-2 的关系

V7-agent-2 已证明 dynamic early-slot 在 strict diagnostic 上有正信号。本轮 V7-agent-PM 新增 PM scoring、memory ablation、dynamic planning ablation、bandit slot policy、true FiD/T5 reader 检查与 subgroup/per-query 分析。

## 3. 方法设计

PM score 使用 base delta、early prior、coverage gain、utility EMA、failure recovery、rarity、instability penalty。所有策略必须保持 top-k=3。

## 4. Same-budget 约束确认

见 strict diagnostic 表中的 `avg_topk` 与 `budget_std`。

## 5. Strict Diagnostic 结果

{csv_block(ANALYSIS / 'strict_diagnostic_summary.csv')}

## 6. True FiD/T5 Official Eval 结果

{csv_block(ANALYSIS / 'official_fid_t5_summary.csv')}

## 7. Dynamic Planning Ablation

{csv_block(ANALYSIS / 'dynamic_ablation_summary.csv')}

## 8. Memory Ablation

{csv_block(ANALYSIS / 'memory_ablation_summary.csv')}

## 9. Bandit Slot Policy

{csv_block(ANALYSIS / 'bandit_slot_summary.csv')}

## 10. Subgroup Analysis

{csv_block(ANALYSIS / 'subgroup_analysis.csv')}

## 11. Per-query Behavior Case Study

见 `analysis/per_query_behavior.csv` 与 `analysis/representative_cases.md`。

## 12. 统计检验

{csv_block(ANALYSIS / 'statistical_tests.csv')}

## 13. 失败与限制

若 FiD/T5 reader fallback、OOM、缺失 run 或 official baseline 不完整，必须在本节记录。当前脚本已先修复 sentencepiece 并强制 smoke check T5 tokenizer。

## 14. 下一步建议

优先比较 `agent_rule_v7_dynamic` 与 `agent_pm_dynamic_full` 的 strict 和 true FiD/T5 指标；若整体差异小，聚焦 hard-query、rare-domain、hard-client 子集。

## 15. 可写入论文的结论段

V7-agent-PM tests whether planning-memory upload selection can improve same-budget federated RAG beyond dynamic early-slot heuristics. The final claim should depend on true FiD/T5 and subgroup evidence rather than fallback reader scores.
"""
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
