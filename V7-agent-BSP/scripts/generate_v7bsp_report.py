from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE / 'analysis'
REPORTS = BASE / 'reports'
REPORTS.mkdir(exist_ok=True)
OUT = REPORTS / 'v7_agent_bsp_complete_report_20260617.md'

def csv_block(path: Path, max_rows: int = 20) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return 'Not available yet.'
    try:
        df = pd.read_csv(path)
        if len(df) > max_rows: df = df.head(max_rows)
        return df.to_markdown(index=False)
    except Exception as exc:
        return f'Could not read {path.name}: {exc}'

def main() -> None:
    text = f'''# V7-agent-BSP Complete Experiment Report

Generated: 2026-06-17

## 1. Experiment Background

V7-agent-PM showed that `agent_pm_dynamic_full` did not outperform the dynamic rule baseline, while `agent_pm_bandit_slot` produced the strongest strict diagnostic signal under fixed same-budget top-k=3. V7-agent-BSP therefore shifts the main hypothesis from block-score memory reweighting to slot-level Bandit Slot Planning.

## 2. Why BSP

The core question is whether an agent can plan communication structure, not merely rescore blocks. BSP treats the action as slot allocation over early, bridge, target, and general evidence slots while keeping the upload budget fixed.

## 3. Same-Budget Protocol

All main methods use fixed top-k=3. The analysis reports `avg_topk` and `budget_std`; any method with non-zero budget deviation must be excluded from paper-facing claims.

## 4. BSP Method Design

Implemented methods include no-memory bandit rewards, memory-state bandit rewards, and state ablations for failure, rarity, instability, and history. Memory is used as bandit state for slot allocation rather than as a direct block-score bonus.

## 5. Action, State, Reward

Action: `early_slot_num`, `bridge_guard`, `target_slot_num`, and exploration level. State includes client hardness, domain rarity, failure-memory flags, previous instability, and history availability. Rewards are encoded as strict, retrieval, and reader-aware variants.

## 6. Strict Diagnostic Summary

{csv_block(ANALYSIS / 'strict_diagnostic_bsp_summary.csv')}

## 7. Method-Balanced True FiD/T5 Eval

{csv_block(ANALYSIS / 'official_fid_t5_method_balanced.csv')}

## 8. Reader Sensitivity

{csv_block(ANALYSIS / 'reader_sensitivity_summary.csv')}

`gold_oracle_debug` is diagnostic only and must not be used as the formal main result.

## 9. True Subgroup Analysis

{csv_block(ANALYSIS / 'true_subgroup_analysis.csv')}

## 10. Per-Query Alignment Cases

See `analysis/per_query_alignment.csv` and `analysis/representative_cases_bsp.md`.

## 11. Memory State Ablation

{csv_block(ANALYSIS / 'memory_state_ablation.csv')}

## 12. Statistical Tests

{csv_block(ANALYSIS / 'statistical_tests_bsp.csv')}

## 13. Failure Case Analysis

The report should be interpreted cautiously until official eval and per-query alignment are complete. If true FiD/T5 remains flat while strict HP1 improves, the likely bottleneck is reader sensitivity or passage truncation rather than same-budget selection itself.

## 14. Paper-Facing Conclusion Template

Under a strict fixed top-k=3 protocol, V7-agent-BSP tests whether agentic clients can plan upload structure through slot-level bandit actions. Evidence should be claimed at three levels: strict diagnostic behavior, subgroup/reader-sensitivity transfer, and final true FiD/T5 endpoint transfer.

## 15. Next Steps

1. Finish all seed-balanced BSP runs.
2. Run method-balanced official FiD/T5.
3. Run reader sensitivity on the five core methods.
4. Inspect per-query alignment for hard-query and rare-domain cases.
'''
    OUT.write_text(text, encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
