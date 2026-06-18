from __future__ import annotations
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[1]; A=BASE/'analysis'; R=BASE/'reports'; R.mkdir(exist_ok=True)
def read(p):
    p=Path(p)
    try: return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
    except Exception: return pd.DataFrame()
def fmt(v):
    if pd.isna(v): return ''
    return f'{v:.6g}' if isinstance(v,float) else str(v)
def table(df,n=18):
    if df.empty: return 'Not available yet.'
    df=df.head(n); cols=list(df.columns); lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows(): lines.append('| '+' | '.join(fmt(r[c]) for c in cols)+' |')
    return '\n'.join(lines)
strict=read(A/'strict_diagnostic_diag_summary.csv')
off=read(A/'official_fid_t5_diag_all_runs.csv')
sens=read(A/'reader_sensitivity_summary_final.csv')
ver=read(A/'reader_input_ordering_verification.csv')
stat=read(A/'statistical_tests_diag.csv')
sub=read(A/'true_subgroup_analysis_final.csv')
corr=read(A/'selection_to_qa_correlation.csv')
mem=''
cache=(A/'cache_reuse_audit.md').read_text(encoding='utf-8') if (A/'cache_reuse_audit.md').exists() else 'Not available yet.'
text=f'''# V7-agent-BSP-DIAG Complete Diagnostic Report

Generated: 2026-06-18

## 1. Background

V7 moved from PM full to BSP because direct memory-weighted block scoring did not beat dynamic planning. BSP then showed strong same-budget slot-planning behavior, but true FiD/T5 endpoint metrics remained flat. BSP-DIAG is a diagnostic closure experiment: verify reader inputs, audit cache reuse, align per-query selection with QA, and test a clean history+failure-only bandit.

## 2. Current Result Recap

BSP strict diagnostics previously ranked `agent_pm_bandit_slot` slightly above BSP memory bandit, while history and failure state showed clear positive ablation signals. Official FiD/T5 remained nearly unchanged across methods.

## 3. Same-Budget Constraint

All strict diagnostic rows must retain `avg_topk=3.0` and `budget_std=0.0`. Official metadata has been patched in DIAG to write `avg_topk` fallback from `selective_topk_blocks`.

## 4. Reader Input Verification

{table(ver)}

## 5. Cache Reuse Audit

{cache}

## 6. Gold Oracle Debug

See `analysis/gold_oracle_debug_effect.csv`. `gold_oracle_debug` is diagnostic only and must not be used as a formal main result.

## 7. agent_bsp_hf_bandit Design

`agent_bsp_hf_bandit_strict` and `agent_bsp_hf_bandit_retrieval` use only history + failure state for slot-level planning. Rarity state, instability penalty, direct block-score memory, and utility EMA direct scoring are disabled. Top-k remains fixed at 3.

## 8. Strict Diagnostic Results

{table(strict)}

## 9. True FiD/T5 Official Eval

{table(off)}

## 10. Reader Sensitivity Final

{table(sens)}

## 11. Per-Query Alignment

See `analysis/per_query_alignment_final.csv`.

## 12. Selection-to-QA Correlation

{table(corr)}

## 13. True Subgroup Analysis

{table(sub)}

## 14. Representative Cases

See `analysis/representative_cases_diag.md`.

## 15. Statistical Tests

{table(stat)}

## 16. Paper-Usable Conclusions

- Same-budget slot-level planning changes strict multihop retrieval behavior.
- History and failure state are the most credible positive BSP signals.
- Endpoint FiD/T5 metrics remain insensitive unless reader input verification shows otherwise.

## 17. Not Yet Paper-Usable

- Do not claim BSP-DIAG or BSP has stable endpoint QA improvement until true FiD/T5/subgroup results support it.
- Do not use `gold_oracle_debug` as a main result.

## 18. Limitations

The current reader path may be insensitive to evidence ordering, or the ordering control may not be fully connected. The report must distinguish these two cases using input hashes.

## 19. Next Steps

Finish HF runs, export reader inputs for HF, refresh sensitivity after BSP grid completion, and replace placeholder representative cases with concrete aligned cases.
'''
out=R/'v7_agent_bsp_diag_complete_report_20260618.md'; out.write_text(text,encoding='utf-8'); print(out)
