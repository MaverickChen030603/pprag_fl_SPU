# V7-agent-BSP-DIAG Run / Closure Audit

Generated: 2026-07-03 14:13:58 
Project: `/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## Process State

```text
PID    PPID STAT     ELAPSED %CPU %MEM   RSS CMD
  97504       1 S    14-19:10:54  0.0  0.0  2880 bash -c bash scripts/run_v7bspdiag_suite.sh v7bspdiag_hf && bash scripts/run_v7bspdiag_official_fid.sh && /home/iiserver31/anaconda3/envs/supv2/bin/python scripts/export_reader_inputs_diag.py --methods hypernet_v6,agent_rule_v7_dynamic,agent_pm_bandit_slot,agent_bsp_memory_bandit_retrieval,agent_bsp_hf_bandit_strict,agent_bsp_hf_bandit_retrieval --seeds 0,1,2,3,4 --sample-size 50 --orderings retrieval_score,agent_priority,gold_oracle_debug --device cpu && /home/iiserver31/anaconda3/envs/supv2/bin/python scripts/analyze_v7bspdiag.py && /home/iiserver31/anaconda3/envs/supv2/bin/python scripts/generate_v7bspdiag_report.py
ERROR: Command '['ps', '-p', '155141', '-o', 'pid,ppid,stat,etime,%cpu,%mem,rss,cmd']' returned non-zero exit status 1.
```

## Current Official FiD/T5 State

- official metrics completed: 24/40
- current output dir exists: True
- current output dir size bytes: 2946392
- prediction files: 24
- reader input files: 0

The first official FiD/T5 run has exceeded the 3-hour slow-run threshold, but the Python process is still CPU-active. This audit does not terminate it. The current implementation writes predictions and metrics only after the run finishes, so file-growth checks are not sufficient by themselves.

## Hard Error Scan

- No hard errors found in the scanned logs.

## Metadata Problems

- No official metadata problems found in completed official metrics.

## Acceptance Checklist

- official_summary: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/official_fid_t5_diag_summary.csv)
- strict_hf_final: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/strict_diag_hf_final.csv)
- reader_ordering_verification: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/reader_input_ordering_verification.csv)
- cache_reuse_audit: present (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/cache_reuse_audit.md)
- gold_oracle_effect: present (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/gold_oracle_debug_effect.csv)
- per_query_alignment: present (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/per_query_alignment_final.csv)
- selection_to_qa_correlation: present (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/selection_to_qa_correlation.csv)
- true_subgroup_analysis: present (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/true_subgroup_analysis_final.csv)
- representative_cases: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/representative_cases_final.md)
- statistical_tests: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/analysis/statistical_tests_final.csv)
- final_landing_report: missing (/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/reports/v7_agent_bsp_diag_final_landing_report_20260619.md)

## Current Decision

Decision B is not yet proven, but it remains the leading risk: reader ordering may not be connected to actual FiD/T5 inputs. The final decision is blocked until official runs finish or the current slow run is explicitly terminated and repaired.

## Immediate Next Actions

1. Keep the current official run alive while CPU activity remains high and no hard error appears.
2. If the first official run remains unfinished after the next monitoring window, inspect stack/profiling or migrate official eval to a faster device/config while preserving logs.
3. After official metrics exist, regenerate reader input export with full method/seed/order coverage and run hash verification before interpreting QA metrics.
4. Replace placeholder representative cases and statistical summaries only after per-query official predictions are available.
