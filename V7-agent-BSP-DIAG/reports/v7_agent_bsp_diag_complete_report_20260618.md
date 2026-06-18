# V7-agent-BSP-DIAG Complete Diagnostic Report

Generated: 2026-06-18

## 1. Background

V7 moved from PM full to BSP because direct memory-weighted block scoring did not beat dynamic planning. BSP then showed strong same-budget slot-planning behavior, but true FiD/T5 endpoint metrics remained flat. BSP-DIAG is a diagnostic closure experiment: verify reader inputs, audit cache reuse, align per-query selection with QA, and test a clean history+failure-only bandit.

## 2. Current Result Recap

BSP strict diagnostics previously ranked `agent_pm_bandit_slot` slightly above BSP memory bandit, while history and failure state showed clear positive ablation signals. Official FiD/T5 remained nearly unchanged across methods.

## 3. Same-Budget Constraint

All strict diagnostic rows must retain `avg_topk=3.0` and `budget_std=0.0`. Official metadata has been patched in DIAG to write `avg_topk` fallback from `selective_topk_blocks`.

## 4. Reader Input Verification

Not available yet.

## 5. Cache Reuse Audit

# Cache Reuse Audit

- sensitivity directories: 4
- examples: beam1_len512_agent_priority, beam1_len512_gold_oracle_debug, beam1_len512_retrieval_score, beam1_len768_retrieval_score

## sensitivity grid observed
beam_size,max_input_length,passage_ordering,n
1,512,agent_priority,45
1,512,gold_oracle_debug,45
1,512,retrieval_score,45
1,768,retrieval_score,22


## 6. Gold Oracle Debug

See `analysis/gold_oracle_debug_effect.csv`. `gold_oracle_debug` is diagnostic only and must not be used as a formal main result.

## 7. agent_bsp_hf_bandit Design

`agent_bsp_hf_bandit_strict` and `agent_bsp_hf_bandit_retrieval` use only history + failure state for slot-level planning. Rarity state, instability penalty, direct block-score memory, and utility EMA direct scoring are disabled. Top-k remains fixed at 3.

## 8. Strict Diagnostic Results

| method | n | alpha | seed | avg_budget_topk_hp1 | budget_std_hp1 | rare_client_budget_hp1 | bridge_block_recall_hp1 | early_evidence_recall_hp1 | target_block_recall_hp1 | selection_diversity_hp1 | hp1_multihop_score | event_count | avg_topk | budget_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_v6 | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.588364 | 0 | 0.294182 | 0.42 | 0.413462 | 55 | 3 | 0 |
| agent_bsp_bandit_reader | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 55 | 3 | 0 |
| agent_bsp_bandit_retrieval | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 55 | 3 | 0 |
| agent_bsp_bandit_strict | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_no_failure_state | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.373818 | 0.363636 | 0.368727 | 0.88 | 0.534516 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_no_history_state | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.536 | 0.2 | 0.368 | 0.46 | 0.46304 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_no_instability_state | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.368727 | 0.381818 | 0.375273 | 0.9 | 0.542167 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_no_rarity_state | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.412364 | 0.381091 | 0.396727 | 0.84 | 0.54696 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 10 | 0.3 | 2 | 3 | 0 | 3 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 15 | 0.3 | 2 | 3 | 0 | 3 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 55 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 10 | 0.3 | 2 | 3 | 0 | 3 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 55 | 3 | 0 |
| agent_pm_bandit_slot | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.429091 | 0.363636 | 0.396364 | 0.84 | 0.548073 | 55 | 3 | 0 |
| agent_pm_dynamic_full | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.432727 | 0.363636 | 0.398182 | 0.64 | 0.509491 | 55 | 3 | 0 |
| agent_pm_dynamic_no_memory | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.402182 | 0.345455 | 0.373818 | 0.86 | 0.535942 | 55 | 3 | 0 |
| agent_rule_v7 | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.598545 | 0.2 | 0.399273 | 0.54 | 0.503433 | 55 | 3 | 0 |
| agent_rule_v7_dynamic | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.436364 | 0.363636 | 0.4 | 0.7 | 0.522909 | 55 | 3 | 0 |
| hypernet_v6 | 5 | 0.3 | 2 | 3 | 0 | 3 | 0.588364 | 0 | 0.294182 | 0.42 | 0.413462 | 55 | 3 | 0 |

## 9. True FiD/T5 Official Eval

| source | method | seed | n_examples | answer_F1 | support_F1 | joint_F1 | support_title_recall | answer_EM | support_EM | joint_EM | avg_topk | budget_std | reader_model | beam_size | max_input_length | passage_ordering | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bsp | agent_bsp_bandit_reader | 0 | 1000 | 0.654558 | 0.506 | 0.331641 | 0.744 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-reader_k3_w1_s0_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_reader | 1 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-reader_k3_w1_s1_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_reader | 2 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-reader_k3_w1_s2_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_reader | 3 | 1000 | 0.655624 | 0.506 | 0.332274 | 0.7435 | 0.578 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-reader_k3_w1_s3_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_reader | 4 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-reader_k3_w1_s4_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_retrieval | 0 | 1000 | 0.654558 | 0.506 | 0.331641 | 0.744 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-retrieval_k3_w1_s0_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_retrieval | 1 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-retrieval_k3_w1_s1_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_retrieval | 2 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-retrieval_k3_w1_s2_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_retrieval | 3 | 1000 | 0.655624 | 0.506 | 0.332274 | 0.7435 | 0.578 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-retrieval_k3_w1_s3_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_retrieval | 4 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-retrieval_k3_w1_s4_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_strict | 0 | 1000 | 0.654558 | 0.506 | 0.331641 | 0.744 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-strict_k3_w1_s0_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_strict | 1 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-strict_k3_w1_s1_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_strict | 2 | 1000 | 0.654558 | 0.5065 | 0.331841 | 0.7445 | 0.576 | 0.209 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-strict_k3_w1_s2_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_strict | 3 | 1000 | 0.655624 | 0.506 | 0.332274 | 0.7435 | 0.578 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-strict_k3_w1_s3_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_bandit_strict | 4 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-bandit-strict_k3_w1_s4_enc0_score-downstream_value_budget-fixed_hist7_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf0r0i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_memory_bandit_reader | 0 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-memory-bandit-reader_k3_w1_s0_enc0_score-downstream_value_budget-fixed_hist11_client1_block1_hard1_util1_ep1_cov1_mem1_dyn1_ecw000_slotdyn_pmf1r1i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_memory_bandit_reader | 1 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-memory-bandit-reader_k3_w1_s1_enc0_score-downstream_value_budget-fixed_hist11_client1_block1_hard1_util1_ep1_cov1_mem1_dyn1_ecw000_slotdyn_pmf1r1i1bg1dh0/official_metrics.json |
| bsp | agent_bsp_memory_bandit_reader | 2 | 1000 | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0.576 | 0.208 | 0.108 | 3 | 0 | t5-base | 3 | 768 | retrieval_score | /home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/eval_outputs/official_fid_t5/v7bsp_bsp_methods/agent-bsp-memory-bandit-reader_k3_w1_s2_enc0_score-downstream_value_budget-fixed_hist11_client1_block1_hard1_util1_ep1_cov1_mem1_dyn1_ecw000_slotdyn_pmf1r1i1bg1dh0/official_metrics.json |

## 10. Reader Sensitivity Final

| method | seed | beam_size | max_input_length | passage_ordering | n_examples | answer_EM | answer_F1 | support_EM | support_F1 | joint_EM | joint_F1 | support_title_recall | avg_topk | budget_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_bsp_memory_bandit_reader | 0 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 1 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 2 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 3 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 4 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 0 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 1 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 2 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 3 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 4 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 0 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 1 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 2 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 3 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 4 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 0 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 1 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 2 | 1 | 512 | retrieval_score | 300 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 3 | 0 |

## 11. Per-Query Alignment

See `analysis/per_query_alignment_final.csv`.

## 12. Selection-to-QA Correlation

| signal | metric | corr |
| --- | --- | --- |
| support_title_hit | answer_F1 | 0.113837 |
| support_title_hit | support_F1 | 0.359834 |
| support_title_hit | joint_F1 | 0.253391 |
| support_title_hit | support_title_hit | 1 |

## 13. True Subgroup Analysis

| method | subgroup | n | answer_F1 | support_F1 | joint_F1 | support_title_recall | avg_topk | budget_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_bsp_bandit_reader | all | 5000 | 0.654771 | 0.5062 | 0.331888 | 0.7442 | 3 | 0 |
| agent_bsp_bandit_reader | hard_query | 2558 | 0.605427 | 0.394253 | 0.250169 | 0.5 | 3 | 0 |
| agent_bsp_bandit_reader | easy_query | 2442 | 0.706458 | 0.623464 | 0.417488 | 1 | 3 | 0 |
| agent_bsp_bandit_reader | bandit_helped | 15 | 0.817778 | 0.633333 | 0.462222 | 0.633333 | 3 | 0 |
| agent_bsp_bandit_reader | bandit_hurt | 16 | 0.2 | 0.5 | 0.1 | 0.65625 | 3 | 0 |
| agent_bsp_bandit_reader | answer_failed_baseline | 1603 | 0.0298386 | 0.505303 | 0.0172422 | 0.703681 | 3 | 0 |
| agent_bsp_bandit_reader | support_failed_baseline | 980 | 0.608485 | 0.00255102 | 0.00255102 | 0.721429 | 3 | 0 |
| agent_bsp_bandit_retrieval | all | 5000 | 0.654771 | 0.5062 | 0.331888 | 0.7442 | 3 | 0 |
| agent_bsp_bandit_retrieval | hard_query | 2558 | 0.605427 | 0.394253 | 0.250169 | 0.5 | 3 | 0 |
| agent_bsp_bandit_retrieval | easy_query | 2442 | 0.706458 | 0.623464 | 0.417488 | 1 | 3 | 0 |
| agent_bsp_bandit_retrieval | bandit_helped | 15 | 0.817778 | 0.633333 | 0.462222 | 0.633333 | 3 | 0 |
| agent_bsp_bandit_retrieval | bandit_hurt | 16 | 0.2 | 0.5 | 0.1 | 0.65625 | 3 | 0 |
| agent_bsp_bandit_retrieval | answer_failed_baseline | 1603 | 0.0298386 | 0.505303 | 0.0172422 | 0.703681 | 3 | 0 |
| agent_bsp_bandit_retrieval | support_failed_baseline | 980 | 0.608485 | 0.00255102 | 0.00255102 | 0.721429 | 3 | 0 |
| agent_bsp_bandit_strict | all | 5000 | 0.654771 | 0.5062 | 0.331888 | 0.7442 | 3 | 0 |
| agent_bsp_bandit_strict | hard_query | 2558 | 0.605427 | 0.394253 | 0.250169 | 0.5 | 3 | 0 |
| agent_bsp_bandit_strict | easy_query | 2442 | 0.706458 | 0.623464 | 0.417488 | 1 | 3 | 0 |
| agent_bsp_bandit_strict | bandit_helped | 15 | 0.817778 | 0.633333 | 0.462222 | 0.633333 | 3 | 0 |

## 14. Representative Cases

See `analysis/representative_cases_diag.md`.

## 15. Statistical Tests

| method_a | method_b | metric | n | mean_delta |
| --- | --- | --- | --- | --- |
| agent_bsp_hf_bandit_strict | agent_pm_bandit_slot | hp1_multihop_score | 0 |  |
| agent_bsp_hf_bandit_retrieval | agent_pm_bandit_slot | hp1_multihop_score | 0 |  |
| agent_bsp_hf_bandit_retrieval | agent_bsp_memory_bandit_retrieval | hp1_multihop_score | 0 |  |
| agent_bsp_memory_bandit_retrieval | agent_bsp_memory_bandit_no_history_state | hp1_multihop_score | 15 | 0.0838618 |
| agent_bsp_memory_bandit_retrieval | agent_bsp_memory_bandit_no_failure_state | hp1_multihop_score | 15 | 0.0123855 |

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
