# V7-HP V4 同源 3,000-query Frozen Scale-Up 报告

## 1. 目的与约束

本阶段检验 1,000-query 开发结果能否在更大、未参与调参的同源 HotpotQA 样本上复现。3,000 条 query 来自与原开发集完全相同的 `hotpot_qa/distractor/validation`，沿用 seed 44 的固定打乱顺序，并取开发 1,000 之后的互斥切片。

- 开发/scale-up query overlap: **0**。
- 原 1,000 source reconstruction: **true**。
- 原 1,000 baseline title-order reproduction: **1000/1000**。
- Baseline: `HybridSoftRetriever(alpha=0.55, uniform weights, top_k=5)`。
- BM25-only top-5 substitution: **未使用**。
- Generator、selector threshold、reader prompt/decoding、support threshold: **均未在 3,000 条上调参**。

## 2. 冻结执行

- Frozen context queries: **3,000**。
- Generator effective actions: **23,724**。
- Selector interventions: **774/3,000 (25.8%)**。
- Official sentence-support threshold: **0.7**，来自原 1,000 五折一致阈值。

## 3. FLAN-T5-Large Official Metrics

| Metric | Baseline | V4 selected | Delta | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer_em | 0.5013 | 0.5093 | +0.0080 | [+0.0017, +0.0147] | 0.0200 |
| answer_f1 | 0.6183 | 0.6271 | +0.0088 | [+0.0023, +0.0152] | 0.0096 |
| sp_em | 0.0700 | 0.0737 | +0.0037 | [+0.0003, +0.0070] | 0.0404 |
| sp_f1 | 0.4930 | 0.4987 | +0.0056 | [+0.0031, +0.0083] | 0.0004 |
| joint_em | 0.0460 | 0.0483 | +0.0023 | [-0.0007, +0.0053] | 0.1468 |
| joint_f1 | 0.3292 | 0.3356 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |

## 4. UnifiedQA-T5-Large Official Metrics

| Metric | Baseline | V4 selected | Delta | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer_em | 0.4407 | 0.4527 | +0.0120 | [+0.0057, +0.0187] | <0.0002 |
| answer_f1 | 0.5662 | 0.5772 | +0.0110 | [+0.0050, +0.0172] | <0.0002 |
| sp_em | 0.0700 | 0.0737 | +0.0037 | [+0.0003, +0.0070] | 0.0404 |
| sp_f1 | 0.4930 | 0.4987 | +0.0056 | [+0.0031, +0.0083] | 0.0004 |
| joint_em | 0.0410 | 0.0433 | +0.0023 | [-0.0003, +0.0050] | 0.1044 |
| joint_f1 | 0.3045 | 0.3130 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

## 5. 稳健性判断

- Dual-reader answer direction consistent: **true**。
- Systematic answer degradation: **false**。
- FLAN answer-drop rate: **2.00%**。
- UnifiedQA answer-drop rate: **1.73%**。

结果应按效应量、置信区间和双 reader 一致性解释。该 3,000-query 结果是同源规模化验证；它不等价于跨数据集 external validation。

## 6. 产物

- Context provenance audit: `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/same_source_context_audit.json`
- Frozen generator audit: `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_generator_audit.json`
- Frozen selector manifest: `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_selector_manifest.json`
- Official dual-reader summary: `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/opportunity_aware_semantic_generation_v4/outputs/scaleup/official_metrics/scaleup_official_summary.json`
