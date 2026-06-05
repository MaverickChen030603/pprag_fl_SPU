# V6-H1 完整实验结果记录

- 生成时间：2026-06-05T14:57:33
- 实验版本：V6-H1
- 实验目的：在 V6 同预算框架基础上构建 stable hard-query 子集，检验选择性上传策略是否能在更难下游检索样本上拉开差距。
- full-pipeline 报告数量：11
- suite 报告数量：7
- stable hard-query 数量：3
- hard-query 构建规则：recall_3 == 0 OR gold_rank is None OR gold_rank > 3 OR mrr < 0.5

## 1. 全流程报告目录

- `v6h1_ablation_budget`: upstream=6, downstream=6, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_ablation_budget_2026-06-04_06-41-09`
- `v6h1_ablation_signal`: upstream=12, downstream=12, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_ablation_signal_2026-06-04_01-43-03`
- `v6h1_budget_aligned`: upstream=12, downstream=12, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_budget_aligned_2026-06-02_05-38-30`
- `v6h1_budget_aligned_stable_hardquery`: upstream=12, downstream=12, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_budget_aligned_stable_hardquery_2026-06-04_08-28-14`
- `v6h1_explain`: upstream=3, downstream=3, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_explain_2026-06-04_08-23-54`
- `v6h1_hardquery`: upstream=9, downstream=9, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_hardquery_2026-06-03_15-34-35`
- `v6h1_hardquery_stable_hardquery`: upstream=9, downstream=9, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_hardquery_stable_hardquery_2026-06-04_08-29-36`
- `v6h1_heterogeneity`: upstream=48, downstream=48, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_heterogeneity_2026-06-03_07-57-44`
- `v6h1_heterogeneity_stable_hardquery`: upstream=48, downstream=48, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_heterogeneity_stable_hardquery_2026-06-04_08-37-02`
- `v6h1_main`: upstream=15, downstream=15, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_main_2026-06-02_00-46-17`
- `v6h1_main_stable_hardquery`: upstream=15, downstream=15, dir=`实验分析报告/V6-H1/full_pipeline_v6h1_main_stable_hardquery_2026-06-04_08-26-19`

## 2. 全量下游评估聚合表

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_ablation_budget | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_ablation_signal | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2267±0.0021 | 0.7733±0.0021 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_explain | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.3163±0.0000 | 0.6837±0.0000 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2265±0.0024 | 0.7735±0.0024 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | delta_norm | 0.3071±0.0040 | 0.6929±0.0040 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v3 | 0.2240±0.0023 | 0.7760±0.0023 | 0.9600 | 0.9366 | 0.9067 | 0.8975 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v6 | 0.2240±0.0023 | 0.7760±0.0023 | 0.9600 | 0.9366 | 0.9067 | 0.8975 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | delta_norm | 0.3042±0.0060 | 0.6958±0.0060 | 0.9600 | 0.9433 | 0.9067 | 0.9016 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v3 | 0.2221±0.0019 | 0.7779±0.0019 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v6 | 0.2221±0.0019 | 0.7779±0.0019 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | delta_norm | 0.2418±0.0015 | 0.7582±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v3 | 0.2237±0.0009 | 0.7763±0.0009 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v6 | 0.2237±0.0009 | 0.7763±0.0009 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_main | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_main | num5_dir_a03_imb00_ts0_v6h1 | full | 1.0000±0.0000 | 0.0000±0.0000 | 0.9400 | 0.9133 | 0.9000 | 0.8871 |
| v6h1_main | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_main | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_main | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |


## 3. Stable Hard-Query 子集评估聚合表

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_hardquery_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2265±0.0024 | 0.7735±0.0024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_hardquery_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | delta_norm | 0.3071±0.0040 | 0.6929±0.0040 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v3 | 0.2240±0.0023 | 0.7760±0.0023 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v6 | 0.2240±0.0023 | 0.7760±0.0023 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | delta_norm | 0.3042±0.0060 | 0.6958±0.0060 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v3 | 0.2221±0.0019 | 0.7779±0.0019 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v6 | 0.2221±0.0019 | 0.7779±0.0019 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | delta_norm | 0.2418±0.0015 | 0.7582±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v3 | 0.2237±0.0009 | 0.7763±0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v6 | 0.2237±0.0009 | 0.7763±0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_main_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_main_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | full | 1.0000±0.0000 | 0.0000±0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_main_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_main_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_main_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


## 4. Hard Query 子集元信息

- per-query 来源：`/home/iiserver31/projects/FedE4RAG-main/V6-H1/outputs/rag_eval_all_v6_h1/v6h1_budget_aligned`
- baseline pattern：`hypernet-v3`
- min_hard_seeds：2
- total_per_query_rows：150
- stable_hard_query_count：3
- query_id=15, votes=3, reason=gold_not_retrieved|recall_3_zero|mrr_lt_0.5, question=Are JPM's gross margins historically consistent (not fluctuating more than roughly 2% each year)? If gross margins are n
- query_id=2, votes=3, reason=gold_not_retrieved|recall_3_zero|mrr_lt_0.5, question=As of Q2'2023, is Pfizer spinning off any large business segments?
- query_id=48, votes=3, reason=gold_not_retrieved|recall_3_zero|mrr_lt_0.5, question=Was there any drop in Cash & Cash equivalents between FY 2023 and Q2 of FY2024?
