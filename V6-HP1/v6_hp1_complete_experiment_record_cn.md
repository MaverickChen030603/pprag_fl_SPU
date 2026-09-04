# V6-HP1 完整实验结果记录

- 生成时间：2026-06-10T02:20:02
- 实验版本：V6-HP1
- 数据集：HotpotQA (fullwiki)
- 上游训练：Hotpot train split 生成的 question-supporting-facts 配对语料
- 下游评测：Hotpot validation split，same-budget / heterogeneity / hardquery / ablation 全流程
- 全流程报告数量：8
- 上游 suite 报告数量：7

## 1. 报告目录

- `all_v6_hp1`: upstream=93, downstream=93, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_all_v6_hp1_2026-06-10_02-20-01`
- `v6hp1_ablation_budget`: upstream=6, downstream=6, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_ablation_budget_2026-06-09_21-07-16`
- `v6hp1_ablation_signal`: upstream=12, downstream=12, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_ablation_signal_2026-06-09_09-43-19`
- `v6hp1_budget_aligned`: upstream=12, downstream=12, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_budget_aligned_2026-06-05_04-59-11`
- `v6hp1_explain`: upstream=3, downstream=3, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_explain_2026-06-10_02-19-49`
- `v6hp1_hardquery`: upstream=9, downstream=9, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_hardquery_2026-06-08_11-46-33`
- `v6hp1_heterogeneity`: upstream=36, downstream=36, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_heterogeneity_2026-06-07_20-07-46`
- `v6hp1_main`: upstream=15, downstream=15, dir=`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V6-HP1/full_pipeline_v6hp1_main_2026-06-04_08-24-19`

## 2. Seed 聚合结果

### all_v6_hp1
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.3163±0.0000, reduction=0.6837±0.0000, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `delta_norm` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.3177±0.0023, reduction=0.6823±0.0023, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2235±0.0025, reduction=0.7765±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v6` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2235±0.0025, reduction=0.7765±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `random` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8967±0.0047, NDCG=0.8830±0.0035
- `delta_norm` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.3059±0.0067, reduction=0.6941±0.0067, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2228±0.0022, reduction=0.7772±0.0022, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v6` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2228±0.0022, reduction=0.7772±0.0022, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `random` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `full` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=1.0000±0.0000, reduction=0.0000±0.0000, cos_3=0.9400±0.0000, recall_3=0.9066±0.0047, mrr=0.8578±0.0016, NDCG=0.8551±0.0032
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
### v6hp1_ablation_budget
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
### v6hp1_ablation_signal
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
### v6hp1_budget_aligned
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
### v6hp1_explain
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.3163±0.0000, reduction=0.6837±0.0000, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
### v6hp1_hardquery
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
### v6hp1_heterogeneity
- `delta_norm` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.3177±0.0023, reduction=0.6823±0.0023, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2235±0.0025, reduction=0.7765±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v6` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2235±0.0025, reduction=0.7765±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `random` task=`num5_dir_a005_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8967±0.0047, NDCG=0.8830±0.0035
- `delta_norm` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.3059±0.0067, reduction=0.6941±0.0067, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2228±0.0022, reduction=0.7772±0.0022, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v6` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2228±0.0022, reduction=0.7772±0.0022, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `random` task=`num5_dir_a01_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
### v6hp1_main
- `delta_norm` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2863±0.0025, reduction=0.7137±0.0025, cos_3=0.9400±0.0000, recall_3=0.9033±0.0000, mrr=0.8800±0.0000, NDCG=0.8675±0.0000
- `full` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=1.0000±0.0000, reduction=0.0000±0.0000, cos_3=0.9400±0.0000, recall_3=0.9066±0.0047, mrr=0.8578±0.0016, NDCG=0.8551±0.0032
- `hypernet_v3` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `hypernet_v6` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2254±0.0011, reduction=0.7746±0.0011, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.8800±0.0000, NDCG=0.8736±0.0000
- `random` task=`num5_dir_a03_imb00_ts0_v6hp1`: payload=0.2575±0.0015, reduction=0.7425±0.0015, cos_3=0.9400±0.0000, recall_3=0.9133±0.0000, mrr=0.9000±0.0000, NDCG=0.8855±0.0000
