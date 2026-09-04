# Subset Smoke Test Report

- Run directory: `experiments/v6_hp_hyper_next/results/subset_smoke_20260617_193626`
- Method: V6-HP-hyper anchor retriever
- Payload ratio recorded from anchor config: `0.070134`
- Note: RAGTest loaded 1500 HotpotQA validation examples so hard subset query ids up to 1493 can match, but each smoke subset contains only 50 target queries.
- Note: `smoke_hard_50` intentionally uses the first 50 ranked hard queries, so all-zero retrieval metrics are possible and indicate a strong stress subset rather than a loader failure.

| case | subset | per_query_lines | F1 | EM | MRR | NDCG | Recall@3 | Hit@10 | hotpot_confirmed | json_regression | runtime_sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| smoke_all_50 | smoke_all_50.json | 50 | 0.88 | 0.8 | 0.95 | 0.8997091350595096 | 0.8900 | 0.96 | True | False | 112.52 |
| smoke_hard_50 | smoke_hard_50.json | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0000 | 0.0 | True | False | 111.60 |

## Verdict

- PASS: both smoke tests produced exactly 50 per-query rows.
- PASS: logs confirm `hotpot_split: validation` and `hotpot_examples: 1500`.
- PASS: no old financial/json_download regression patterns were found in smoke stdout.
- PASS: `--query-subset` is compatible with the new `queries`-based subset JSON files.

Task B/C/D can proceed, but multi-seed large experiments should continue to monitor `/home` usage.
