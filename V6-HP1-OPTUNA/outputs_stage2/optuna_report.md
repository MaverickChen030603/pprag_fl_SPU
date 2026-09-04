# V6-HP1 Optuna Stage2 Search Report

- Trials: 24
- Completed trials: 24
- Best trial: 10
- Best objective: 0.843032
- Best payload: 0.07013433411665673
- Best payload penalty: 0.1

## Best Parameters

- `payload_penalty`: `0.1`
- `score_mode`: `value`
- `budget_mode`: `fixed`
- `hard_query_scale`: `0.8999999999999999`
- `hard_client_threshold`: `0.7200000000000001`
- `adaptive_expand_threshold`: `0.86`
- `utility_expand_threshold`: `1.5`
- `adaptive_shrink_threshold`: `0.56`
- `history_window`: `3`
- `use_hard_query_weighting`: `True`
- `layerwise_budget`: `True`

Fixed search constraints: `topk=2`, `warmup=0`, `use_utility_memory=False`.

## Best Metrics

- `DCG`: `1.0388`
- `F1`: `0.7967`
- `IDCG`: `1.1714`
- `MAP`: `0.9`
- `NDCG`: `0.8855`
- `cos_1`: `0.86`
- `cos_10`: `0.94`
- `cos_3`: `0.94`
- `cos_5`: `0.94`
- `em`: `0.72`
- `hit1`: `0.88`
- `hit10`: `0.94`
- `hit_2`: `0.94`
- `hit_4`: `0.94`
- `hit_8`: `0.94`
- `mrr`: `0.9`
- `precision`: `0.86`
- `precision_10`: `0.3833`
- `precision_3`: `0.3833`
- `precision_5`: `0.3833`
- `recall_1`: `0.7467`
- `recall_10`: `0.9133`
- `recall_3`: `0.9133`
- `recall_5`: `0.9133`
