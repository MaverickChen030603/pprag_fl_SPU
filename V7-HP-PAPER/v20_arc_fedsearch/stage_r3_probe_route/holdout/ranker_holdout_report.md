# R3 Lightweight Probe Ranker Fresh-Holdout Report

- Status: `probe_routing_method_confirmed`
- Holdout: sealed R2-A.6 Recovery-Holdout, N=300 per dataset.
- Reader: not started. Final test: not accessed.
- Comparator: B1 static P0 Top-3. B3 is the frozen dataset-specific label-free probe comparator.

## Pre-registered Decision
- 2wikimultihopqa: three-seed success 3/3; all-three=True.
- musique: three-seed success 3/3; all-three=True.

## Main Means
- 2wikimultihopqa / B1_static_p0: complete_client_set_recall_at_3=0.4700, gold_client_recall_at_3=0.6706, local_complete_at_10=0.1967, transmitted_complete_at_15=0.1833, raw_merged_complete_at_10=0.1833, percentile_merged_complete_at_10=0.1800
- 2wikimultihopqa / B3_label_free_probe: complete_client_set_recall_at_3=0.6267, gold_client_recall_at_3=0.7978, local_complete_at_10=0.2700, transmitted_complete_at_15=0.2500, raw_merged_complete_at_10=0.2500, percentile_merged_complete_at_10=0.2467
- 2wikimultihopqa / B4_logistic_seed_20260807: complete_client_set_recall_at_3=0.6567, gold_client_recall_at_3=0.8272, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2700, raw_merged_complete_at_10=0.2700, percentile_merged_complete_at_10=0.2667
- 2wikimultihopqa / B4_logistic_seed_20260808: complete_client_set_recall_at_3=0.6567, gold_client_recall_at_3=0.8272, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2700, raw_merged_complete_at_10=0.2700, percentile_merged_complete_at_10=0.2667
- 2wikimultihopqa / B4_logistic_seed_20260809: complete_client_set_recall_at_3=0.6567, gold_client_recall_at_3=0.8272, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2700, raw_merged_complete_at_10=0.2700, percentile_merged_complete_at_10=0.2667
- musique / B1_static_p0: complete_client_set_recall_at_3=0.4900, gold_client_recall_at_3=0.7111, local_complete_at_10=0.1933, transmitted_complete_at_15=0.1467, raw_merged_complete_at_10=0.1433, percentile_merged_complete_at_10=0.1333
- musique / B3_label_free_probe: complete_client_set_recall_at_3=0.6267, gold_client_recall_at_3=0.8169, local_complete_at_10=0.2767, transmitted_complete_at_15=0.2267, raw_merged_complete_at_10=0.2100, percentile_merged_complete_at_10=0.2033
- musique / B4_logistic_seed_20260807: complete_client_set_recall_at_3=0.6533, gold_client_recall_at_3=0.8317, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2367, raw_merged_complete_at_10=0.2233, percentile_merged_complete_at_10=0.2133
- musique / B4_logistic_seed_20260808: complete_client_set_recall_at_3=0.6533, gold_client_recall_at_3=0.8317, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2367, raw_merged_complete_at_10=0.2233, percentile_merged_complete_at_10=0.2133
- musique / B4_logistic_seed_20260809: complete_client_set_recall_at_3=0.6533, gold_client_recall_at_3=0.8317, local_complete_at_10=0.2867, transmitted_complete_at_15=0.2367, raw_merged_complete_at_10=0.2233, percentile_merged_complete_at_10=0.2133
