# V20 R2-A.6: Representative Evidence Memory Profile (REMP)

This is a recovery gate for client-resource representation, not a router, selector, reader, or QA experiment. It asks whether a fixed 32-unit memory built from client-local Router-Train documents can recover complete support-client sets more reliably than the inherited single centroid and K-means multi-prototype controls.

## Entry points

```bash
bash run_recovery_dev_all.sh
```

The Dev entry point freezes `Recovery-Dev` (unused Router-Dev indices 100--199) and `Recovery-Holdout` (unread first 300 Router-Holdout rows), performs split and unit-membership no-leak audits, builds `R0..R3`, evaluates exactly `S0/S1/S2`, repeats query-level results, and writes the go/no-go report. It does not run holdout automatically unless the gate selects an eligible shared method.

```bash
bash run_recovery_holdout_all.sh
```

The Holdout entry point is guarded by the Dev decision. It freezes the selected `(strategy, pooling)`, performs three profile-construction seed checks, evaluates only `B0 + selected method`, and keeps Reader blocked.

## Required deliverables

- `protocol/recovery_split_manifest.json`
- `protocol/recovery_preregistration.md`
- `protocol/no_leak_audit.json`
- `memory_profiles/client_memory_units.jsonl`
- `memory_profiles/memory_statistics.csv`
- `candidate_generation/recovery_dev_results.csv`
- `candidate_generation/per_query_client_ranks.jsonl`
- `efficiency/profile_costs.csv`
- `efficiency/quality_storage_pareto.csv`
- `reports/recovery_go_no_go.md`
- `reports/next_method_decision.json`
- `reports/reader_start_decision.json`

No result may be used to start a reader in R2-A.6. The permanent value of `reader_start_decision.json` is `blocked_before_reader`.
