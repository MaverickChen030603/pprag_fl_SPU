# V20 R2-B0 Subset Attainability Audit

This independent stage converts frozen REM-P Top-5 candidates into all ten
possible `Bc=3` client subsets, materializes the unchanged local retrieval and
merge contract, then evaluates gold only after those artifacts are written.

Run one dataset with:

```bash
bash run_r2b0_dataset.sh 2wikimultihopqa 1
```

For the required five-query dry run, set `R2B0_MAX_QUERIES=5`. The script keeps
the old REM-P and R2-A.5 outputs untouched, uses the same Router-Dev smoke100
split, and reuses the existing local ranker pool only after checking its
manifest and query ids.

The primary result is `raw_merged_complete_support_at_10`. No result in this
directory authorizes reader execution.
