# V20 LR Memory-Conditioned Local Retrieval Recovery

Run LR-0 first:

```bash
bash run_lr0_dataset.sh 2wikimultihopqa 1
```

Use `LR_MAX_QUERIES=5` for the required dry run. It freezes D0/D1/O1
selections, materializes original-query dense ranks through depth 50, compares
L0 against R2-B0, and writes the C0--C3 audit.

Only when both full LR-0 decisions permit it:

```bash
bash run_lr1_dataset.sh 2wikimultihopqa 1
```

LR-1 adds the pre-registered L1--L4 matrix. Both scripts perform two complete,
non-overwriting runs. Reader and final test remain forbidden.

`local_rankings_*.jsonl` contains only canonical retrieval evidence and is
byte-compared across runs. Per-query elapsed time is retained separately in
`local_timing_*.jsonl` and is intentionally excluded from that comparison. To
preserve a failed or exploratory attempt, use the same explicit name for both
stages, for example `LR_ATTEMPT=timing_split_fix_20260806 bash run_lr0_dataset.sh ...`.
