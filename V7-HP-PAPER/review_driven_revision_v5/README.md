# V7-HP-PAPER Review-Driven Revision V5

This directory is an additive major revision workspace. It does not modify the
frozen V4 paper or result directories.

## Reproduce

On the experiment server:

```bash
cd /home/iiserver31/projects/FedE4RAG-main
bash V7-HP-PAPER/review_driven_revision_v5/run_review_driven_revision_v5.sh
```

Set `SKIP_RECOMP_DEV=1` or `SKIP_LITE_DEV=1` only when the corresponding
development artifacts have already passed their audits. GPU stages use four
shards and resume from existing JSONL outputs.

## Protocol Boundaries

- HotpotQA development: validation indices `[0, 1000)`.
- Frozen confirmatory holdout: `[1000, 4000)`.
- Untouched revision holdout: `[4000, 7405)`; opened only after Lite was frozen.
- RECOMP budget is frozen at 660 tokens before the 3,000-query evaluation.
- 2Wiki calibration uses train examples; the fixed 1,000-query evaluation split
  never supplies calibration labels.
- Unknown historical times remain `[NEEDS MEASUREMENT]` or `[NOT AVAILABLE]`.

The main execution entrypoints are numbered `00` through `08`. Final paper and
rebuttal artifacts are written under `paper/` and the V5 directory root.
