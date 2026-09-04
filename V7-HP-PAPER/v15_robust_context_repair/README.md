# V15 Robust Context Repair

V15 is a protocol-reset experiment for complete-sequence context repair under a
fixed reader budget. It does not reuse the inspected HotpotQA validation split
as a confirmatory test. Fresh train-derived splits, retrievers, pools, readers,
metrics, risk targets, seeds, and ablations are frozen before outcome-bearing
experiments are run.

## First-stage commands

```bash
python protocol/01_protocol_and_data_audit.py --project-root /home/iiserver31/projects/FedE4RAG-main
python protocol/02_freeze_data_splits.py \
  --hotpot-source /path/to/hotpot_train.json \
  --two-wiki-source /path/to/2wiki_train.json
python -m unittest discover -s tests -v
```

The `data/sealed/` directory contains final-test labels. Development, model
selection, threshold selection, and calibration code must not read that path.

## Current checkpoint

Protocol freeze, real-corpus indexes, two-dataset retrieval smoke tests, and a
100-query HotpotQA dual-reader pilot are complete. The current decision is
`needs_more_experiments`: action-set opportunity is positive, but cross-reader
action ranking and the cheap cascade need more query-level training data.

- Current Chinese report: `reports/v15_current_report_cn.md`
- Machine-readable pilot results: `results/pilot_hotpot100/`
- Robust same-action audit: `analysis/07_robust_utility_diagnostic.py`
- Reader opportunity audit: `analysis/08_pilot_opportunity_analysis.py`
- Cheap gate and cost-aware inference: `cascade/07_train_cheap_gate.py` and
  `cascade/08_cost_aware_inference.py`

The sealed final splits must remain untouched until the robust beta, risk gate,
cascade, baseline contracts, and all preregistered ablations are frozen.
