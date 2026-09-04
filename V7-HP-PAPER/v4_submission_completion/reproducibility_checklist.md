# Reproducibility Checklist

| Item | Status | Artifact |
| --- | --- | --- |
| Frozen V4 configuration | Complete | `opportunity_aware_semantic_generation_v4/configs/experiment_v4.json` |
| Query-level outer folds | Complete | Five train/test fingerprints in generator and selector audits |
| Generator no-leak audit | Pass | `1000` queries, `7934` effective actions, output SHA-256 `b269ab83368c329d80dc446d2e8787c640ba92903f5fb6fcd5b605e39bb9bb1e` |
| Selector no-leak audit | Pass | Five outer folds; outer-test outcomes never used for training/tuning |
| Official development per-query metrics | Complete | `outputs/official_metrics/official_hotpotqa_per_query.jsonl` |
| Same-source holdout source and disjointness | Complete | seed `44`, reproduction rate `1.0`, disjoint=`True` |
| Holdout thresholds frozen | Complete | generator/selector/prompt/support threshold unchanged |
| Generator ablation no-leak | Complete | `93,399` rows, `7,825` newly evaluated contexts, holdout_used=`False` |
| External sample | Complete | 1,000 deterministic hash-sampled 2Wiki development queries; fingerprint `recorded in audit` |
| External frozen-transfer audit | Complete | target training/tuning disabled; support threshold `0.7` not retuned |
| Faithful baseline | Complete | official RECOMP commit/checkpoint and fixed compression budget recorded |
| Environment/model revisions | Complete for primary reader | FLAN revision pinned in experiment config; reader environment manifest present |
| Statistics | Complete | paired bootstrap implementation and per-query outputs retained |
| References | Complete | `references.bib` and citation verification report |

The local submission package contains the scripts that prepare 2Wiki, apply the frozen generator/selector, run the reader, evaluate external metrics, run RECOMP, retrain generator ablations, and assemble the paper. Large per-query outputs remain in the experiment directories and on the execution server; the submission package retains their summaries and audits.
