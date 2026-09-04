# Multi-Reader Support Audit

## Frozen 3,000-query results

| Answer reader | Baseline Answer F1 | Selected Answer F1 | Delta | Baseline SP F1 | Selected SP F1 | Delta | Baseline Joint F1 | Selected Joint F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAN-T5-Large | 0.618307 | 0.627081 | +0.008773 | 0.493025* | 0.498664* | +0.005639* | 0.329182 | 0.335631 | +0.006450 |
| UnifiedQA-T5-Large | 0.566173 | 0.577188 | +0.011015 | 0.493025* | 0.498664* | +0.005639* | 0.304453 | 0.312964 | +0.008510 |

`*` Both rows share one frozen support predictor. The answer generator changes; support prediction does not.

## Supported interpretation

- The same frozen context changes have directionally positive Answer F1 under two answer readers.
- Joint point estimates are directionally consistent, but each includes the same support component.
- This is useful supporting evidence that the Answer result is not unique to one decoder.

## Unsupported interpretation

- Two independent end-to-end reader pipelines.
- Independent SP replication.
- Independent Joint replication.
- General reader robustness.

No new support-predictor experiment is added during final polishing. A credible independent replication would require a separately frozen support model and a preregistered evaluation protocol; adding one after reviewing V5 outcomes would create a new experiment family rather than repair wording.
