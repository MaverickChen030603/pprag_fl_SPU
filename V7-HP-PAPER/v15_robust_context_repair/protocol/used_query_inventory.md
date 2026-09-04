# V15 Used-Query Inventory

This inventory is a conservative exclusion list. A query found in any
architecture, calibration, evaluation, ablation, diagnostic, or transfer
artifact is treated as previously used. Absence from this list is not by
itself proof of non-exposure.

## Dataset counts

| Dataset | Unique query IDs | Unique normalized questions | ID fingerprint |
|---|---:|---:|---|
| 2wikimultihopqa | 2,187 | 2,187 | `d49fecb984ed1e4ff64cbdc9ba2f475f26738e486d96b18c5c8fda4e2939d9c0` |
| hotpotqa | 8,405 | 12,405 | `fd76baba90d55f61f534e90069453730b924bf52b43f4e0d9389bdc91d76fb20` |
| unknown | 0 | 22,070 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Usage counts

| Usage | Unique query IDs |
|---|---:|
| ablation | 1,000 |
| architecture_selection | 8,683 |
| result_inspection | 10,592 |
| subgroup_analysis | 7,705 |
| threshold_tuning | 2,900 |
| transfer_calibration | 6,187 |

## Audit coverage

- Files containing query IDs: 1,563
- Files skipped by size policy: 77
- Parse errors: 5
- Raw corpora, environments, archives, and V15 itself are intentionally excluded.
- The complete per-source inventory and exclusion IDs are in `used_query_inventory.json`.

## Confirmatory boundary

The HotpotQA validation set previously used throughout HP1--HP4 and V4--V14
must not be described as a new confirmatory test. V15 confirmation uses
train-derived IDs absent from this inventory and reports that provenance
explicitly.

## Parse errors requiring review

- `V7-HP-PAPER/high_tier_extension/outputs/multi_reader/per_example_multi_reader_delta.jsonl`: JSONDecodeError: Expecting property name enclosed in double quotes: line 2 column 1 (char 2)
- `experiments/v6_hp_hyper_next/results/._method_identity_audit_after_ablation_raw.jsonl`: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 45: invalid start byte
- `experiments/v6_hp_hyper_next/results/._method_identity_audit_after_ablation_summary.csv`: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 45: invalid start byte
- `experiments/v6_hp_hyper_next/results/._score_logging_raw.jsonl`: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 45: invalid start byte
- `experiments/v6_hp_hyper_next/results/._score_logging_summary.csv`: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 45: invalid start byte
