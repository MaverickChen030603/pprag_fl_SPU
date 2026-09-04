# 2WikiMultiHopQA Data Preparation Report

## Status

2WikiMultiHopQA data has been downloaded and prepared on the server.

## Sources

- Official repository: https://github.com/Alab-NII/2wikimultihop
- `data.zip`: full dataset with `train.json`, `dev.json`, `test.json`
- `data_ids_april7.zip`: ID-enhanced dataset with `answer_id`, `evidences_id`, and `id_aliases.json`

## Prepared Files

```text
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/data.zip
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/data_ids_april7.zip
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted/data/dev.json
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted/data/test.json
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted_data_ids_april7/dev.json
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted_data_ids_april7/test.json
V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted_data_ids_april7/id_aliases.json
```

Adapter outputs:

```text
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/2wiki_converted.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/2wiki_dev_converted.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/2wiki_test_converted.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/2wiki_dev_ids_converted.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/2wiki_test_ids_converted.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/adapter_summary.json
```

## Field Audit

Dev split is evaluation-ready: it contains answer, context docs, supporting_facts, supporting_titles, evidences, answer_id, and evidences_id.

Test split contains answer, context docs, answer_id, and evidences_id. It does not contain sentence-level supporting_facts or raw evidences, so sentence-level support/evidence F1 should use dev unless an ID-level test metric is implemented.

## Recommendation

Proceed with `run_2wiki_smoke_300.py` on the dev split. Do not use the test split for sentence-level support/joint evaluation unless the metric is explicitly adapted to `evidences_id`.
