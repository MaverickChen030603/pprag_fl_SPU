# Data Provenance Report

## Main sample

- Dataset: Hugging Face `hotpot_qa`
- Configuration: `distractor`
- Split: `validation`
- Source rows: 7,405
- Valid normalized candidates: 7,405
- Sampling: normalize all valid rows, apply `random.Random(44).shuffle`, take the first 1,000
- Stored examples: 1,000

The construction was independently replayed with the source dataset. Both the ordered sequence of 1,000 query IDs and the unordered ID set match the stored `V7-HP4/data/hotpot_validation_1000.json` exactly.

## Checksums

| Artifact | SHA-256 |
|---|---|
| `V7-HP4/data/hotpot_validation_1000.json` | `fba436e504e84d0180058fd0035884f53d4930f3c7a7ba9b6ce2dcf06acf7d59` |
| `V7-HP4/data/hotpot_validation_1000.meta.json` | `4dbf71130b5afa4ee66952d56241c4df62d1b3d804f7ef40be72d8232ea49190` |
| v2.3 `action_labels.jsonl` | `28306bd681176e7597cb9bd9c577649cc54d56820d5071292ed884bf59c47a67` |
| legacy v2.3 `per_example_delta.jsonl` | `c35e1e7c45749e63937f42647cb186e4359cd0c64905e2b656e85970595ccd16` |

The machine-readable `data_manifest.json` contains the full ordered query-ID list, duplicate audit, schema, source reconstruction result, and checksums.

## Conversion and gold labels

The stored converted schema is `_id`, `question`, `answer`, `supporting_titles`, and concatenated `reference`. Sentence-level supporting facts were dropped from this converted file. Submission v2 recovers the gold `(title, sent_id)` pairs from the source validation split and exports them in `gold_supporting_sentence_labels.jsonl`. These labels are for audit and future official evaluation; they are not injected into target-query selector features.

## Sanitization and clients

Candidate documents are sanitized of latent support/bridge annotations before candidate generation. They are assigned synthetic clients by `client_{document_index mod 5}`. This is a deterministic round-robin simulation, not a natural non-IID federation. Full candidate text is centrally available to the organizer.

## Folds

Unique query IDs are sorted by integer MD5 hash. Fold `i` receives `ordered_ids[i::5]` as its 200 test queries; the other 800 are training queries. `fold_manifest.json` stores every train/test ID and verifies zero overlap.

## Duplicate handling

No post-sampling duplicate-removal pass was applied. Duplicate IDs and repeated question strings are explicitly audited in `data_manifest.json`; the split operates at query ID level.
