# R4-P0 Reader Context Contract Audit

**Decision:** `pass`; no unresolved reader-contract ambiguity.

## Evidence hierarchy

1. V20 preregistration fixes a retrieval global pool/context contract of `Top-10 / Top-5`.
2. V17 preregistration independently fixes final context `K=5` and the two reader checkpoints.
3. V17's executed `01_label_oracle_contexts.py` uniquely specifies the input serialization, truncation and decoding used below.

## Frozen contract carried into R4

| Component | Contract |
|---|---|
| Retrieval global pool | raw merged Top-10 |
| Reader context | first 5 documents in that frozen order |
| Duplicate policy | no duplicate document IDs; no padding |
| FLAN format | `[rank] title: text` with answer-only instruction |
| UnifiedQA format | `question` followed by `title: text` spans |
| Character truncation | first 4,000 context characters after ordered serialization |
| Tokenizer encoding | truncation to 1,024 input tokens; default truncation direction (right) |
| Decode | greedy (`num_beams=1`, `do_sample=False`), `max_new_tokens=32` |
| Support extraction | frozen V16 support predictor, threshold plus deterministic top-two fallback |

The V20 text does not restate every prompt detail, but its `Top-10 / Top-5` table and the V17 executed script agree. Therefore this is recorded as **legacy frozen reader K=5 carried forward**, not a post-reader choice.

## Cached checkpoints

| Reader | Revision |
|---|---|
| `google/flan-t5-large` | `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` |
| `allenai/unifiedqa-v2-t5-large-1363200` | `1d3b8e13b29dbd161494b0b15428378f4713c418` |
