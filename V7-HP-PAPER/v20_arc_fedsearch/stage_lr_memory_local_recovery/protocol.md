# V20 LR: Evidence-Memory-Conditioned Local Retrieval Recovery

## Scope

This stage follows R2-B0 diagnostic_shift / Switch-to-LR. It keeps the
physical local indexes, frozen REM-P candidate scorer, Bc=3, five transmitted
documents per client, 15-document total budget, raw global top-10 merge, and
Reader/final-test seal unchanged.

LR-0 first audits original-query dense availability at local depths
{5,10,20,50}. LR-1 runs only if the D1 C-eligible population is not dominated
by C0 (support absent from a selected client's physical index).

## Frozen tracks

| Track | Selection | Role |
| --- | --- | --- |
| D0_P0_naive_top3 | P0 frozen Top-5, first 3 | stable deployable baseline |
| D1_REMP_naive_top3 | REM-P frozen Top-5, first 3 | sole deployable gate track |
| O1_REMP_client_oracle_subset | R2-B0 frozen gold-only client oracle | diagnostic only |

O1 is read from the already-frozen R2-B0 oracle artifact and is always labeled
gold-only. It cannot select a method, parameter, anchor, or gate outcome.

## LR-0

Use original-query L0 dense ranks at depths 5/10/20/50, with final transmission
fixed to the first 5 documents/client. C0--C3 is assigned only when all gold
clients are in the track's selected clients:

- C0: support is absent from a selected physical SQLite index;
- C1: present but not retrieved by dense top-50;
- C2: in top-50 but not top-10;
- C3: in top-10 but not final top-5 transmission.

Selection misses remain a routing/candidate category, not local-retrieval C0.
If C0 is the largest C category on D1, LR-1 must not run.

## LR-1 fixed matrix

| Method | Local ranking and transmission |
| --- | --- |
| L0 | q0 original-query dense, top-5 |
| L1 | q0 dense depth-50, same dense top-5 |
| L2 | q0 dense-50 + BM25-50, RRF k=60, top-5 |
| L3 | q0 dense-25 + q1 anchor dense-25, RRF k=60, top-5 |
| L4 | q0/q1 dense-25 and BM25-25, four-way RRF k=60, top-5 |

For L3/L4, q1 is q0 + [ANCHOR] + sanitized frozen REM-P top-1 unit text.
The anchor is the highest cosine unit for the query within that selected
client's existing bounded memory. It is not an LLM rationale and uses no gold,
answer, support, or reader input.

## Gate

Only D1 decides deployment. A method must keep raw merged complete-support@10
non-decreasing on both datasets, improve one by >= +0.05 and the other by
>= +0.02, have total rescue/harm ratio >2 with no dataset harm above 2, avoid a
merge-only cancellation of transmission gain, and not lower D0 by more than 1pp.
Candidates, anchors, per-query artifacts, and summaries must match between two
full runs.

No method search, tuning, selector, query generator, LLM, reader, or final-test
access is allowed.
