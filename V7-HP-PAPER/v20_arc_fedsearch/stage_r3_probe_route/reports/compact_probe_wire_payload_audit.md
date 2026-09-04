# R3 Compact Probe Wire-Payload Audit

Status: `compact_probe_communication_contract_confirmed`.

This is an offline serialization audit over the completed, two-run verified Probe-Dev artifacts. No retriever was called, no query/client selection was changed, and no reader or final-test label was accessed.

## Fixed Wire Contract

- Schema header: 16 bytes/query.
- Existing scalar feature vector: 18 IEEE-754 float32 values/client = 72 bytes/client.
- `L=8` formal probe response: 592 bytes/query (`16 + 8 x 72`).
- No title text, entity string, document text, passage ID, full embedding, gold label, answer, or reader value is present on the formal wire.
- Static P0 candidate IDs and profile scores are already known at the server and are not returned by clients.

## Frozen Quality and Cost

| Dataset | Frozen rule | Wire bytes | Prior verbose debug bytes | 15-doc bytes | Wire/doc ratio | Selection exact | Local@10 | Transmitted@15 | Raw merged@10 | Percentile merged@10 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | P5_static_plus_probe_alpha_0.25 | 592 | 7656 | 7474 | 7.92% | True | 0.210 | 0.210 | 0.210 | 0.210 |
| musique | P1_probe_dense_top1 | 592 | 7695 | 8818 | 6.71% | True | 0.350 | 0.310 | 0.290 | 0.270 |

The wire payload is under 10% of the matched 15-document payload on both datasets, while all frozen P0--P5 choices are reproduced exactly after float32 packing. The prior 7.6KB figure was an on-disk verbose JSON debug transcript, not the communication format. The audit therefore clears the communication precondition. Per the current instruction, no supervised ranker or reader has been launched; reader evaluation remains blocked until the separate fresh-holdout gate.
