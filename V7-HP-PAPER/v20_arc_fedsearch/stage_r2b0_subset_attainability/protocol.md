# V20 R2-B0: Subset Attainability Audit

## Scope and decision

This stage tests whether the frozen REM-P candidate gain can be realized under
the existing federated retrieval contract. It is an audit, not a learned
selector experiment. The only permitted decision is `Proceed`,
`Switch-to-LR`, or `Stop` after the R2-B0 gate.

No reader, answer generation, final test, selector training, self-distillation,
retriever change, partition change, candidate-score reweighting, or dynamic
budget is permitted.

## Frozen contract

| Item | Setting |
| --- | --- |
| Datasets | 2WikiMultiHopQA and MuSiQue Router-Dev smoke100 |
| Candidate methods | `P0_single_centroid`, `REMP_rrf_p0_dense_lexical` |
| Candidate budget | `L=5` |
| Selection budget | exactly `Bc=3`, all `C(5,3)=10` subsets enumerated |
| Local retrieval | inherited `L0_dense`, depth 10, unchanged client-local indexes |
| Transmission | first 5 documents per selected client, exactly 15 documents |
| Primary merge | inherited raw dense-score merge, global top-10 |
| Diagnostic merge | inherited rank-percentile merge, global top-10 |
| Gold use | only after candidate lists, subset decisions, and retrieval outputs are frozen |
| Reproducibility | two complete runs; candidate, frozen retrieval, evaluation, and summary bytes must match |

The primary raw merge is retained because M0 established rank-percentile as a
conditional rather than cross-dataset main method. The percentile column is
diagnostic only and cannot decide the R2-B0 gate.

## Pre-registered outputs

For each candidate method and query, the audit saves an inference-only Top-5
candidate list, all ten subsets, each subset's local/transmitted/merged document
ids, and only then computes offline gold metrics. The evaluator reports:

- naive Top-3, P0 Top-3, client-coverage oracle, and retrieval oracle;
- selected-client complete coverage@3;
- local complete support@5/@10, transmitted complete support@15, raw merged
  complete support@5/@10, percentile merged complete support@5/@10, and
  supporting-document/title recall;
- rescue/harm for each oracle relative to naive Top-3;
- A--F failure taxonomy for the REM-P candidate list.

`client_oracle_subset` maximizes complete gold-client coverage, then gold-client
recall, then lexicographic subset order. `retrieval_oracle_subset` maximizes raw
merged complete-support@10, then raw merged support recall@10, then client
coverage, then lexicographic subset order. Both are offline-only diagnostics.

## A--F taxonomy

The taxonomy is evaluated on the REM-P Top-5 with the client-coverage oracle:

| Class | Definition |
| --- | --- |
| A | At least one gold client is absent from Top-5. |
| B | Gold clients are in Top-5, but naive Top-3 omits one or more. |
| C | The client oracle covers gold clients but local dense top-10 misses complete support. |
| D | Local top-10 succeeds but fixed top-5-per-client transmission loses support. |
| E | Transmission succeeds but raw merged top-10 loses support. |
| F | Complete support reaches raw merged top-10. |

Queries with more than three gold clients are separately marked
`outside_bc3_attainability`; they are reported but not forced into A--F.

## R2-B0 gate

The stage permits R2-B1 only when both datasets satisfy all conditions:

1. REM-P retrieval oracle improves raw merged complete-support@10 by at least
   `+0.05` absolute over REM-P naive Top-3.
2. B-class compression errors account for at least 10% of queries.
3. Retrieval-oracle rescues exceed harms.
4. The primary opportunity is not C/D local retrieval or transmission failure.

If the gap is primarily C/D, the next permitted stage is `Switch-to-LR`.
Otherwise a failed gate stops selector development. Reader and final test remain
sealed regardless of outcome.
