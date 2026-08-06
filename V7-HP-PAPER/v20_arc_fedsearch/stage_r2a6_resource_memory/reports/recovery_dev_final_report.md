# V20 R2-A.6 Recovery-Dev Final Report

## Decision

**Status: resource_memory_recovery_failed.** The preregistered shared Dev
gate did not pass, so the 300-query Recovery-Holdout was not started. This
stage did not start a reader and did not access the final test set.

## Frozen protocol and audit

- Fresh Recovery-Dev: 100 queries per dataset; fresh Recovery-Holdout: 300
  queries per dataset.
- The frozen split manifest records all query IDs and SHA256 fingerprints.
- The combined no-leak audit passed. All 2,560 selected units per dataset
  came from Router-Train client corpora; zero selected source documents were
  outside the allowed train corpus.
- The two independent Dev evaluations were byte-identical for both datasets.
- Fixed conditions: inherited partition and encoders, Bc=3, K in {5,8}, no
  router training, no query multiview, no reader, and no support/answer/gold
  client fields used during profile construction or routing.

## K=5 primary results

`gcr` denotes gold-client recall and `complete` denotes complete
support-client-set recall. Values are absolute recalls on the 100-query Dev
set. The control is B0 (one inherited client centroid).

| Method | 2Wiki gcr | 2Wiki complete | MuSiQue gcr | MuSiQue complete |
| --- | ---: | ---: | ---: | ---: |
| B0 single centroid | .896 | .810 | .837 | .670 |
| B1 inherited multiproto | .893 | .800 | .843 | .700 |
| R0 medoids, S0 max | .870 | .760 | .793 | .600 |
| R0 medoids, S1 top-3 mean | .868 | .780 | .818 | .650 |
| R0 medoids, S2 LSE | .879 | .790 | .821 | .640 |
| R1 farthest-first, S0 max | .805 | .680 | .708 | .520 |
| R1 farthest-first, S1 top-3 mean | .845 | .740 | .727 | .540 |
| R1 farthest-first, S2 LSE | .823 | .680 | .789 | .610 |
| R2 discriminative, S0 max | .822 | .700 | .758 | .540 |
| R2 discriminative, S1 top-3 mean | .852 | .740 | .768 | .560 |
| R2 discriminative, S2 LSE | .862 | .770 | .778 | .630 |
| R3 hybrid, S0 max | .880 | .780 | .760 | .550 |
| R3 hybrid, S1 top-3 mean | .878 | .790 | .796 | .650 |
| R3 hybrid, S2 LSE | **.905** | **.830** | .829 | **.680** |

R3+S2 is the best REMP configuration, but it changes complete@5 by only
`+0.020` on 2Wiki and `+0.010` on MuSiQue relative to B0. Its MuSiQue gcr is
`-0.008` below B0. The Dev gate required at least `+0.050` complete@5 on both
datasets, non-decreasing gcr@5, and no meaningful complete@8 loss; it fails
all of these requirements.

## Candidate-absence and storage interpretation

At K=8, B0 CandidateAbsenceLoss is `.030` on 2Wiki and `.130` on MuSiQue.
R3+S2 is `.040` and `.170`, respectively. Therefore its small K=5 routing
movement does not reduce candidate absence. REMP profiles cost about 99 KB per
client after counting both bounded unit text and the 32 selected float32
embeddings, compared with 3 KB for B0; the added finite memory does not justify
the storage increase under this frozen routing contract.

## Conclusion and next-method constraint

The result supports the R2-A.5 diagnosis: a small finite resource summary
cannot restore enough client-level multi-hop opportunity once the candidate
pool/compression bottleneck is present. This is a negative but clean result:
all four preregistered profile constructions and all three fixed pooling rules
were tested without reader leakage or a post-hoc selector. Per the frozen
protocol, do not begin Recovery-Holdout, do not train a selector, and do not
start a reader from R2-A.6. The next method decision is `stop_mars_route`.
