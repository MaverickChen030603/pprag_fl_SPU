# V20 R2-A Candidate Recall Smoke: Go/No-Go

## Contract

Both datasets use the frozen Router-Dev first 100 queries, all local documents
for resource cards, and no reader/final-test access. Candidate recall is an
offline support-client metric only. P0 is the inherited single-centroid Q0
reference. P1 separates prototype count from query views; P3 uses rank fusion,
not a concatenation that would hide the lexical branch below the cutoff.

## L=5 Complete Client-Set Recall

| Dataset | P0 Q0 | Best P1 Q0 | Delta | P3 P8 Q0 lexical RRF | Delta |
|---|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.720 | P8: 0.750 | +0.030 | 0.630 | -0.090 |
| MuSiQue | 0.620 | P16: 0.660 | +0.040 | 0.700 | +0.080 |

The alternate candidate-gold-client recall gives the same qualitative result:
P1 P8 rises by +2.83pp on 2Wiki and P1 P16 rises by +2.50pp on MuSiQue, below
the pre-registered +5pp threshold. Multi-view P1 is consistently harmful
(e.g., P8: 0.470 on 2Wiki and 0.480 on MuSiQue), so the first deterministic
entity/clause/relation view implementation must not be used by a router.

## Decision

**R2-A does not pass the candidate-generation gate.** No candidate family
improves P0 by at least five percentage points on both development datasets.
The lexical resource sketch is useful on MuSiQue but damages 2Wiki, while
multi-prototype Q0 yields only small, inconsistent improvements.

Consequently, R2-B set-aware Bc=3 selection, ReSLLM teacher calls, student
distillation, and reader evaluation are blocked. The current state is
`resource_representation_failure` for this first resource-card design, not a
failure of the U1 routing diagnosis: U1's Oracle evidence still identifies
routing coverage as the large recoverable opportunity, but R2-A has not yet
produced a deployable high-recall candidate generator.

## Narrow Follow-up

Before any router training, run a profile-quality audit only: prototype size
balance, variance, profile overlap, and whether the high-scoring incorrect
clients share entity/title vocabulary with positives. A revised candidate
representation must be preregistered and pass the same smoke gate on a fresh
Router-Dev slice. Do not tune set weights, train a selector, or start a reader
against this failing smoke split.
