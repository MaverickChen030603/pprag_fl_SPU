# PC-1 Loss Alignment and Negative Contract Audit

Training manifest queries: 1000
Total explicit negatives: 4000
Negative source distribution: {'entity_overlap': 3998, 'lexical_overlap': 2}
Average baseline rank for negatives with pool rank: unavailable on train-only manifest
Rank 4-10 boundary negatives: 0 / 4000
False-negative risks where a negative is a gold support title: 5
Duplicate negative titles within queries: 6
Potential partial-hop/context negatives: 3935
Queries with multiple gold supports but only one explicit positive used: 992 / 1000

## Stage 0B-2 Answers

1. The PC-1 manifest is train-only and mostly entity-overlap negatives; boundary-rank coverage is only measurable when a frozen pool rank is available for the same query.
2. If boundary negatives are near zero, PC-1 optimizes global separation rather than rank-5 boundary crossing.
3. Multi-hop supervision is currently single-positive: the first support is used as the explicit positive even when two gold supports exist.
4. Partial-hop documents are not separately labeled in PC-1, so treating all non-support entity-overlap docs as strong negatives can misalign multi-hop retrieval.
5. This audit does not use calibration or final-test labels.
