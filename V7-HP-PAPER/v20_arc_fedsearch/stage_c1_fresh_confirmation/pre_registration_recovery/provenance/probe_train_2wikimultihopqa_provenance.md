# M2 Probe-Train Provenance: 2wikimultihopqa

**Recovery status:** `recovered_confidence_A`

- Direct historical query-ID artifact: `/srv/lab/projects/jiankangchen/pprag_fl_SPU/V7-HP-PAPER/v17_fedaction_rag/partitions/client_query_distribution.csv`.
- Artifact commit: `ff3532148a6b0af57bbe6f990f70159608476188`.
- Source split: `V17 train`; recovered query count: `5000`.
- Query-ID SHA256: `7696c68c9e9d81e325eab2072af80f9a81a87e98910ad4573cb6bdd9fe954a8c`.
- Historical dataset source SHA256: `b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa`.
- Historical training rows: `{'train_queries': 5000, 'candidate_rows': 40000, 'positive_rows': 8804, 'hard_negative_rows': 31196}`.
- Candidate packets were not recovered; their absence leaves B2p unavailable but does not invalidate the direct M2 train-ID artifact.

The manifest is recovered from a version-controlled historical training artifact, not inferred from a current sampling rule.
