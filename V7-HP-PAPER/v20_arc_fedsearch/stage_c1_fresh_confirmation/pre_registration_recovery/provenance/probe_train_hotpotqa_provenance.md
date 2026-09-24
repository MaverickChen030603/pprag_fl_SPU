# M2 Probe-Train Provenance: hotpotqa

**Recovery status:** `recovered_confidence_A`

- Direct historical query-ID artifact: `/srv/lab/projects/jiankangchen/pprag_fl_SPU/V7-HP-PAPER/v17_fedaction_rag/partitions/client_query_distribution.csv`.
- Artifact commit: `ff3532148a6b0af57bbe6f990f70159608476188`.
- Source split: `V17 train`; recovered query count: `5000`.
- Query-ID SHA256: `d490414823e45aadf7bdc7ff2fb3b0b64a4616114fca780042860f71557c2ac0`.
- Historical dataset source SHA256: `205173be35443520aa89478ad7b1084a9be4f8ae4f542672c18cc25b5add1cdc`.
- Historical training rows: `{'train_queries': 5000, 'candidate_rows': 40000, 'positive_rows': 7149, 'hard_negative_rows': 32851}`.
- Candidate packets were not recovered; their absence leaves B2p unavailable but does not invalidate the direct M2 train-ID artifact.

The manifest is recovered from a version-controlled historical training artifact, not inferred from a current sampling rule.
