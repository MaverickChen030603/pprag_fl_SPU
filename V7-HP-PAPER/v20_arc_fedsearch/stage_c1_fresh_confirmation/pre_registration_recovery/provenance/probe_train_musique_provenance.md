# M2 Probe-Train Provenance: musique

**Recovery status:** `recovered_confidence_A`

- Direct historical query-ID artifact: `/srv/lab/projects/jiankangchen/pprag_fl_SPU/V7-HP-PAPER/v17_fedaction_rag/partitions/client_query_distribution.csv`.
- Artifact commit: `ff3532148a6b0af57bbe6f990f70159608476188`.
- Source split: `V17 train`; recovered query count: `5000`.
- Query-ID SHA256: `61aefbf7a9b380779c6734fbfcce261b88668366b9ed147a571b44319e68a27d`.
- Historical dataset source SHA256: `24be55519b341d8f5ce0c298fcd2f076bfba07b619e83955e2fe39e7264ae504`.
- Historical training rows: `{'train_queries': 5000, 'candidate_rows': 40000, 'positive_rows': 8808, 'hard_negative_rows': 31192}`.
- Candidate packets were not recovered; their absence leaves B2p unavailable but does not invalidate the direct M2 train-ID artifact.

The manifest is recovered from a version-controlled historical training artifact, not inferred from a current sampling rule.
