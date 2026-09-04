# Superseded P0 Audit Attempt

This directory preserves the first generated P0 reports for auditability.

The first pass classified the bare V17 `final_test_inputs.jsonl` candidate-pool
listings as outcome contamination because their filenames contained
`final_test`. A targeted ID/path-only cross-check established that the 300
candidate IDs per dataset appeared only in those pool listings and in the
separate R5 execution that had already started before P0. The regenerated P0
reports classify the V17 pool listings as non-outcome split metadata and retain
the later R5 execution as 300/300 execution exposure.

No candidate text or label file was opened during the correction.
