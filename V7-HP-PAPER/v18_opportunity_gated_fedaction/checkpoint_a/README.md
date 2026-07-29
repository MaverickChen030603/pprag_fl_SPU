# Post-Completion Checkpoint-A Audit

`01_checkpoint_a_integrity_audit.py` is intentionally dormant while V17 runs.
It verifies the thirty reader-backed cells, N=100 query alignment, partition
and budget contract, local-k, K=10 candidate pools, document de-duplication,
client IDs, frozen commit, artifact hashes, and the no-leak audit. It refuses a
partial phase that lacks the V17 formal decision marker.

After completion, invoke it from the V17 project environment:

```bash
python V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/01_checkpoint_a_integrity_audit.py \
  --v17-root V7-HP-PAPER/v17_fedaction_rag \
  --phase-tag phase_a_checkpoint100 \
  --output-dir V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a
```

Only a `pass` result permits the V18 Go/No-Go report. The audit deliberately
does not make that decision itself.
