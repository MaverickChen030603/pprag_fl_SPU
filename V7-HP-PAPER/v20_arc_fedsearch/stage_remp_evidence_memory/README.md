# Stage REM-P Evidence Memory

This is an isolated exploration branch for the LEANN / self-distillation / PPML
combination idea, reduced to the smallest safe retrieval-only gate.

Run one dataset:

```bash
FEDSEARCH_ROOT=/home/iiserver31/projects/FedE4RAG-main \
  bash v20_arc_fedsearch/stage_remp_evidence_memory/run_remp_dataset.sh 2wikimultihopqa 0
```

Outputs are written under:

```text
v20_arc_fedsearch/stage_remp_evidence_memory/<dataset>/
```

This stage does not train a model, start a reader, touch final test, or use gold
fields for inference. Gold support-client sets are used only after candidate
lists exist, for offline complete-set recall metrics.

