# Disk Cleanup Manifest

- Check time: 2026-06-17 19:35 JST
- Server: `/home/iiserver31/projects/FedE4RAG-main`
- `/home` usage: 87% (`3.1T` used, `470G` available)
- `experiments/v6_hp_hyper_next`: 286M
- `V6-HP1/outputs`: 57G
- `V6-HP1-OPTUNA/outputs`: 17M

## Decision

No cleanup, compression, or file movement was performed because `/home` is currently below the 95% risk threshold. Formal reports, CSV/JSONL/subset/config/log files were left untouched.

## Notes

- GPU resources were idle enough for small smoke tests.
- If `/home` rises above 95% again before multi-seed experiments, prioritize archiving reproducible intermediate retriever/checkpoint directories under `V6-HP1/outputs` while preserving reports, CSV, JSONL, configs, logs, and subset files.
