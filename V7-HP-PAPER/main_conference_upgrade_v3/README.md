# V7-HP-PAPER Main-Conference Upgrade v3

This directory is an independent, result-gated extension of the frozen `submission_revision_v2/`. It asks whether bounded reader-compatible candidate generation raises answer-safe positive-action opportunity enough to justify a new fully nested selector.

## Frozen decision thresholds

- `<25%` positive-query coverage: stop downstream upgrade.
- `>=30%`: meaningful candidate-opportunity result.
- `>=40%`: strong candidate-opportunity result.

The observed v3 result is **23.4%**, so Stages 4-7 emit explicit gate-skip artifacts. No official, multi-reader, or scale-up gain is claimed.

## Reproduction

```bash
cd /home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/main_conference_upgrade_v3
PYTHON_BIN=/home/iiserver31/anaconda3/envs/supv2/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash run_full_v3_pipeline.sh
```

Stage 3 is resumable and uses the fixed local FLAN-T5-large snapshot. The no-leak candidate audit is written before reader execution.

## Current primary artifacts

- `reports/main_conference_gap_audit.md`
- `reports/candidate_opportunity_report.md`
- `outputs/action_outcomes/v3_action_outcome_summary.json`
- `reports/main_conference_readiness_report.md`
- `paper_findings_fallback_v3.md`
- `final_main_conference_claim_audit.md`
- `final_submission_decision.md`
