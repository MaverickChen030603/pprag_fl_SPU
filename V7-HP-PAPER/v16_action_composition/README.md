# V16 Synergy-Aware Action Composition

V16 studies state-dependent sequences of atomic edits over a frozen five-document context. Phase A is an oracle landscape, not a learned-policy claim. The composer phase starts only after the preregistered synergy checkpoint passes.

## Final Phase-A Decision

The exact 100-query Oracle run completed 314,476 reader-context evaluations across HotpotQA, 2WikiMultiHopQA, MuSiQue, FLAN-T5-Large, and UnifiedQA-T5-Large. StrictSynJoint is positive with a paired-bootstrap confidence interval above zero in all six cells. However, the preregistered composition-only threshold is met on both readers only for MuSiQue, not for the required two datasets.

Machine decision: `hold_or_redirect`. Scientific status: `analysis_paper_only`. No composer was trained, Top-20 was not used to rescue the checkpoint, and final-test labels remain sealed.

## Current execution order

1. `protocol/01_data_usage_audit.py`
2. prepare official MuSiQue source and freeze three-dataset splits
3. `protocol/03_no_leak_audit.py`
4. build frozen Top-10/Top-20 pools
5. `oracle_search/01_generate_oracle_trajectories.py`
6. offline two-reader labeling
7. `oracle_search/02_oracle_action_landscape.py`
8. Go/No-Go 1 before composer training

The V15 pilot and V16 20-query exact batches are exploratory only. The final decision uses the preregistered 100-query minimum in every dataset-reader cell; final-test data remain sealed throughout Phase A.

Current execution entry points:

- `run_v16_phase_a_smoke.sh`: retrieval and search-contract smoke.
- `run_v16_phase_a_exact_readers.sh`: corrected 20-query exact search smoke.
- `run_v16_phase_a_exact100.sh`: preregistered 100-query Oracle Checkpoint batch with resumable two-reader labeling.
