# V16 Synergy-Aware Action Composition

V16 studies state-dependent sequences of atomic edits over a frozen five-document context. Phase A is an oracle landscape, not a learned-policy claim. The composer phase starts only after the preregistered synergy checkpoint passes.

## Current execution order

1. `protocol/01_data_usage_audit.py`
2. prepare official MuSiQue source and freeze three-dataset splits
3. `protocol/03_no_leak_audit.py`
4. build frozen Top-10/Top-20 pools
5. `oracle_search/01_generate_oracle_trajectories.py`
6. offline two-reader labeling
7. `oracle_search/02_oracle_action_landscape.py`
8. Go/No-Go 1 before composer training

The V15 pilot synergy probe is exploratory only and cannot satisfy V16 confirmation.
