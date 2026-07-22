# V16 Preregistration

**Experiment:** V7-HP-PAPER-v16-synergy-aware-action-composition  
**Internal codename:** CompoRepair, not a frozen publication name  
**Freeze date:** 2026-07-22  
**Primary question:** Can a learned sequence of state-dependent, fixed-budget context edits expose and select reader-effective contexts that no single legal edit can reach?

## Confirmatory boundary

V1--V15 artifacts, including the repeatedly inspected 7,405 HotpotQA validation examples and all V15 fresh pilot examples, are development history. They cannot be V16 confirmatory tests. V16 uses query-ID and normalized-question exclusion before deterministic stratified splitting. Final labels are physically separated from final inputs and remain sealed until the architecture, objectives, readers, baselines, gates, and analyses below are frozen.

## Data and frozen splits

Datasets: HotpotQA, 2WikiMultiHopQA, and MuSiQue. Each uses 5,000 train, 1,000 model-development, 1,000 calibration, and 2,000 final-test examples drawn from eligible labeled source data. If a source has insufficient unused rows, split sizes must be reduced before any reader evaluation and recorded as a protocol amendment; examples cannot be borrowed from prior holdouts.

Candidate pools are real retriever Top-10 and Top-20 outputs. The five-document context budget is constant across every method. No random padding is allowed.

## Methods and development order

1. Frozen Top-5, CrossEncoder-Top5, old V14 Full, best learned single edit, RECOMP, and contract-compatible official baselines.
2. Hand-crafted greedy composer.
3. Direct-delta greedy composer.
4. Imitation composer trained from oracle trajectories.
5. Synergy-aware composer with opportunity, trajectory value, harm, ranking, and strict-synergy objectives.
6. Offline RL only if simpler methods pass learnability criteria.

All methods share candidate pools, document budget, reader, embeddings, and one final reader call. Candidate-count, parameter, compute, and oracle search-budget controls are mandatory.

## Readers and metrics

Readers are fixed before evaluation: `google/flan-t5-large` and `allenai/unifiedqa-v2-t5-large-1363200`. Primary metric is dataset-official Joint F1 where defined. Secondary metrics are Answer EM/F1, support/evidence F1, intervention coverage, Answer-drop, Joint-drop, and mean/P95 latency. Dataset-native evidence metrics remain labeled as such; a constructed Answer x Evidence measure cannot be called official Joint.

## Primary hypotheses

- H1: learned composition exceeds the strongest single-edit baseline on paired Joint F1.
- H2: composition reduces the no-positive-repair rate.
- H3: synergy-aware training exceeds ordinary sequential imitation.
- H4: composition-only positive cases occur across multiple datasets and readers.
- H5: gains survive matched candidate-count, parameter, and compute controls.

The primary test is learned composition versus the strongest single-edit baseline on Joint F1. It uses query-level paired bootstrap with 5,000 resamples and a 95% CI. Secondary tests use Benjamini-Hochberg correction.

## Oracle Checkpoint 1

Composition continues as the core method only after every dataset-reader cell contains at least 100 development queries and at least two datasets show: composition-only positive rate >= 10%, positive mean `StrictSynJoint`, and a paired bootstrap 95% CI whose lower bound exceeds zero. Smaller batches are search-integrity smoke tests only. Top-20 beam estimates are explicitly lower bounds on the exhaustive oracle.

## Learnability Checkpoint 2

At least three of the following must hold: composer search recall@K exceeds greedy; Joint exceeds best single edit; selected harm does not exceed old Full; both readers agree in direction; composition recoverability is meaningful. RL is not added solely for method appearance.

## Final decision vocabulary

Exactly one status will be selected: `composition_method_breakthrough`, `strong_ecir_method_paper`, `promising_but_not_learned`, `analysis_paper_only`, `composition_gap_not_supported`, or `abandon_and_redirect`.

## Prohibited adaptation

Final-test outcomes cannot select features, objectives, thresholds, beam widths, trajectory depth, readers, pool size, or reported subgroups. Any post-freeze change creates a new version and a new untouched final split.
