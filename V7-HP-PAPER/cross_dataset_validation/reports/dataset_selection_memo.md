# Dataset Selection Memo

## Executive Decision

2WikiMultiHopQA is not available locally with usable evidence labels. Do not run smoke/final evaluation yet; prepare or download the dataset first.

MuSiQue should remain second priority and should only run after 2Wiki feasibility is resolved.

IIRC and MultiHop-RAG remain optional appendix/limitation candidates.

## Feasibility Table

| dataset | available | path | examples | answer | evidence | context | recommendation |
|---|---:|---|---:|---:|---:|---:|---|
| 2WikiMultiHopQA | False | not_found_locally | 0 | False | False | False | proceed_first_if_available |
| MuSiQue | False | not_found_locally | 0 | False | False | False | after_2wiki |
| IIRC | False | not_found_locally | 0 | False | False | False | optional |
| MultiHop-RAG | False | not_found_locally | 0 | False | False | False | optional |

## Paper Impact

The HotpotQA v2.3 main result remains frozen. Cross-dataset validation is blocked until P1 data is available, so current paper claims should remain HotpotQA-centered with cross-dataset validation listed as pending robustness work.
