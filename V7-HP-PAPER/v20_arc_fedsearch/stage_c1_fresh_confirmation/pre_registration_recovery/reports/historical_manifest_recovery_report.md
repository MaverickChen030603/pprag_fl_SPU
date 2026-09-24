# V20-C1 Historical Manifest Recovery Report

**Decision:** `ready_for_one_shot_fresh_confirmation`

## Recovered M2 Probe-Train

All three M2 Probe-Train query sets were recovered from the version-controlled V17 `topic_silo` client-query distribution artifact. R2 freezes `router_train` as V17 `train[0:5000]`; R3 copies that router train set for 2Wiki/MuSiQue, while the Hotpot transfer uses the V17 train split directly.

| Dataset | Queries | Candidate rows | Positive rows | Negative rows | Confidence |
| --- | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 5000 | 40000 | 7149 | 32851 | A |
| 2wikimultihopqa | 5000 | 40000 | 8804 | 31196 | A |
| musique | 5000 | 40000 | 8808 | 31192 | A |

The original Probe-Train packet files were not recovered. This is sufficient to keep B2p unavailable, but does not weaken recovery of the M2 training query IDs because the IDs and their 5,000-query cardinality are direct historical artifacts and the row statistics agree with the frozen M2 result files.

## Historical Exclusion

The exclusion union combines: M2 Probe-Train IDs, the V16 direct used-query inventory, conservative V17 development/calibration supersets covering R2--R4 selection activity, and all 300 revealed R5 IDs per dataset. Conservative supersets avoid an unsupported claim about exact pruned per-stage membership.

## Freshness Audit

Each dataset has 500 C1 IDs. The ID-level Probe-Train, method-selection, R5, and full-union intersections are zero. A normalized-question SHA256 secondary audit is also zero. Candidate source rows are not ranked, retrieved, read by a Reader, or scored here.

## Remaining Boundary

This recovery authorizes only the one-shot C1 execution under the separately frozen protocol. It does not authorize B2p, B5, dynamic Top-k, new features, new baselines, or any result-triggered method change.
