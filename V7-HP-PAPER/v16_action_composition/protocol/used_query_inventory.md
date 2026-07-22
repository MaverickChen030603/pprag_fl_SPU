# V16 Used-Query Inventory

This is a conservative exclusion inventory over V1--V15 artifacts. IDs or normalized questions found in architecture, feature, threshold, utility, ablation, transfer, or reporting artifacts are treated as previously exposed.

| Dataset | IDs | Normalized questions | Fingerprint |
|---|---:|---:|---|
| 2wikimultihopqa | 34,790 | 34,789 | `b17c587e3ef362dbaa0af6b23d32fcba6ab8c6fe731949a48bf60ead4f299c00` |
| hotpotqa | 16,405 | 16,405 | `082a5a2581be17cc964e64b4d893303f64deeb1fa1271f30bf00b8b330b1b382` |

## Boundary

- V15 artifacts are included; V16 outputs are excluded to prevent self-contamination of the inventory.
- The repeatedly inspected 7,405 HotpotQA validation queries remain ineligible for V16 confirmatory evaluation.
- Absence from this inventory is not proof of non-exposure; source provenance and question-fingerprint exclusion are both enforced during split freezing.
- Parsed sources: 191; parse errors: 3; oversized files skipped: 0.
