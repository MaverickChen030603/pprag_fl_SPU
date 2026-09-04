# Claim Consistency Audit

No unsafe claim phrase was detected in the generated drafts and tables.

Checked phrases:

- `significantly improves answer_f1` -> `preserves answer_f1 with a small non-significant positive delta`
- `generalizes to 2Wiki` -> `2Wiki is reported as an external diagnostic and limitation`
- `solves cross-dataset generalization` -> `cross-dataset selector transfer remains limited`
- `reaches oracle` -> `oracle is used only as a diagnostic upper bound`
- `uses oracle selector` -> `formal inference does not use oracle selection`
- `fully solves reader sensitivity` -> `mitigates reader sensitivity under the tested HotpotQA setup`
