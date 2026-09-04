# R5 Power and Feasibility Report

## Authority Boundary

The authoritative 118-byte R5 recommendation specifies no sample size, split, dataset, reader role, alpha, or gate. Therefore this section is planning evidence only and cannot freeze an R5 contract.

## Conservative Paired Planning

No bootstrap was run. Existing R4 exploratory per-query Joint-F1 deltas were used only for a two-sided normal-approximation planning calculation.

- R4 six-cell equal-weight macro effect: `0.038666748670`
- Smallest R4 cell mean effect: `0.017745310245`
- Planning effect: 50% shrinkage of the smallest cell = `0.008872655123`
- Conservative paired SD: largest six-cell paired-delta SD = `0.203019804987`
- Two-sided alpha: `0.05`
- Target power: `0.80`
- Approximate minimum N: **4110 per independently tested dataset-reader primary cell**

The later, non-authoritative R5 preregistration selected N=300/dataset. N=300 is below this conservative planning requirement. Alpha was not relaxed and the test was not changed to one-sided.

## Pool Feasibility

Before the unapproved R5 launch, the recovered current-tree ledger found zero overlap for the 300 selected IDs per dataset. At P0 start, however, the full selected split had already been supplied to an active R5 retrieval pipeline. Thus eligible N for a future clean confirmation using these same selected IDs is 0. The recommendation also does not name a larger candidate pool from which `4110` clean IDs could be selected.
