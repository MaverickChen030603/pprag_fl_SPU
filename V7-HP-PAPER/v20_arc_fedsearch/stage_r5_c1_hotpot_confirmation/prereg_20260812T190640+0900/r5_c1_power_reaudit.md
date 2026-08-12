# R5-C1 Power Re-audit

## Implementation Check

- Unit: query-level paired Logistic-minus-baseline Joint-F1 deltas.
- Two-sided alpha: `0.05`; target power: `0.80`.
- Conservative SD is the largest query-level paired-delta SD across the six R4 cells, not a macro-cell variance.
- Planning rule is 50% shrinkage of the smallest positive R4 cell mean.
- P0 N=4110 reproduced exactly: `true`.

## Effects and Analytic N

- Six-cell macro effect: `0.038666748670`; analytic N `217`.
- Smallest cell `musique/unifiedqa` effect: `0.017745310245`; analytic N `1028`.
- Shrunken planning effect: `0.008872655123`; analytic N `4110`.

## Empirical-residual Power Simulation

Using 5,000 fixed-seed Monte Carlo bootstrap resamples from centered query-level residuals of the max-SD cell:

- N=4110 estimated power: `0.8044`.
- N=4200 estimated power: `0.8174`.

No implementation error was found. The mentor-frozen N=4200 is retained unchanged.
