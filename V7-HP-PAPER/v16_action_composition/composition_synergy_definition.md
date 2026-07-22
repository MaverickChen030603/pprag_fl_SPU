# Composition Synergy Definition

For a frozen query, candidate pool, reader, and ordered five-document baseline context (C_0), a trajectory is a sequence of zero to three legal atomic edits. Every intermediate state contains exactly five distinct pool documents. (T=0) is exact baseline fallback.

For metric (M \in \{\text{Answer F1}, \text{SP F1}, \text{Joint F1}\}):

\[
\Delta M(\tau)=M(C_\tau)-M(C_0).
\]

Let \(\mathcal{A}_1\) contain **all reader-evaluated legal depth-1 edits** under the frozen search contract. The primary quantity is:

\[
\operatorname{StrictSyn}_M(\tau)=\Delta M(\tau)-\max_{a\in\mathcal{A}_1}\Delta M(a).
\]

The weaker within-trajectory quantity is reported only as a diagnostic:

\[
\operatorname{Syn}_M(\tau)=\Delta M(\tau)-\max_{a\in\tau}\Delta M(a).
\]

A composition is synergistic at margin \(\epsilon\) when `StrictSyn > epsilon`, with the preregistered grid `0`, `0.01`, and `0.02`. A query is composition-only positive when its best single-edit delta is non-positive and its best depth-2/3 delta is positive.

## Interpretation boundaries

- Strict synergy is conditional on the frozen candidate pool, action legality, ordering families, search budget, and reader.
- For Top-20 beam search, the estimate is a lower bound on the full combinatorial oracle because not every trajectory is evaluated.
- A larger composed candidate count alone does not establish synergy; the comparison is against the best evaluated single edit, with candidate-count and compute controls required later.
- The V15 pilot probe is exploratory and does not satisfy V16 confirmatory Go/No-Go because its actions were not generated from the complete V16 single-edit set.
