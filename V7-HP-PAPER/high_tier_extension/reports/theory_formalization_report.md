# Theory Formalization Report

## 1. Policy-Action-to-Reader Gap

Let `q` denote a query, `C_b` the baseline reader context, `a` a context action produced after federated routing, and `C_a` the context after applying `a`. A reader `R(q, C)` produces an answer from a context. We evaluate answer quality with `M_ans`, support/evidence quality with `M_sup`, and combined multi-hop utility with `M_joint`.

We define:

- `Delta_sup(a) = M_sup(q, C_a) - M_sup(q, C_b)`
- `Delta_ans(a) = M_ans(q, C_a) - M_ans(q, C_b)`
- `Delta_joint(a) = M_joint(q, C_a) - M_joint(q, C_b)`

The central empirical gap is:

`Delta_sup(a) > 0` does not imply `Delta_ans(a) >= 0` or `Delta_joint(a) > 0`.

Federated routing can expose support-relevant evidence, but applying the resulting context action may still damage answer quality or fail to improve the joint metric.

## 2. Answer-Neutral Positive Action

An action is answer-neutral positive when:

- `Delta_ans(a) >= 0`
- `Delta_joint(a) > 0`
- `Delta_sup(a) >= 0`

This definition separates support discovery from reader-safe application. The selector is therefore not a retriever replacement; it is an action filter deciding whether routed context changes should be applied to the reader input.

## 3. No-Leak Constraint

At inference time, the selector can only use features `phi(q, C_b, a)` that do not depend on held-out reader outcomes, gold answer strings, or gold support labels. Train-fold outcomes may be used to construct labels, but held-out query outcomes are not used for selection.

## 4. Selector Objective

The inference-time decision can be written as:

`select a* = argmax_a s_theta(phi(q, C_b, a))`

subject to predicted answer-safety and action-effectiveness constraints. The frozen v2.3 selector operationalizes this as answer-neutral positive-action selection under query-level cross-fitting.

## 5. Paper Contribution Statement

The central problem is not whether federated routing can retrieve additional support evidence, but whether the resulting action should be applied to the reader context. We formalize this as an answer-neutral action selection problem under no-leak constraints.
