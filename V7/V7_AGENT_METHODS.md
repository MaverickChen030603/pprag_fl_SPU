# V7 Agent Method Overlay

This V7 directory was bootstrapped from the working V6 pipeline.

Required V7 methods:

- `agent_rule_v7`: rule-based state/memory/downstream-aware block selector.
- `agent_bandit_v7`: contextual bandit block selector.
- `agent_policy_v7`: learned policy selector.
- `agent_llm_planner_v7`: optional high-level planner that only emits strategy mode and weights.

The first executable pass may map `agent_rule_v7` and `agent_bandit_v7` onto the existing V6
selector interfaces while adding V7 logging fields. Do not let any V7 method exceed the same
payload budget used by `v7_budget_aligned`.
