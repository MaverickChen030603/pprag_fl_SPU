# V17 Federated Opportunity Gap Definitions

1. **Routing absence:** at least one evidence-bearing client is not contacted under `B_c`.
2. **Local retrieval absence:** an evidence client is contacted, but its evidence does not enter local top-k.
3. **Pool absence:** the aggregated returned pool lacks a reader-effective repair.
4. **Single-action absence:** a positive context exists in the pool, but no single action is positive.
5. **Cross-client composition absence/search miss:** a positive cross-client context exists but the action generator does not expose it.
6. **Scoring miss:** an exposed positive action is not ranked first.
7. **Personalization miss:** the shared model fails where a local adapter succeeds.
8. **Gating miss:** the best positive action is rejected by the risk/communication gate.
9. **Harmful realization:** the applied action lowers Answer or Joint/composite outcome.

Phase A computes items 1-5 with offline gold/oracle information. Items 6-9 remain undefined until the corresponding learned modules exist.
