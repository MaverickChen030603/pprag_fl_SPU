# Extension Feasibility Report

This audit preserves the frozen HotpotQA v2.3 result and checks only low-cost high-tier extensions.

## Decision Summary

- Multi-reader replication: **not_executed**. final_1000 artifacts expose metrics and titles but not full reader contexts
- Theory formalization: **execute**. This is low-cost and high paper value.
- Strong baseline comparison: **execute proxy**. SetR-style, RankRAG-style, and Influence-style proxies can be computed from existing action labels.
- MuSiQue: **feasibility only**. Do not run reader or full validation by default.
- HotpotQA scale-up: **not recommended now** because joint/support metrics are already significant and extra reader cost is high.

## Claim Boundary

The frozen v2.3 result remains the main result: joint/support metrics improve significantly, while answer_f1 is a small non-significant positive delta. 2Wiki remains diagnostic/limitation evidence, not a successful generalization claim.
