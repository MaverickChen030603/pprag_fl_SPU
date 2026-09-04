# High-Tier Submission Recommendation

## What Was Completed

1. Extension feasibility audit: complete.
2. Multi-reader replication: partial_existing_reader_only. Additional readers were not executed because frozen context text is absent.
3. Theory formalization: complete. The paper can now frame the core novelty as the policy-action-to-reader gap and answer-neutral action selection under no-leak constraints.
4. Strong baseline proxy comparison: complete_proxy_only. SetR-style, RankRAG-style, and Influence-style proxy baselines were computed from the existing action table.
5. MuSiQue: feasibility only; not executed.
6. HotpotQA scale-up: feasibility only; not recommended without reviewer request.

## Does This Move the Paper Toward a Higher-Tier Submission?

The strongest successful extension is the **theory formalization**, plus a **proxy strong-baseline comparison**. This improves framing and reviewer defensibility. It does not yet provide the strongest possible empirical boost because multi-reader replication could not be safely executed from the available frozen artifacts.

Recommendation: the paper is stronger than the previous writing pack for a HotpotQA-centered submission. It can be positioned as a main-conference stretch only if the authors are comfortable with proxy baselines and a single-reader main empirical setup; otherwise EMNLP/NAACL Findings or COLING remains the safer target.

## Should Claims Change?

Allowed strengthening:

- Emphasize the formal policy-action-to-reader gap.
- Add proxy comparisons against set-level, reader-aware, and utility-style action-selection heuristics.
- State that current multi-reader replication is prepared but blocked by missing frozen context snapshots.

Still forbidden:

- Do not claim answer_f1 significantly improves.
- Do not claim successful 2Wiki or MuSiQue generalization.
- Do not claim exact SetR/RankRAG/Influence reproduction.
- Do not present oracle diagnostics as inference-time methods.

## Next Best Single Action

If more time is available, materialize frozen baseline and selected contexts for final_1000 and run `google/flan-t5-base` plus `google/flan-t5-large` under identical prompts. That is the most likely low-cost empirical upgrade.
