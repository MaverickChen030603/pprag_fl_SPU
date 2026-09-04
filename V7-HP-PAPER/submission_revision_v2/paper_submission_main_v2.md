NOT READY FOR MAIN-CONFERENCE SUBMISSION

# Reader-Safe Context Action Selection for Multi-Hop Question Answering

## Main-track status note

This file records the strongest defensible stretch framing, but it must not be submitted as a main-track manuscript in its current evidence state. The fully nested rerun is complete and title-level evidence gains are significant. However, answer F1 and the answer-title-support product are not significant; official sentence-level HotpotQA support/joint predictions are unavailable; only one reader completed; and the evaluated client identities are synthetic with centralized candidate text.

## Stretch abstract

We study a policy-action-to-reader gap in multi-hop retrieval-augmented QA: upstream routing changes can be flattened by top-k retrieval, while support-oriented context edits can damage answer-bearing anchors. We formulate downstream context construction as selective bounded action choice. A two-stage organizer either preserves a frozen top-5 context or applies one extractive reorder/insertion after an answer-safety gate. In fully nested five-fold query-level evaluation on 1,000 reproducibly sampled HotpotQA validation questions, title-level support recall improves by 0.0120 (p=0.007) and title-level support F1 by 0.0150 (p=0.018). Answer F1 improves by 0.0028 (p=0.344) and a custom answer-title product by 0.0079 (p=0.1245), neither significantly. The main-eligible action set contains no positive action for 797 queries. The present evidence therefore supports a mechanism and diagnostic paper, not a main-track claim of improved end-to-end QA.

## Required main-track thesis

A credible main-track version would need to establish that reader-safe action selection improves end-to-end multi-hop answer quality under a deployment-valid protocol, not only title-level evidence coverage. The current method and protocol provide the infrastructure, but the thesis is not yet supported.

## Results available now

| Metric | Baseline | Selector | Delta | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Title recall@5 | 0.8190 | 0.8310 | +0.0120 | [+0.0025,+0.0215] | 0.0070 |
| Title F1 | 0.7483 | 0.7633 | +0.0150 | [+0.0011,+0.0289] | 0.0180 |
| Answer F1 | 0.6100 | 0.6127 | +0.0028 | [-0.0102,+0.0161] | 0.3440 |
| Answer-title product | 0.5170 | 0.5249 | +0.0079 | [-0.0061,+0.0225] | 0.1245 |

## Mandatory experiments before promotion

1. Add a predicted support-sentence component and run official HotpotQA answer, supporting-fact, and joint evaluation.
2. Pre-specify a broader action generator or high-coverage policy using train-only selection, then evaluate once on untouched queries. The generator must reduce the 797-query no-opportunity ceiling.
3. Complete at least one additional reader with the same frozen contexts and nested decisions.
4. If “federated” returns to the title, rerun candidate production through documented non-IID clients and measure payload/communication. Otherwise retain the non-federated title.
5. Pin exact reader/tokenizer revisions and archive a clean commit.

## Promotion criterion

Remove the status banner only if a new untouched evaluation shows a statistically supported answer or official joint gain, all official/custom metric names are separated, provenance remains exact, and any federated wording matches a genuine end-to-end experiment. Until then, `paper_submission_findings_v2.md` is the recommended submission manuscript.
