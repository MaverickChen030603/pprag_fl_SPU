# Submission Readiness Report

```yaml
fully_nested_generator: true
fully_nested_selector: true
development_official_metrics: true
same_source_confirmatory_holdout: true
multi_reader_answer_replication: true
independent_multi_reader_support_replication: false
faithful_external_baseline: true
external_dataset_validation: true
reproducibility_complete: true
citation_complete: true
final_grade: main_conference_ready
```

## Decision

The package meets the task's `main_conference_ready` rule because a faithful-method RECOMP comparison and a strictly frozen external 2Wiki validation are both complete, while the fully nested development and frozen same-source holdout results support the core candidate-opportunity and reader-safe selection story. "Ready" refers to submission completeness, not to an unrestricted empirical claim: external answer/joint gains are directional rather than significant, the component ablation is mixed, and independent support replication remains absent.

## Submission posture

Submit the main-conference version with the narrow title and bounded claims. Lead with the 3,000-query HotpotQA confirmation, use 2Wiki as a generalization-boundary experiment, and describe RECOMP as a faithful method reproduction with reader adaptation. Do not introduce Federated RAG, privacy, SOTA, full opportunity success, or reader-independent support language during final polishing.
