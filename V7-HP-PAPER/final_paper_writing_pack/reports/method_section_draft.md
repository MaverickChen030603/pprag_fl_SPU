# Method Section Draft

## 1. Federated Routing and Candidate Actions

We assume a federated RAG setting in which distributed clients expose candidate context actions to a downstream reader. A candidate action modifies the reader context, for example by inserting a bridge document or replacing a lower-ranked background document while preserving important answer anchors.

## 2. Policy-Action-to-Reader Gap

Routing-side signals can identify support-relevant evidence, but support relevance alone does not guarantee reader-side gain. A context action may improve support coverage while disrupting answer-bearing context or distracting the generator. We call this mismatch the policy-action-to-reader gap.

## 3. Answer-Neutral Positive Action Definition

We define `answer_safe` as an action whose reader answer quality is not harmed relative to the baseline context. We define `support_positive` as an action that improves support-side retrieval or evidence utility. We define `joint_positive` as an action that improves downstream joint reasoning. We define `paper_positive` as the subset of actions that are answer-safe and improve joint/support utility. An answer-neutral positive action is therefore one that preserves answer quality while improving joint/support behavior.

## 4. No-Leak Query-Level Cross-Fitting

The selector is evaluated under strict query-level cross-fitting. Held-out reader outcomes are not used for selecting actions. Gold answers and supporting facts are not inference features. The selector only uses features available to the action policy and calibration folds, with query-level separation between training/calibration and held-out evaluation.

## 5. Selector Variants and Calibration

We compare support-first selection, two-stage selection, paper-positive classification, safety removal, support-feature removal, and the final answer-neutral positive-action selector. Calibration chooses action-selection parameters on training folds and applies them to held-out folds.

## 6. Inference-Time Decision Rule

At inference time, the policy selects candidate actions predicted to be answer-neutral and positive under the calibrated selector. Oracle diagnostics are not used for inference and are reported only as upper-bound analyses.
