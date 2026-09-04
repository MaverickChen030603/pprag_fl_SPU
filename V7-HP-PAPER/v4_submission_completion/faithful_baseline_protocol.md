# Faithful Baseline Protocol

## Selection decision

We considered Reader-Centered Passage Selection, SetR, RECOMP, and RankRAG. RECOMP was selected because an author-maintained implementation and an author-released HotpotQA extractive-compressor checkpoint were directly executable under the frozen V4 evaluation. The SetR repository did not expose a completed evaluation path suitable for this run, and no equivalent executable official package was located for Reader-Centered Passage Selection during the audit. This availability decision was made before inspecting comparison outcomes.

## Reproduction contract

- Official implementation: `https://github.com/carriex/recomp` at commit `51d4432`.
- Author checkpoint: `fangyuan/hotpotqa_extractive_compressor`.
- Paper/code hyperparameters: five input documents and one selected sentence.
- Data: the same frozen 1,000 HotpotQA development queries used by V4.
- Input context: the exact frozen HybridSoftRetriever Top-5 documents.
- Context budget: RECOMP compresses the same Top-5 pool; no alternative BM25 baseline is introduced.
- Reader: the same frozen FLAN-T5-Large reader and prompt used for baseline and V4.
- Tuning: no threshold, checkpoint, prompt, or hyperparameter is tuned on the 3,000-query holdout.
- Metrics: official answer, supporting-fact, and joint EM/F1. Supporting-fact scoring is an explicit extension that treats the selected sentence as the predicted support fact.

## Classification and limitation

The comparison is classified as **faithful method reproduction with standardized reader adaptation**, not an exact end-to-end reproduction of the RECOMP paper. The official compressor, checkpoint, and compression budget are retained, while the original FLAN-UL2 reader is replaced to isolate context construction under the V4 reader. This makes the downstream comparison controlled but narrower than reproducing the original paper's full stack.
