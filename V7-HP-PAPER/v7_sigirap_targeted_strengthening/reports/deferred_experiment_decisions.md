# Deferred Experiment Decisions

This package strengthens the frozen study without changing Full, its selector thresholds, or its two observed holdouts. The following experiments are deliberately deferred.

## Simplified RankRAG reproduction

**Decision: do not run.** A partial implementation would not establish parity with official code, checkpoints, reader adaptation, input pool, or context budget. Calling such a system RankRAG would add a reproducibility and fairness dispute rather than resolve the strong-baseline question. The independent CrossEncoder-Top5 analysis is the bounded, protocol-matched comparison used here.

## Second retriever

**Decision: do not run.** A new retriever would alter the candidate pool, available opportunities, action distribution, and selector inputs. It requires a new end-to-end frozen protocol and cannot be inserted as a small post-hoc control.

## Additional 2Wiki calibration search

**Decision: do not run.** Target outcomes and a prior calibration grid have already been observed. Searching new seeds, K values, thresholds, or coverage would be outcome chasing. The present work keeps the frozen zero-shot result and analyzes its structure and feature shift.

## New primary pruned method

**Decision: do not promote.** The pair-pruning study is development-only cost sensitivity. Both Hotpot holdouts have already been observed, so no untouched split remains for an independent non-inferiority or confirmation test. No pruned row replaces Full.

## Boundary

These deferrals protect the main claim: a same-source, bounded-pool quality-risk-cost result under one frozen retriever, generator, selector, and reader protocol. They are not claims that the deferred methods would fail.
