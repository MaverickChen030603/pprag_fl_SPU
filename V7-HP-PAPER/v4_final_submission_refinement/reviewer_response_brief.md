# Reviewer Response Brief

## Core response

The paper's claim is deliberately narrow: a semantic opportunity generator plus fully nested reader-safe selection yields small but statistically reliable gains on a frozen same-source holdout. We do not claim that the opportunity gap is solved, that the external transfer is significant, or that the full support pipeline replicates across readers.

## Likely questions

**Q1: The effects are small. Why are they meaningful?**  
The intervention changes only 25.8% of holdout contexts, preserves the original context elsewhere, and improves answer, support, and joint F1 without retuning. The contribution is the controlled conversion of context opportunity into reader gains, not a large leaderboard jump.

**Q2: Is the 3,000-query sample a test set?**  
No. It is a disjoint same-source HotpotQA holdout evaluated after freezing the pipeline. We call it frozen holdout confirmation and make no external-domain claim from it.

**Q3: Was joint F1 formally pre-specified as primary?**  
No immutable pre-run hierarchy was found. We present joint F1 as the headline holdout metric and report all three unadjusted paired-bootstrap tests. We do not claim formal familywise control.

**Q4: Does the method transfer to 2Wiki?**  
The point estimates for answer and joint F1 are positive, but all intervals include zero and support F1 is flat. The result bounds catastrophic collapse but does not establish reliable transfer. Safety calibration worsens.

**Q5: Is the RECOMP comparison fair?**  
It uses official code/checkpoint and the same Top-5 input, but not the same output budget or original reader. RECOMP emits about 7.35% of baseline context tokens. We therefore use it only as a standardized-reader compatibility analysis and move details to the appendix.

**Q6: Do all generator components help?**  
No. Pair complementarity and two-document chains have the clearest contributions. The document model trades breadth for answer safety; other feature effects are mixed. The paper states this explicitly.

**Q7: Is the second reader an independent replication?**  
It is an answer-reader directional replication on identical contexts. The support predictor is shared, so it is not an independent full-pipeline replication.
