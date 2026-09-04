# Limitation Section Draft

The main result is HotpotQA-centered. Although the result is statistically meaningful for joint_f1 and support-side metrics, answer_f1 is not significantly improved and should be described as preserved rather than improved.

Selector-level transfer to 2WikiMultiHopQA is not established. The 2Wiki diagnostic shows that a strong BM25 baseline is difficult to beat, that candidate exposure is limited, and that cross-dataset safety calibration is dataset-sensitive. Candidate generation beyond BM25 remains future work.

Oracle diagnostics are not inference-time methods. They are included only to estimate upper bounds and motivate future candidate/action generation research.
