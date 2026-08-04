# MARS-Route R2 Preregistration

R2 holds partition, local corpus, local indexes, `Bc=3`, local dense retrieval,
A0 5/5/5 transmission, 15 documents, global top-10, and server raw merge fixed.
Rank-percentile is auxiliary only. Stage R2-A uses Router-Dev N=100 to choose
only profile family/prototype count/candidate L. Stage R2-B uses all Router-Dev
N=300 to choose a fixed selector. Router-Calibration is reserved for set
weights and early stopping. Router-Holdout N=500 is not read for any selection.

No reader, final-test query, support label, answer, or reader output is used at
inference. Training labels are multi-label support-client sets from train only.
