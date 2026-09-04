# Training Signal Audit

Train pairs: 1000; active clients: 20/20.

The loss is in-batch contrastive with batch size 8. Its only negative source is another query's positive passage. Thus the initial smoke can establish adapter trainability but cannot test the stated hard-negative objective. The next legal single-factor positive control is PC-1: retain model, rank, learning rate, and steps while replacing implicit-only negatives with a precomputed train-only hard-negative manifest.
