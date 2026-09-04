# Final Revision Summary

- Reordered Method so Full is primary; Lite is a review-driven simplification experiment.
- Added the untouched 3,405-query holdout to the main evidence with exact paired statistics.
- Recomputed selected-policy wins, losses, ties, quantiles, harm rates, and exact fallback behavior.
- Replaced the Top-1-centered RECOMP discussion with a 660-token holdout comparison and fair objective boundary.
- Added a synchronized 50-warmup/500-query end-to-end benchmark for five frozen systems. Full is 213.48 ms/query versus 140.88 ms/query for Frozen Top-5.
- Preserved Lite non-inferiority failure and 2Wiki calibration failure.
- Removed unresolved measurement placeholders and narrowed claims to bounded same-source context construction.
