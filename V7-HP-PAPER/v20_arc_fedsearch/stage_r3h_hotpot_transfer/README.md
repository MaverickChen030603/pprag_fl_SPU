# R3-H HotpotQA Transfer

Run only through `run_r3h_hotpot.sh`. The runner seals train/holdout splits,
uses the frozen R3 packet and Logistic Regression implementations by absolute
path, and refuses to overwrite completed outputs. Reader and final test are
not part of this stage.
