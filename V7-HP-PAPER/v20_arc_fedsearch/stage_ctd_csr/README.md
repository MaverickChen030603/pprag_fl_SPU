# CTD-CSR

Run `prepare_h0_splits.py` once to freeze fresh-holdout manifests, then run
H0 with `run_h0_dataset.sh <dataset> <cuda-device>`. Set `H0_MAX_QUERIES=5`
for the required dry-run. The script runs P0 and frozen REM-P twice and refuses
to overwrite completed or incomplete output.

Do not run CT-0 or train a Student unless `reports/h0_cross_dataset_gate.json`
records `h0_passed=true`.
