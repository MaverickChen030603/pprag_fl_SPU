#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuda-device>" >&2
  exit 2
fi

DEVICE="$1"
ROOT=/home/iiserver31/projects/FedE4RAG-main
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
STAGE="$V20/stage_r3h_hotpot_transfer"
MODELS="$STAGE/train/models"

[[ -f "$MODELS/frozen_model_manifest.json" ]] || {
  echo "missing frozen R3-H model manifest; do not start a replacement run" >&2
  exit 3
}

# The completed train run is immutable. Recreate only the tiny, label-free smoke
# state after a fresh checkout, then validate the three saved model objects.
if [[ ! -f "$STAGE/smoke/complete.json" ]]; then
  bash "$STAGE/run_r3h_smoke.sh" "$DEVICE"
fi

python3 - "$MODELS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

models = Path(sys.argv[1])
manifest = json.loads((models / "frozen_model_manifest.json").read_text())
expected = manifest["models"]
for name, metadata in expected.items():
    path = models / name
    if not path.is_file():
        raise SystemExit(f"missing frozen model: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise SystemExit(f"checksum mismatch: {path}")
if len(expected) != 3:
    raise SystemExit(f"expected exactly three seed models, found {len(expected)}")
print(json.dumps({"status": "R3-H checkpoint verified", "models": sorted(expected)}))
PY

echo "R3-H recovery check passed. The completed train run remains frozen; do not run run_r3h_train.sh."
