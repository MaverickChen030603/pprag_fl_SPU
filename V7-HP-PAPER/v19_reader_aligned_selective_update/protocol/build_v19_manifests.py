#!/usr/bin/env python3
"""Copy audited V17 split identities into V19 without reading sealed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17-manifest", type=Path, required=True)
    parser.add_argument("--v17-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.v17_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.v17_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("V17 source split audit is not passing")
    output = {"schema_version": 1, "experiment": "V7-HP-PAPER-v19-reader-aligned-selective-update",
              "inherits": {"split_manifest": str(args.v17_manifest.resolve()), "split_manifest_sha256": digest(args.v17_manifest),
                           "no_leak_audit": str(args.v17_audit.resolve()), "no_leak_audit_sha256": digest(args.v17_audit)},
              "final_test_status": "sealed_not_for_development", "datasets": manifest["datasets"],
              "development_contract": "train/development/calibration only; final_test inputs and sealed labels are forbidden before final freeze."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    no_leak = {"status": "pass", "checks": {"inherits_v17_passing_audit": True, "final_test_declared_sealed": True,
        "final_test_labels_opened": False, "v19_dev_sources": ["train", "development", "calibration"]}, "source_manifest_sha256": digest(args.v17_manifest)}
    args.output.with_name("no_leak_audit.json").write_text(json.dumps(no_leak, indent=2) + "\n", encoding="utf-8")
    seal = {"status": "sealed", "rule": "Do not open or copy final_test input/label files during Stages 0-3.",
            "final_test_fingerprints": {name: spec["files"]["final_test"] for name, spec in manifest["datasets"].items()}}
    args.output.with_name("final_seal_manifest.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "datasets": list(manifest["datasets"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
