#!/usr/bin/env python3
"""Fail-closed C2 preflight before any fresh retrieval begins."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    prereg = args.stage / "pre_registration"
    split = json.loads((prereg / "protocol/c2_fresh_split_manifest.json").read_text())
    audit = json.loads((prereg / "protocol/c2_zero_overlap_audit.json").read_text())
    c1 = json.loads((args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_c1_fresh_confirmation/protocol/c1_execution_preflight.json").read_text())
    assets = args.base / "inputs/c1_frozen_assets/V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes"
    local_indexes = {dataset: len(list((assets / dataset / "topic_silo").glob("client_*.sqlite"))) for dataset in split["datasets"]}
    passed = audit.get("status") == "pass" and all(len(rows) == 500 for rows in split["datasets"].values()) and c1.get("status") == "pass" and all(count == 20 for count in local_indexes.values())
    output = {"stage":"V20-C2-A","status":"pass" if passed else "integrity_failure","retrieval_started":False,"reader_started":False,"labels_scored":False,"split_sha256":sha256(prereg / "protocol/c2_fresh_split_manifest.json"),"zero_overlap_audit_sha256":sha256(prereg / "protocol/c2_zero_overlap_audit.json"),"c1_asset_preflight_sha256":sha256(args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_c1_fresh_confirmation/protocol/c1_execution_preflight.json"),"local_index_count":local_indexes,"git_head":subprocess.check_output(["git","-C",str(args.repo),"rev-parse","HEAD"],text=True).strip()}
    target=args.stage/"protocol/c2_execution_preflight.json"; target.parent.mkdir(exist_ok=True); target.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
