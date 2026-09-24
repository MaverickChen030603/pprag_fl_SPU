#!/usr/bin/env python3
"""Create the high-cost all-client retrieval route for the frozen R5 set."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from materialize_posthoc_baselines import query_id, rows, sha256


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
METHOD = "b4_all_client_retrieval"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summaries = {}
    for dataset in DATASETS:
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True)
        input_path = args.input_root / "protocol" / f"{dataset}_final_test_inputs_n300.jsonl"
        route_path = dataset_dir / "r5_routes_unlabeled.jsonl"
        count = 0
        with route_path.open("x", encoding="utf-8") as handle:
            for source in rows(input_path):
                qid = query_id(source)
                handle.write(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "query_id": qid,
                            "method": METHOD,
                            "selected_clients": list(range(20)),
                            "candidate_clients": list(range(20)),
                            "feature_contract": "all_clients_no_routing",
                            "probe_features_used": False,
                            "trained_router": False,
                            "r5_labels_used": False,
                            "reader_started": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
        if count != 300:
            raise ValueError(f"{dataset}: expected 300 routes, found {count}")
        manifest = {
            "status": "routed_unlabeled",
            "method": METHOD,
            "dataset": dataset,
            "queries": count,
            "selected_clients": 20,
            "documents_per_client": 5,
            "transmission_budget": 100,
            "r5_inputs_sha256": sha256(input_path),
            "r5_routes_sha256": sha256(route_path),
            "r5_labels_opened": False,
            "reader_started": False,
            "evidence_role": "retrospective_post_hoc_r5_revealed",
            "claim_restriction": "diagnostic_only_not_fresh_confirmation",
        }
        atomic_json(dataset_dir / "route_manifest.json", manifest)
        summaries[dataset] = manifest
    atomic_json(
        args.output_dir / "route_manifest.json",
        {
            "status": "complete_all_client_routes",
            "method": METHOD,
            "datasets": summaries,
            "r5_labels_opened": False,
            "reader_started": False,
            "evidence_role": "retrospective_post_hoc_r5_revealed",
        },
    )
    print(json.dumps({"method": METHOD, "datasets": sorted(summaries)}, indent=2))


if __name__ == "__main__":
    main()
