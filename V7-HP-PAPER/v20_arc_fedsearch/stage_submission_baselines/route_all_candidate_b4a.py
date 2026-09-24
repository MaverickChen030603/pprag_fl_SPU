#!/usr/bin/env python3
"""Create the high-cost all-candidate route for the frozen R5 Top-8 pool."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from materialize_posthoc_baselines import rows, sha256


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
METHOD = "b4a_all_candidate_top8_retrieval"


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
        packet_path = args.input_root / "retrieval" / f"{dataset}_probe_packets.jsonl"
        route_path = dataset_dir / "r5_routes_unlabeled.jsonl"
        count = 0
        with route_path.open("x", encoding="utf-8") as handle:
            for packet in rows(packet_path):
                candidates = [int(item["client_id"]) for item in packet["p0_candidate_records"]]
                if len(candidates) != 8 or len(set(candidates)) != 8:
                    raise ValueError(f"{dataset}/{packet['query_id']}: invalid frozen Top-8")
                handle.write(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "query_id": str(packet["query_id"]),
                            "method": METHOD,
                            "selected_clients": candidates,
                            "candidate_clients": candidates,
                            "feature_contract": "all_frozen_top8_candidates_no_routing",
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
            "selected_clients": 8,
            "documents_per_client": 5,
            "transmission_budget": 40,
            "r5_packets_sha256": sha256(packet_path),
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
            "status": "complete_all_candidate_routes",
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
