#!/usr/bin/env python3
"""Route frozen R5 candidate clients with query-to-source-centroid similarity."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from train_ragroute_b3 import encode, load_encoder, query_id, rows, sha256


METHOD = "b6_dense_centroid_top3"


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
    parser.add_argument(
        "--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True
    )
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--r5-packets", type=Path, required=True)
    parser.add_argument("--r5-inputs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    packets = list(rows(args.r5_packets))
    r5_inputs = {query_id(row): row for row in rows(args.r5_inputs)}
    packet_ids = [str(row["query_id"]) for row in packets]
    if len(packet_ids) != 300 or set(packet_ids) != set(r5_inputs):
        raise ValueError("frozen R5 packets and inputs must contain the same 300 IDs")

    centroids = np.load(args.centroids).astype(np.float32)
    tokenizer, encoder = load_encoder(args.model, args.revision, args.device)
    embeddings = encode(
        tokenizer,
        encoder,
        [str(r5_inputs[str(row["query_id"])]["question"]) for row in packets],
        args.device,
        args.batch_size,
    ).astype(np.float32)
    scores = embeddings @ centroids.T

    args.output_dir.mkdir(parents=True)
    route_path = args.output_dir / "r5_routes_unlabeled.jsonl"
    with route_path.open("x", encoding="utf-8") as handle:
        for packet, row_scores in zip(packets, scores):
            candidates = [int(item["client_id"]) for item in packet["p0_candidate_records"]]
            if len(candidates) != 8 or len(set(candidates)) != 8:
                raise ValueError(f"invalid frozen candidate pool for {packet['query_id']}")
            selected = sorted(candidates, key=lambda client: (-float(row_scores[client]), client))[:3]
            handle.write(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "query_id": str(packet["query_id"]),
                        "method": METHOD,
                        "selected_clients": selected,
                        "candidate_clients": candidates,
                        "candidate_scores": {
                            str(client): float(row_scores[client])
                            for client in candidates
                        },
                        "feature_contract": "query_bge_dot_full_source_bge_centroid",
                        "probe_features_used": False,
                        "trained_router": False,
                        "r5_labels_used": False,
                        "reader_started": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "status": "routed_unlabeled",
        "method": METHOD,
        "dataset": args.dataset,
        "queries": len(packet_ids),
        "candidate_clients": 8,
        "client_budget": 3,
        "model": args.model,
        "revision": args.revision,
        "centroids_sha256": sha256(args.centroids),
        "r5_packets_sha256": sha256(args.r5_packets),
        "r5_inputs_sha256": sha256(args.r5_inputs),
        "r5_routes_sha256": sha256(route_path),
        "r5_labels_opened": False,
        "reader_started": False,
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "claim_restriction": "diagnostic_only_not_fresh_confirmation",
    }
    atomic_json(args.output_dir / "route_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
