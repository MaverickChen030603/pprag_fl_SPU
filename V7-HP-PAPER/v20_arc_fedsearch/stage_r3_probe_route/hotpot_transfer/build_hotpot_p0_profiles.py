#!/usr/bin/env python3
"""Materialize the frozen P0-only Hotpot profile format used by R3."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--p0-centroids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    centroids = np.load(args.p0_centroids).astype(np.float32)
    if centroids.shape[0] != 20:
        raise AssertionError(f"expected 20 P0 centroids, got {centroids.shape}")
    profiles = []
    for client in range(20):
        connection = sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite")
        count = int(connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
        connection.close()
        profiles.append({"dataset": "hotpotqa", "client_id": client, "collection_size": count,
                         "p0_single_centroid": centroids[client].astype(float).tolist()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "client_profiles.json").write_text(json.dumps({
        "dataset": "hotpotqa", "profiles": profiles,
        "profile_contract": "reused_frozen_P0_centroids_only_no_new_profile_features",
        "gold_or_development_fields_used": False,
    }) + "\n", encoding="utf-8")
    (args.output_dir / "profile_manifest.json").write_text(json.dumps({
        "dataset": "hotpotqa", "clients": 20, "source_centroids": str(args.p0_centroids.resolve()),
        "local_index_root": str(args.local_index_root.resolve()), "new_profile_features": False,
        "gold_or_development_fields_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": "hotpotqa", "clients": 20, "status": "complete"}))


if __name__ == "__main__":
    main()
