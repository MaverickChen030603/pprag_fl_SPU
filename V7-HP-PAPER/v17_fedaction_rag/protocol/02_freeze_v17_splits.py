#!/usr/bin/env python3
"""Freeze V17 splits by reusing the audited V16 split implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v16_split_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-split-script", type=Path, required=True)
    parser.add_argument("--used-inventory", type=Path, required=True)
    parser.add_argument("--hotpot-source", type=Path, required=True)
    parser.add_argument("--two-wiki-source", type=Path, required=True)
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--development-size", type=int, default=1000)
    parser.add_argument("--calibration-size", type=int, default=1000)
    parser.add_argument("--final-test-size", type=int, default=2000)
    args = parser.parse_args()

    impl = load_module(args.v16_split_script)
    inventory = json.loads(args.used_inventory.read_text(encoding="utf-8"))
    used_ids = {key: set(map(str, values)) for key, values in inventory.get("used_query_ids_by_dataset", {}).items()}
    used_questions = {key: set(map(str, values)) for key, values in inventory.get("used_normalized_questions_by_dataset", {}).items()}
    unknown = used_questions.get("unknown", set())
    sizes = {
        "train": args.train_size,
        "development": args.development_size,
        "calibration": args.calibration_size,
        "final_test": args.final_test_size,
    }
    sources = {
        "hotpotqa": args.hotpot_source,
        "2wikimultihopqa": args.two_wiki_source,
        "musique": args.musique_source,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "V7-HP-PAPER-v17-federated-action-rag",
        "seed": args.seed,
        "split_sizes": sizes,
        "history_through": "V16",
        "final_test_status": "sealed_not_for_development",
        "datasets": {},
    }
    for offset, (dataset, source) in enumerate(sources.items()):
        manifest["datasets"][dataset] = impl.freeze_one(
            dataset,
            source,
            used_ids.get(dataset, set()),
            used_questions.get(dataset, set()) | unknown,
            args.output_root,
            sizes,
            args.seed + offset,
        )
    output = Path(__file__).with_name("dataset_split_manifest.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {str(path.resolve()): sha256(path) for path in args.output_root.rglob("*.jsonl")}
    Path(__file__).with_name("artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(output), "splits": {key: value["split_counts"] for key, value in manifest["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
