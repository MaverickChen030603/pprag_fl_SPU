#!/usr/bin/env python3
"""Materialize official train sources without touching prior dev/test splits."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_hotpot(pattern: str, output: Path) -> int:
    from datasets import Dataset

    shards = sorted(Path(path) for path in glob.glob(pattern, recursive=True) if "train" in Path(path).name)
    if not shards:
        raise FileNotFoundError(f"No Hotpot train Arrow shards match {pattern}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for shard in shards:
            dataset = Dataset.from_file(str(shard))
            for row in dataset:
                row = dict(row)
                row["query_id"] = str(row.get("id", row.get("_id")))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def prepare_two_wiki(archive: Path, member: str, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member)
        with source.open(info) as reader, output.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
    # Counting `_id` is memory-safe and exact for the official schema.
    count = 0
    needle = b'"_id"'
    with output.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            count += chunk.count(needle)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-arrow-glob", required=True)
    parser.add_argument("--two-wiki-archive", type=Path, required=True)
    parser.add_argument("--two-wiki-member", default="data/train.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "sources")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    hotpot_path = output / "hotpotqa_distractor_train.jsonl"
    two_wiki_path = output / "2wikimultihopqa_train.json"
    hotpot_count = prepare_hotpot(args.hotpot_arrow_glob, hotpot_path)
    two_wiki_count = prepare_two_wiki(args.two_wiki_archive, args.two_wiki_member, two_wiki_path)
    manifest = {
        "provenance": "official_train_sources_only",
        "hotpotqa": {"path": str(hotpot_path), "rows": hotpot_count, "sha256": sha256(hotpot_path)},
        "2wikimultihopqa": {"path": str(two_wiki_path), "rows": two_wiki_count, "sha256": sha256(two_wiki_path)},
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

