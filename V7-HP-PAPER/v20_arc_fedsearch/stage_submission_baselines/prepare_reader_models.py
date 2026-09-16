#!/usr/bin/env python3
"""Download the exact Reader revisions recorded by the frozen V20 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "flan": (
        "google/flan-t5-large",
        "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    ),
    "unifiedqa": (
        "allenai/unifiedqa-v2-t5-large-1363200",
        "1d3b8e13b29dbd161494b0b15428378f4713c418",
    ),
}
MODEL_PATTERNS = {
    "flan": (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "unifiedqa": (
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for reader, (repository, revision) in MODELS.items():
        snapshot = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                cache_dir=args.cache_dir,
                allow_patterns=MODEL_PATTERNS[reader],
            )
        )
        files = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "relative_path": str(path.relative_to(snapshot)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        records[reader] = {
            "repository": repository,
            "revision": revision,
            "snapshot": str(snapshot.resolve()),
            "files": files,
            "total_bytes": sum(row["bytes"] for row in files),
        }
    payload = {"status": "complete", "models": records}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
