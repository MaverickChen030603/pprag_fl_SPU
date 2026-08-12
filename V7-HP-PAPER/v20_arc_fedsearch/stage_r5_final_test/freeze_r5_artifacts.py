#!/usr/bin/env python3
"""Record immutable scientific and execution artifacts before R5 retrieval starts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BASE = "13091c6deb6b3868705a49041e89f578f14b4e0e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path, names: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    files = [path for path in root.rglob("*") if path.is_file() and (names is None or path.name in names)]
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode()); digest.update(sha256(path).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    output = run / "protocol/r5_frozen_artifact_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    git = lambda *values: subprocess.check_output(["git", *values], cwd=root, text=True).strip()
    execution_commit = git("rev-parse", "HEAD")
    changed = git("diff", "--name-only", f"{BASE}..{execution_commit}").splitlines()
    if any(not name.startswith("V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test/") for name in changed):
        raise RuntimeError(f"post-R4 commit changes scientific files: {changed}")
    v17 = root / "V7-HP-PAPER/v17_fedaction_rag"
    v20 = root / "V7-HP-PAPER/v20_arc_fedsearch"
    model_roots = {
        "flan": Path.home() / ".cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        "unifiedqa": Path.home() / ".cache/huggingface/hub/models--allenai--unifiedqa-v2-t5-large-1363200/snapshots/1d3b8e13b29dbd161494b0b15428378f4713c418",
    }
    rankers = {
        "hotpotqa": v20 / "stage_r3_probe_route/hotpot_transfer/models/logistic_seed_20260807.pkl",
        "2wikimultihopqa": v20 / "stage_r3_probe_route/ranker_training/models/2wikimultihopqa/logistic_seed_20260807.pkl",
        "musique": v20 / "stage_r3_probe_route/ranker_training/models/musique/logistic_seed_20260807.pkl",
    }
    indexes = {
        "hotpotqa": root / "V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite",
        "2wikimultihopqa": root / "V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite",
        "musique": root / "V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite",
    }
    tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"}
    checkpoint_names = tokenizer_names | {"config.json", "generation_config.json", "pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json"}
    payload = {
        "status": "frozen_before_retrieval", "scientific_base_commit": BASE, "execution_commit": execution_commit,
        "post_base_changed_files": changed, "working_tree_status": git("status", "--porcelain=v1").splitlines(),
        "router_checkpoint_hash": tree_hash(v17 / "partitions/centroids"),
        "probe_ranker_hash": {dataset: sha256(path) for dataset, path in rankers.items()},
        "probe_feature_schema_hash": sha256(v20 / "stage_r3_probe_route/run_compact_payload_audit.py"),
        "candidate_generator_hash": sha256(v20 / "stage_r3_probe_route/materialize_candidate_probe_packets.py"),
        # Hash the full snapshots so sharded model weights cannot escape the audit.
        "flan_checkpoint_hash": tree_hash(model_roots["flan"]),
        "unifiedqa_checkpoint_hash": tree_hash(model_roots["unifiedqa"]),
        "tokenizer_hash": {name: tree_hash(path, tokenizer_names) for name, path in model_roots.items()},
        "partition_hash": tree_hash(v17 / "partitions", {"topic_silo_manifest.json"}),
        "client_index_hash": {dataset: sha256(path) for dataset, path in indexes.items()},
        "local_retriever_hash": sha256(v20 / "stage_r3_probe_route/materialize_candidate_probe_packets.py"),
        "reader_prompt_hash": sha256(v20 / "stage_r4_frozen_reader/run_r4_reader.py"),
        "context_serializer_hash": sha256(root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test/materialize_r5_contexts.py"),
        "evaluation_script_hash": sha256(root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test/evaluate_r5_after_unseal.py"),
        "reader_revisions": {"flan": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a", "unifiedqa": "1d3b8e13b29dbd161494b0b15428378f4713c418"},
        "contract": {"candidate_L": 8, "probe_float32_features_per_client": 18, "probe_payload_bytes": 592, "client_budget": 3, "local_depth": 10, "docs_per_client": 5, "transmitted_docs": 15, "global_pool": 10, "reader_context_k": 5},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "execution_commit": execution_commit, "changed_files": len(changed)}, indent=2))


if __name__ == "__main__":
    main()
