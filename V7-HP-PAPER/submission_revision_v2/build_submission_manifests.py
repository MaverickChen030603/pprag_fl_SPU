#!/usr/bin/env python3
"""Build data, fold, environment, and artifact manifests for submission v2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    configured = os.environ.get("FEDE4RAG_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
HERE = Path(__file__).resolve().parent
DATA = ROOT / "V7-HP4" / "data" / "hotpot_validation_1000.json"
DATA_META = DATA.with_suffix(".meta.json")
ACTIONS = ROOT / "V7-HP-PAPER" / "selector_v2_3" / "outputs" / "labels" / "action_labels.jsonl"
NESTED = HERE / "outputs" / "nested"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def query_fold(query_id: str, k: int = 5) -> int:
    return int(hashlib.md5(query_id.encode("utf-8")).hexdigest(), 16) % k


def actual_v23_folds(query_ids: list[str], k: int = 5) -> list[list[str]]:
    ordered = sorted(query_ids, key=lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16))
    return [ordered[i::k] for i in range(k)]


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def module_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ["torch", "transformers", "datasets", "sklearn", "numpy", "scipy", "matplotlib", "sentencepiece"]:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "installed-version-unknown"))
        except Exception as exc:
            versions[name] = f"MISSING: {type(exc).__name__}: {exc}"
    return versions


def source_reconstruction(data: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = {
        "attempted": False,
        "exact_id_sequence_match": False,
        "same_id_set": False,
        "source_rows": None,
        "normalized_candidates": None,
        "error": None,
    }
    support_rows: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        result["attempted"] = True
        normalized_ids = []
        source_by_id = {}
        for idx, raw in enumerate(dataset):
            item = dict(raw)
            qid = str(item.get("_id", item.get("id", idx)))
            context = item.get("context", {}) or {}
            titles = context.get("title", []) or []
            sentences = context.get("sentences", []) or []
            reference = " ".join(
                f"[{title}] " + " ".join(str(sentence) for sentence in body)
                for title, body in zip(titles, sentences)
            ).strip()
            facts = item.get("supporting_facts", {}) or {}
            fact_titles = facts.get("title", []) if isinstance(facts, dict) else []
            fact_ids = facts.get("sent_id", []) if isinstance(facts, dict) else []
            supporting_titles = []
            for title in fact_titles:
                title = str(title)
                if title and title not in supporting_titles:
                    supporting_titles.append(title)
            if not reference or not supporting_titles:
                continue
            normalized_ids.append(qid)
            source_by_id[qid] = {
                "query_id": qid,
                "supporting_facts": [
                    {"title": str(title), "sent_id": int(sent_id)}
                    for title, sent_id in zip(fact_titles, fact_ids)
                ],
            }
        rng = random.Random(seed)
        rng.shuffle(normalized_ids)
        rebuilt = normalized_ids[: len(data)]
        current = [str(row["_id"]) for row in data]
        result.update(
            {
                "source_rows": len(dataset),
                "normalized_candidates": len(normalized_ids),
                "exact_id_sequence_match": rebuilt == current,
                "same_id_set": set(rebuilt) == set(current),
            }
        )
        support_rows = [source_by_id[qid] for qid in current]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result, support_rows


def main() -> None:
    data = read_json(DATA)
    metadata = read_json(DATA_META)
    query_ids = [str(row["_id"]) for row in data]
    source_audit, support_rows = source_reconstruction(data, int(metadata["seed"]))
    folds = actual_v23_folds(query_ids, 5)

    question_counts = Counter(str(row.get("question", "")) for row in data)
    id_counts = Counter(query_ids)
    data_manifest = {
        "dataset": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "source": metadata.get("source"),
        "source_library": "Hugging Face datasets",
        "sampling": {
            "algorithm": "Normalize all valid validation rows, random.Random(seed).shuffle, then take first N",
            "seed": metadata.get("seed"),
            "requested_examples": metadata.get("requested_examples"),
            "actual_examples": metadata.get("actual_examples"),
            "source_reconstruction": source_audit,
        },
        "source_file": str(DATA.relative_to(ROOT)),
        "source_file_sha256": sha256(DATA),
        "metadata_file": str(DATA_META.relative_to(ROOT)),
        "metadata_file_sha256": sha256(DATA_META),
        "query_ids": query_ids,
        "duplicate_query_ids": [qid for qid, count in id_counts.items() if count > 1],
        "duplicate_question_count": sum(1 for count in question_counts.values() if count > 1),
        "duplicate_removal": "No post-sampling duplicate-removal pass. Audit reports duplicate IDs/questions explicitly.",
        "converted_schema": ["_id", "question", "answer", "supporting_titles", "reference"],
        "sentence_labels_in_converted_file": False,
        "sentence_labels_recovered_from_source": bool(support_rows),
        "sanitization": {
            "candidate_documents": "Client metadata and latent support/bridge annotations are sanitized before candidate generation.",
            "client_assignment": "client_{document_index mod 5}",
            "client_partition_type": "synthetic round-robin",
        },
    }
    write_json(HERE / "data_manifest.json", data_manifest)
    if support_rows:
        write_jsonl(HERE / "gold_supporting_sentence_labels.jsonl", support_rows)

    fold_manifest = {
        "algorithm": "Sort unique query IDs by integer MD5 hash; outer test fold i is ordered_ids[i::5]",
        "k": 5,
        "folds": [],
    }
    all_ids = set(query_ids)
    for fold_id, test_ids in enumerate(folds):
        test = set(test_ids)
        train = all_ids - test
        fold_manifest["folds"].append(
            {
                "fold_id": fold_id,
                "n_train": len(train),
                "n_test": len(test),
                "train_query_ids": sorted(train),
                "test_query_ids": test_ids,
                "overlap": len(train & test),
            }
        )
    write_json(HERE / "fold_manifest.json", fold_manifest)

    environment = {
        "host_os": platform.platform(),
        "versions": module_versions(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_worktree_clean": command_output(["git", "status", "--porcelain"]) == "",
        "gpu_snapshot": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "reader_model": "google/flan-t5-large",
        "reader_model_revision": "[NEEDS SOURCE FILE] No explicit Hugging Face revision was logged.",
        "tokenizer_revision": "[NEEDS SOURCE FILE] No explicit Hugging Face revision was logged.",
    }
    write_json(HERE / "environment_snapshot.json", environment)

    artifact_paths = [
        DATA,
        DATA_META,
        ACTIONS,
        ROOT / "V7-HP-PAPER" / "selector_v2_3" / "outputs" / "final_1000" / "per_example_delta.jsonl",
        NESTED / "nested_final_1000_summary.json",
        NESTED / "nested_per_example_delta.jsonl",
        NESTED / "nested_significance_report.json",
        NESTED / "nested_ablation_summary.json",
        NESTED / "nested_feature_audit.json",
        NESTED / "nested_fold_configs.json",
        NESTED / "risk_coverage_curve.csv",
        NESTED / "risk_coverage_figure.pdf",
        NESTED / "utility_weight_sensitivity.json",
        NESTED / "action_scope_statistics.json",
    ]
    inventory = []
    for path in artifact_paths:
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    write_json(
        HERE / "artifact_inventory.json",
        {"generated_by": str(Path(__file__).name), "artifacts": inventory},
    )
    print(json.dumps({"data": data_manifest["sampling"], "environment": environment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
