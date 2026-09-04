#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
V20 = ROOT / "V7-HP-PAPER/v20_arc_fedsearch"
R3 = V20 / "stage_r3_probe_route/hotpot_transfer"
R4 = V20 / "stage_r4_frozen_reader"
U1 = R4 / "u1_20260812"
POOL = R4 / "centralized_reference/hotpotqa_centralized_pool.jsonl"
POOL_MANIFEST = POOL.with_suffix(".manifest.json")
GENERATOR = ROOT / "V7-HP-PAPER/v17_fedaction_rag/retrieval/01_generate_federated_pools.py"
PACKET = R3 / "packets/probe_holdout.jsonl"
FORBIDDEN = {
    "answer", "answers", "aliases", "supporting_facts", "supporting_titles",
    "gold_answer", "gold_answers", "gold_support", "gold_document_ids",
    "answer_em", "answer_f1", "support_em", "support_f1", "reader_outcome",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def nested_keys(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        found.update(map(str, value))
        for item in value.values():
            found.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(nested_keys(item))
    return found


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


packets = [json.loads(line) for line in PACKET.read_text().splitlines() if line.strip()]
query_ids = [str(row["query_id"]) for row in packets]
pool_rows = [json.loads(line) for line in POOL.read_text().splitlines() if line.strip()]
pool_ids = [str(row["query_id"]) for row in pool_rows]
manifest = json.loads(POOL_MANIFEST.read_text())
contract_ok = (
    manifest.get("status") == "complete"
    and manifest.get("dataset") == "hotpotqa"
    and manifest.get("partition") == "centralized"
    and int(manifest.get("queries", -1)) == 300
    and int(manifest.get("client_budget", -1)) == 1
    and int(manifest.get("local_k", -1)) == 10
    and int(manifest.get("context_budget", -1)) == 5
    and int(manifest.get("action_pool_size", -1)) == 10
    and manifest.get("encoder") == "BAAI/bge-base-en-v1.5"
    and manifest.get("gold_or_support_injection") is False
    and manifest.get("random_padding") is False
)
rows_ok = (
    len(pool_rows) == 300
    and len(set(pool_ids)) == 300
    and pool_ids == query_ids
    and not (nested_keys(pool_rows) & FORBIDDEN)
    and POOL.read_bytes().endswith(b"\n")
    and all(
        row.get("dataset") == "hotpotqa"
        and row.get("partition") == "centralized"
        and row.get("selected_clients") == [0]
        and int(row.get("client_budget", -1)) == 1
        and int(row.get("local_k", -1)) == 10
        and int(row.get("pool_size", -1)) == 10
        and row.get("router") == "centralized_global_hybrid"
        and len(row.get("pool", [])) >= 10
        for row in pool_rows
    )
)
if not contract_ok or not rows_ok:
    raise ValueError("completed centralized pool failed frozen contract")

cache = U1 / "centralized_cache/complete_300.jsonl"
cache_manifest = U1 / "centralized_cache/complete_300.manifest.json"
if cache.exists() or cache_manifest.exists():
    raise FileExistsError(cache)
cache.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(POOL, cache)
shutil.copyfile(POOL_MANIFEST, cache_manifest)
cache.chmod(0o444)
cache_manifest.chmod(0o444)

contexts = []
for row in pool_rows:
    top10 = row["pool"][:10]
    docs = [
        {"doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "text": str(doc["text"])}
        for doc in top10[:5]
    ]
    ids = [doc["doc_id"] for doc in docs]
    contexts.append({
        "query_id": str(row["query_id"]),
        "method": "centralized",
        "retrieval_method": "v17_frozen_centralized_global_hybrid_replay",
        "selected_clients": [0],
        "retrieved_doc_ids": [str(doc["doc_id"]) for doc in top10],
        "reader_context_doc_ids": ids,
        "reader_context_docs": docs,
        "document_order_sha256": digest(ids),
        "context_sha256": canonical_hash({"reader_context_docs": docs}),
        "global_pool_size": 10,
        "reader_context_k": 5,
        "source": str(cache),
        "gold_or_answer_used": False,
    })
context_path = U1 / "contexts/centralized.jsonl"
if context_path.exists():
    raise FileExistsError(context_path)
with context_path.open("x", encoding="utf-8") as f:
    for row in contexts:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    f.flush()
    os.fsync(f.fileno())
context_path.chmod(0o444)

context_manifest = {
    "status": "frozen",
    "dataset": "hotpotqa",
    "method": "centralized",
    "retrieval_method": "v17_frozen_centralized_global_hybrid_replay",
    "n": 300,
    "query_id_order_sha256": digest(query_ids),
    "query_id_set_sha256": digest(sorted(query_ids)),
    "context_file": str(context_path),
    "context_file_sha256": sha256(context_path),
    "per_query_context_hashes_sha256": digest([row["context_sha256"] for row in contexts]),
    "reader_context_k": 5,
    "global_pool_size": 10,
    "source_pool_sha256": sha256(POOL),
    "source_pool_manifest_sha256": sha256(POOL_MANIFEST),
    "copied_cache_sha256": sha256(cache),
    "generator_path": str(GENERATOR),
    "generator_sha256": sha256(GENERATOR),
    "generator_access_contract": "qid(row) and row['question'] only; no gold field references in retrieval loop",
    "gold_or_answer_used": False,
    "schema_forbidden_fields_found": [],
}
write_json(U1 / "input_manifests/centralized_manifest.json", context_manifest)
audit = {
    "status": "SAFE_REUSE_COMPLETE_300",
    "state_snapshot_pool_sha256": "d79b1c4b1af8282a2460f8232c2da5df378b14517c3b6404b2213f85044baf5d",
    "current_pool_sha256": sha256(POOL),
    "pool_unchanged_since_u1_snapshot": sha256(POOL) == "d79b1c4b1af8282a2460f8232c2da5df378b14517c3b6404b2213f85044baf5d",
    "old_pids_absent_at_u1_snapshot": True,
    "u1_resumed_old_pipeline": False,
    "u1_reran_centralized_retrieval": False,
    "all_rows_valid": True,
    "last_line_complete": True,
    "unique_query_ids": True,
    "canonical_order_match": True,
    "contract_match": contract_ok,
    "forbidden_fields_found": [],
    "source_process_label_free_semantics": True,
    "source_process_opened_gold_bearing_split": True,
    "new_u1_generation_opened_gold_bearing_split": False,
    "cache_path": str(cache),
    "cache_sha256": sha256(cache),
    "context_manifest": context_manifest,
}
write_json(U1 / "centralized_cache/centralized_reuse_audit.json", audit)
print(json.dumps({"status": audit["status"], "contexts": len(contexts), "sha256": sha256(context_path)}, indent=2))
