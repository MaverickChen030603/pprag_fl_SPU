#!/usr/bin/env python3
"""Materialize label-free, physically separate SQLite FTS indexes per client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_local(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT NOT NULL, text TEXT NOT NULL)")
    connection.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(title, text, content='docs', content_rowid='id')")
    return connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    if len(assignment) == 0 or any(client < 0 or client >= args.m for client in assignment.values()):
        raise ValueError("assignment is empty or contains an invalid client ID")
    output = args.output_root / args.dataset / args.partition
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("assignment_sha256") == sha256(args.assignment) and all(
            (output / f"client_{client:02d}.sqlite").exists() for client in range(args.m)
        ):
            print(json.dumps({"status": "reused", "output": str(output), "clients": args.m}, indent=2))
            return
        raise FileExistsError(f"incomplete or mismatched local index output: {output}; use --overwrite")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing local index directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = [output / f"client_{client:02d}.sqlite.building" for client in range(args.m)]
    finals = [output / f"client_{client:02d}.sqlite" for client in range(args.m)]
    connections = [open_local(path) for path in temporary]
    counts = [0] * args.m
    source = sqlite3.connect(args.source_index)
    try:
        cursor = source.execute("SELECT doc_id,title,text FROM docs ORDER BY id")
        for doc_id, title, text in cursor:
            client = assignment.get(str(doc_id))
            if client is None:
                raise KeyError(f"source document without client assignment: {doc_id}")
            connections[client].execute(
                "INSERT INTO docs(doc_id,title,text) VALUES (?,?,?)", (doc_id, title, text)
            )
            counts[client] += 1
        for connection in connections:
            connection.execute("INSERT INTO docs_fts(docs_fts) VALUES ('rebuild')")
            connection.commit()
            connection.close()
        for temporary_path, final_path in zip(temporary, finals):
            os.replace(temporary_path, final_path)
    except Exception:
        for connection in connections:
            connection.close()
        raise
    finally:
        source.close()
    manifest = {
        "status": "complete",
        "dataset": args.dataset,
        "partition": args.partition,
        "clients": args.m,
        "source_index": str(args.source_index.resolve()),
        "assignment": str(args.assignment.resolve()),
        "assignment_sha256": sha256(args.assignment),
        "client_document_counts": counts,
        "total_documents": sum(counts),
        "files": {str(client): {"path": str(path.resolve()), "sha256": sha256(path)} for client, path in enumerate(finals)},
        "gold_labels_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "clients": args.m, "documents": sum(counts)}, indent=2))


if __name__ == "__main__":
    main()
