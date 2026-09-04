#!/usr/bin/env python3
"""Build a true dataset corpus and SQLite FTS5 BM25 index."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from retrieval_common import documents, iter_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--commit-every", type=int, default=5000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT NOT NULL, text TEXT NOT NULL)")
    start = time.perf_counter()
    rows = 0
    observed = 0
    for row in iter_rows(args.source):
        for doc in documents(row, args.dataset):
            cursor = connection.execute(
                "INSERT OR IGNORE INTO docs(doc_id,title,text) VALUES(?,?,?)",
                (doc["doc_id"], doc["title"], doc["text"]),
            )
            observed += int(cursor.rowcount > 0)
        rows += 1
        if rows % args.commit_every == 0:
            connection.commit()
        if args.max_rows and rows >= args.max_rows:
            break
    connection.commit()
    connection.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(title, text, content='docs', content_rowid='id')")
    connection.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    elapsed = time.perf_counter() - start
    actual = connection.execute("SELECT count(*) FROM docs").fetchone()[0]
    connection.close()
    summary = {"status": "complete", "dataset": args.dataset, "source_rows": rows, "unique_documents": actual, "inserted_documents": observed, "seconds": elapsed, "index": str(args.output.resolve())}
    summary_path = args.output.with_suffix(".manifest.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

