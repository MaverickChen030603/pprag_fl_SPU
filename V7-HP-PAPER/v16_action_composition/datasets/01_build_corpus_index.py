#!/usr/bin/env python3
"""Build a deduplicated SQLite FTS5 corpus index from a labeled source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from payload if isinstance(payload, list) else payload.get("data", payload.get("train", []))


def normalized(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def documents(row: dict[str, Any], dataset: str) -> Iterable[tuple[str, str, str]]:
    if dataset == "musique":
        for paragraph in row.get("paragraphs", []):
            title, text = str(paragraph.get("title", "")), str(paragraph.get("paragraph_text", ""))
            identity = normalized(title) + "\n" + normalized(text)
            if title and text:
                yield f"musique:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}", title, text
        return
    context = row.get("context", [])
    pairs = []
    if isinstance(context, dict):
        titles, sentences = context.get("title", []), context.get("sentences", [])
        pairs = [(title, sentences[index] if index < len(sentences) else []) for index, title in enumerate(titles)]
    else:
        pairs = [value[:2] for value in context if isinstance(value, (list, tuple)) and len(value) >= 2]
    for title, sentences in pairs:
        title = str(title)
        text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
        if title and text:
            yield f"{dataset}:{hashlib.sha1(normalized(title).encode('utf-8')).hexdigest()[:20]}", title, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT NOT NULL, text TEXT NOT NULL)")
    started, source_rows, observed = time.perf_counter(), 0, 0
    for row in rows(args.source):
        for doc_id, title, text in documents(row, args.dataset):
            observed += int(connection.execute("INSERT OR IGNORE INTO docs(doc_id,title,text) VALUES(?,?,?)", (doc_id, title, text)).rowcount > 0)
        source_rows += 1
        if source_rows % 2000 == 0:
            connection.commit()
    connection.commit()
    connection.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(title, text, content='docs', content_rowid='id')")
    connection.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
    connection.commit()
    actual = connection.execute("SELECT count(*) FROM docs").fetchone()[0]
    connection.close()
    manifest = {"status": "complete", "dataset": args.dataset, "source": str(args.source.resolve()), "source_rows": source_rows, "unique_documents": actual, "inserted_documents": observed, "seconds": time.perf_counter() - started, "index": str(args.output.resolve())}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
