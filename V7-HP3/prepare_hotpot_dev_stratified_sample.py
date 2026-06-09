#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def answer_type(answer: str) -> str:
    norm = str(answer).strip().lower()
    if norm in {"yes", "no"}:
        return norm
    return "span"


def supporting_bucket(row: dict[str, Any]) -> str:
    sf = row.get("supporting_facts") or {}
    n = len(sf.get("title", []) or [])
    if n <= 2:
        return "sp2"
    if n <= 4:
        return "sp3_4"
    return "sp5p"


def to_compact_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    ctx = row.get("context") or {}
    titles = [str(t) for t in ctx.get("title", [])]
    sent_lists = ctx.get("sentences", [])
    reference_parts = []
    for title, sentences in zip(titles, sent_lists):
        text = " ".join(str(s) for s in sentences)
        reference_parts.append(f"[{title}] {text}")
    sf = row.get("supporting_facts") or {}
    supporting_titles = []
    seen = set()
    for title in sf.get("title", []) or []:
        title = str(title)
        if title not in seen:
            supporting_titles.append(title)
            seen.add(title)
    return {
        "_id": str(row.get("id", row.get("_id", f"dev-{idx}"))),
        "question": str(row.get("question", "")),
        "answer": row.get("answer", ""),
        "company": row.get("answer", ""),
        "page": supporting_titles[0] if supporting_titles else (titles[0] if titles else ""),
        "reference": " ".join(reference_parts),
        "supporting_titles": supporting_titles,
        "hp2_sample_stratum": f"{answer_type(row.get('answer', ''))}:{supporting_bucket(row)}",
        "hp2_source_split": "hotpot_qa/fullwiki/validation",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="V7-HP3/data/hotpot_dev_stratified_300.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260608)
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(ds):
        row = dict(row)
        key = f"{answer_type(row.get('answer', ''))}:{supporting_bucket(row)}"
        row["_sample_index"] = i
        buckets[key].append(row)
    rng = random.Random(args.seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    target_per = max(1, args.n // max(len(keys), 1))
    selected = []
    remaining = args.n
    for key in keys:
        take = min(len(buckets[key]), target_per, remaining)
        selected.extend(buckets[key][:take])
        buckets[key] = buckets[key][take:]
        remaining -= take
    cursor = 0
    while remaining > 0 and any(buckets.values()):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop(0))
            remaining -= 1
        cursor += 1
    rng.shuffle(selected)
    records = [to_compact_record(row, idx) for idx, row in enumerate(selected[: args.n])]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "output": str(out),
        "n": len(records),
        "seed": args.seed,
        "strata_counts": {k: sum(1 for r in records if r["hp2_sample_stratum"] == k) for k in sorted({r["hp2_sample_stratum"] for r in records})},
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
