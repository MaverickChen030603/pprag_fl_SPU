from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA fullwiki data for FedE upstream training.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default="")
    return parser.parse_args()


def build_record(row: dict) -> dict | None:
    context = row.get("context", {})
    titles = context.get("title", []) or []
    sentences = context.get("sentences", []) or []
    supporting = row.get("supporting_facts", {})
    supporting_titles = supporting.get("title", []) or []
    if not supporting_titles:
        return None
    local_title2sentences = {title: sents for title, sents in zip(titles, sentences)}
    ordered_titles = []
    seen = set()
    for title in supporting_titles:
        if title in seen or title not in local_title2sentences:
            continue
        ordered_titles.append(title)
        seen.add(title)
    if not ordered_titles:
        return None
    merged_sections = []
    for title in ordered_titles:
        merged_sections.append(f"[{title}] " + " ".join(local_title2sentences[title]))
    merged_text = "\n".join(merged_sections).strip()
    if not merged_text:
        return None
    return {
        "question": row.get("question", ""),
        "company": row.get("answer", ""),
        "page": " | ".join(ordered_titles),
        "index": row.get("_id", ""),
        "reference": merged_text,
        "supporting_titles": ordered_titles,
    }


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "fullwiki")
    records = []
    skipped = 0
    for row in dataset[args.split]:
        if args.max_examples and len(records) >= args.max_examples:
            break
        record = build_record(row)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "hotpot_qa",
        "config": "fullwiki",
        "split": args.split,
        "max_examples": args.max_examples,
        "written_examples": len(records),
        "skipped_examples": skipped,
        "output": str(output_path),
    }
    metadata_path = Path(args.metadata_output).expanduser().resolve() if args.metadata_output else output_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
