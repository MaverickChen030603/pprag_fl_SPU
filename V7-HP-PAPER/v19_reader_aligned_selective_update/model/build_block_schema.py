#!/usr/bin/env python3
"""Materialize the immutable V19 block schema from the cached BGE model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_blocks import block_parameters, default_block_specs, inject_lora_blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--payload-table", type=Path, required=True)
    args = parser.parse_args()

    encoder = AutoModel.from_pretrained(args.model, local_files_only=True)
    specs = default_block_specs(len(encoder.encoder.layer))
    inject_lora_blocks(encoder, rank=args.rank, alpha=args.alpha)
    blocks = block_parameters(encoder, specs)
    rows = []
    for spec in specs:
        parameters = blocks[spec.block_id]
        count = sum(parameter.numel() for parameter in parameters)
        byte_count = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        rows.append({"block_id": spec.block_id, "layer": spec.layer, "module_type": spec.module_type,
                     "targets": list(spec.targets), "parameters": count, "payload_bytes_fp32": byte_count})
    payload = {"schema_version": 1, "base_model": args.model, "rank": args.rank, "alpha": args.alpha,
               "block_count": len(rows), "blocks": rows, "total_payload_bytes_fp32": sum(row["payload_bytes_fp32"] for row in rows)}
    args.schema.parent.mkdir(parents=True, exist_ok=True)
    args.schema.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.payload_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_id", "layer", "module_type", "parameters", "payload_bytes_fp32"])
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in rows])
    print(json.dumps({"status": "complete", "blocks": len(rows), "bytes": payload["total_payload_bytes_fp32"]}, indent=2))


if __name__ == "__main__":
    main()
