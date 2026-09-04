#!/usr/bin/env python3
"""Replay frozen outer-test contexts with a second locally cached QA reader."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from v3_common import OUTPUTS, REPORTS, answer_scores, ensure_layout, load_source_examples, markdown_table, paired_bootstrap, read_json, read_jsonl, title_metrics, write_json, write_jsonl


DEFAULT_READER = "/home/iiserver31/.cache/huggingface/hub/models--allenai--unifiedqa-v2-t5-large-1363200/snapshots/1d3b8e13b29dbd161494b0b15428378f4713c418"
READER_REVISION = "1d3b8e13b29dbd161494b0b15428378f4713c418"
SEED = 20260713


def make_prompt(question: str, docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    context = " ".join(f"[{doc['title']}] {doc['text']}" for doc in docs)[:max_chars]
    return f"{question}\n{context}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("V3_SECOND_READER", DEFAULT_READER))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    nested_path = OUTPUTS / "nested_selector/v3_nested_summary.json"
    nested = read_json(nested_path) if nested_path.exists() else {"status": "not_run"}
    if nested.get("status") != "complete":
        reason = "Multi-reader replay requires frozen v3 selected contexts, which were not produced after the opportunity gate failed."
        write_json(OUTPUTS / "multi_reader/multi_reader_summary.json", {"status": "skipped_by_opportunity_gate", "contexts_frozen": False, "reason": reason})
        (OUTPUTS / "tables/multi_reader_table.md").write_text("# Multi-Reader Frozen-Context Replay\n\nStatus: **skipped by opportunity gate**.\n", encoding="utf-8")
        (REPORTS / "multi_reader_report.md").write_text("# Multi-Reader Report\n\nStatus: **skipped by opportunity gate**. " + reason + "\n", encoding="utf-8")
        print(json.dumps({"status": "skipped_by_opportunity_gate"}, indent=2))
        return
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    selections = read_jsonl(OUTPUTS / "nested_selector/v3_nested_per_query.jsonl")
    action_map = {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "candidate_generation/v3_candidate_actions.jsonl")}
    source = load_source_examples()
    output_path = OUTPUTS / "multi_reader/unifiedqa_predictions.jsonl"
    done: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        done = {(str(row["query_id"]), str(row["method"])) for row in read_jsonl(output_path)}
    elif output_path.exists():
        output_path.unlink()

    pending: list[dict[str, Any]] = []
    for selection in selections:
        query_id = str(selection["query_id"])
        for method, action_id in [("baseline", f"{query_id}::fallback"), ("v3_selected", str(selection["action_id"]))]:
            if (query_id, method) in done:
                continue
            action = action_map[action_id]
            pending.append({"query_id": query_id, "method": method, "action_id": action_id, "action": action, "prompt": make_prompt(str(action["question"]), action["context_docs"])})

    runtime_device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype).to(runtime_device)
    model.eval()
    write_json(OUTPUTS / "multi_reader/unifiedqa_environment.json", {
        "reader_id": "allenai/unifiedqa-v2-t5-large-1363200",
        "model_path": args.model_path,
        "model_revision": READER_REVISION,
        "tokenizer_revision": READER_REVISION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompt_format": "question newline bracketed-title context",
        "frozen_contexts": True,
        "selector_retrained": False,
    })
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        encoded = tokenizer([row["prompt"] for row in batch], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime_device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        out: list[dict[str, Any]] = []
        for row, prediction in zip(batch, predictions):
            item, action = source[row["query_id"]], row["action"]
            answer_em, answer_f1 = answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
            out.append({
                "query_id": row["query_id"],
                "method": row["method"],
                "action_id": row["action_id"],
                "prediction": prediction.strip(),
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "title_recall": title_recall,
                "title_f1": title_f1,
                "answer_title_product": answer_f1 * title_f1,
                "context_titles": action["context_titles"],
            })
        write_jsonl(output_path, out, mode="a")

    rows = read_jsonl(output_path)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    metrics = ["answer_f1", "title_recall", "title_f1", "answer_title_product"]
    unified = {method: {metric: mean(row[metric] for row in values) for metric in metrics} | {"n": len(values)} for method, values in by_method.items()}
    baseline_map = {row["query_id"]: row for row in by_method["baseline"]}
    selected_map = {row["query_id"]: row for row in by_method["v3_selected"]}
    unified["deltas"] = {metric: mean(selected_map[query_id][metric] - baseline_map[query_id][metric] for query_id in baseline_map) for metric in metrics}
    unified["selected_answer_drop_rate"] = mean(float(selected_map[query_id]["answer_f1"] < baseline_map[query_id]["answer_f1"] - 1e-12) for query_id in baseline_map)
    unified["significance"] = {metric: paired_bootstrap([selected_map[query_id][metric] - baseline_map[query_id][metric] for query_id in baseline_map], seed=SEED + index) for index, metric in enumerate(metrics)}

    flan = read_json(OUTPUTS / "nested_selector/v3_nested_summary.json")
    summary = {
        "status": "complete",
        "contexts_frozen": True,
        "selector_retrained": False,
        "readers": {
            "google/flan-t5-large": {
                "source": "nested_selector/v3_nested_summary.json",
                "deltas": flan["deltas"],
                "selected_answer_drop_rate": flan["selected_answer_drop_rate"],
            },
            "allenai/unifiedqa-v2-t5-large-1363200": unified,
        },
    }
    flan_product = float(flan["deltas"]["answer_title_product"])
    unified_product = float(unified["deltas"]["answer_title_product"])
    flan_evidence = float(flan["deltas"]["title_f1"])
    unified_evidence = float(unified["deltas"]["title_f1"])
    summary["direction_consistency"] = {
        "product": (flan_product >= 0) == (unified_product >= 0),
        "evidence": (flan_evidence >= 0) == (unified_evidence >= 0),
        "no_large_systematic_answer_degradation": float(unified["deltas"]["answer_f1"]) >= -0.01,
    }
    write_json(OUTPUTS / "multi_reader/multi_reader_summary.json", summary)
    table_rows = [
        ["google/flan-t5-large", f"{flan['deltas']['answer_f1']:+.4f}", f"{flan['deltas']['title_f1']:+.4f}", f"{flan['deltas']['answer_title_product']:+.4f}", f"{flan['selected_answer_drop_rate']:.3f}"],
        ["allenai/unifiedqa-v2-t5-large", f"{unified['deltas']['answer_f1']:+.4f}", f"{unified['deltas']['title_f1']:+.4f}", f"{unified['deltas']['answer_title_product']:+.4f}", f"{unified['selected_answer_drop_rate']:.3f}"],
    ]
    table = markdown_table(["Reader", "Answer F1 delta", "Title F1 delta", "Product delta", "Answer-drop rate"], table_rows)
    (OUTPUTS / "tables/multi_reader_table.md").write_text("# Multi-Reader Frozen-Context Replay\n\n" + table + "\n", encoding="utf-8")
    report = f"""# Multi-Reader Report

The v3 selector and all outer-test contexts were frozen before the second-reader run. No action model was retrained for UnifiedQA.

{table}

Evidence direction consistent: **{summary['direction_consistency']['evidence']}**. Product direction consistent: **{summary['direction_consistency']['product']}**. No large systematic answer degradation on the second reader: **{summary['direction_consistency']['no_large_systematic_answer_degradation']}**.
"""
    (REPORTS / "multi_reader_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["direction_consistency"], indent=2))


if __name__ == "__main__":
    main()
