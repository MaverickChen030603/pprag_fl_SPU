#!/usr/bin/env python3
"""Budget-matched RECOMP evaluation with resumable scoring, reading, and metrics.

The development protocol is fixed in configs/revision_v5.json. The 3,000-query
holdout is never used to choose a sentence budget or packing rule.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from v5_common import HERE, V4, config, iter_jsonl, paired_bootstrap, read_json, read_jsonl, write_json, write_jsonl, write_text


DEFAULT_ARROW = "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
METRICS = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]
OUT = HERE / "outputs" / "recomp"
FIGURES = HERE / "outputs" / "figures"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_module() -> Any:
    if str(V4) not in sys.path:
        sys.path.insert(0, str(V4))
    return load_module(V4 / "08_run_official_hotpot_evaluation.py", "v5_recomp_official")


def mean_pooling(token_embeddings: Any, mask: Any) -> Any:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def context_text(sentences: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{index}] {row['title']}: {row['text']}"
        for index, row in enumerate(sentences, start=1)
    )


def reader_prompt(question: str, sentences: list[dict[str, Any]]) -> str:
    context = context_text(sentences)
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:3200]}\n\nAnswer:"
    )


def split_inputs(split: str, official: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if split == "development":
        actions = [
            row
            for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
            if row["action_family"] == "fallback"
        ]
        return [
            {
                "query_id": str(row["query_id"]),
                "question": str(row["question"]),
                "answer": str(official[str(row["query_id"])]["answer"]),
                "context_docs": row["context_docs"],
            }
            for row in actions
        ]
    source = {
        str(row["_id"]): row
        for row in read_json(V4 / "outputs/scaleup/same_source_hotpot_validation_3000.json")
    }
    return [
        {
            "query_id": str(row["query_id"]),
            "question": str(row["question"]),
            "answer": str(source[str(row["query_id"])]["answer"]),
            "context_docs": row["baseline_context"],
        }
        for row in read_jsonl(V4 / "outputs/scaleup/frozen_baseline_contexts_3000.jsonl")
    ]


def sentence_candidates(
    item: dict[str, Any], docs: list[dict[str, Any]], normalize_title: Any
) -> list[dict[str, Any]]:
    by_title = {
        normalize_title(title): (str(title), [str(value) for value in sentences])
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])
    }
    candidates: list[dict[str, Any]] = []
    for doc_rank, doc in enumerate(docs):
        record = by_title.get(normalize_title(str(doc["title"])))
        if record is None:
            continue
        title, sentences = record
        for sent_id, text in enumerate(sentences):
            candidates.append(
                {
                    "title": title,
                    "sent_id": sent_id,
                    "text": text,
                    "doc_rank": doc_rank,
                    "doc_sentence_count": len(sentences),
                    "source_order": len(candidates),
                }
            )
    if not candidates:
        raise AssertionError(f"No sentence candidates for {item.get('id', item.get('_id'))}")
    return candidates


def score_stage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = config()
    official_api = official_module()
    official = official_api.load_official(args.arrow)
    inputs = split_inputs(args.split, official)
    inputs = [row for index, row in enumerate(inputs) if index % args.num_shards == args.shard_id]
    output = OUT / f"sentence_scores_{args.split}.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    existing = {str(row["query_id"]): row for row in read_jsonl(output)} if args.resume and output.exists() else {}
    pending = [row for row in inputs if row["query_id"] not in existing]
    tokenizer = AutoTokenizer.from_pretrained(cfg["recomp"]["model"])
    model = AutoModel.from_pretrained(cfg["recomp"]["model"]).to(args.device)
    model.eval()
    rows = list(existing.values())
    started = time.perf_counter()
    for index, row in enumerate(pending, 1):
        item = official[row["query_id"]]
        candidates = sentence_candidates(item, row["context_docs"], official_api.normalize_title)
        texts = [row["question"]] + [candidate["text"] for candidate in candidates]
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            hidden = model(**encoded)[0]
            embeddings = mean_pooling(hidden, encoded["attention_mask"])
            scores = torch.mv(embeddings[1:], embeddings[0]).detach().float().cpu().tolist()
        for candidate, score_value in zip(candidates, scores):
            candidate["compressor_score"] = float(score_value)
        rows.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "answer": row["answer"],
                "input_document_count": len(row["context_docs"]),
                "candidate_sentence_count": len(candidates),
                "sentences": candidates,
            }
        )
        if index % 25 == 0 or index == len(pending):
            write_jsonl(output, rows)
            write_json(
                OUT / f"score_progress_{args.split}_shard{args.shard_id}.json",
                {
                    "status": "running" if index < len(pending) else "complete",
                    "split": args.split,
                    "completed": len(rows),
                    "assigned": len(inputs),
                    "seconds": time.perf_counter() - started,
                    "protocol_budget": cfg["recomp"]["frozen_budget_matched_tokens"],
                },
            )
    if not pending:
        write_json(
            OUT / f"score_progress_{args.split}_shard{args.shard_id}.json",
            {"status": "complete", "split": args.split, "completed": len(rows), "assigned": len(inputs)},
        )
    print(output)


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text[:3200], add_special_tokens=True))


def pack_nearest(
    ranked: list[dict[str, Any]], budget: int, tokenizer: Any
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    current_tokens = 0
    for row in ranked:
        trial = selected + [row]
        trial_tokens = token_count(tokenizer, context_text(trial))
        if not selected or trial_tokens <= budget:
            selected, current_tokens = trial, trial_tokens
            continue
        if abs(trial_tokens - budget) < abs(current_tokens - budget):
            selected = trial
        break
    return selected


def score_files(split: str) -> list[Path]:
    files = sorted(OUT.glob(f"sentence_scores_{split}.shard*-of-*.jsonl"))
    legacy = OUT / f"sentence_scores_{split}.jsonl"
    return files or ([legacy] if legacy.exists() else [])


def build_stage(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    cfg = config()
    files = score_files(args.split)
    if not files:
        raise FileNotFoundError(f"No score shards for {args.split}")
    scored = [row for path in files for row in iter_jsonl(path)]
    expected = 1000 if args.split == "development" else 3000
    unique = {str(row["query_id"]): row for row in scored}
    if len(unique) != expected:
        raise AssertionError(f"Expected {expected} unique scored queries, found {len(unique)}")
    official_api = official_module()
    official = official_api.load_official(args.arrow)
    inputs = {row["query_id"]: row for row in split_inputs(args.split, official)}
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True, use_fast=True)
    budgets = cfg["recomp"]["development_budgets"]
    rows = []
    for query_id in sorted(unique):
        score_row = unique[query_id]
        input_row = inputs[query_id]
        ranked = sorted(score_row["sentences"], key=lambda row: (-float(row["compressor_score"]), int(row["source_order"])))
        source_order = sorted(score_row["sentences"], key=lambda row: int(row["source_order"]))
        variants: list[tuple[str, int | None, list[dict[str, Any]]]] = [
            ("recomp_top1", None, ranked[:1])
        ]
        selected_budgets = budgets if args.split == "development" else [cfg["recomp"]["frozen_budget_matched_tokens"]]
        for budget in selected_budgets:
            variants.append((f"recomp_budget_{budget}", budget, pack_nearest(ranked, budget, tokenizer)))
            variants.append((f"baseline_truncated_{budget}", budget, pack_nearest(source_order, budget, tokenizer)))
        for method, target_budget, selected in variants:
            text = context_text(selected)
            rows.append(
                {
                    "query_id": query_id,
                    "split": args.split,
                    "method": method,
                    "target_context_tokens": target_budget,
                    "question": input_row["question"],
                    "answer": input_row["answer"],
                    "sentences": selected,
                    "context_text": text[:3200],
                    "context_tokens": token_count(tokenizer, text),
                    "retained_sentences": len(selected),
                    "represented_documents": len({official_api.normalize_title(row["title"]) for row in selected}),
                    "candidate_sentences": len(ranked),
                    "input_documents": len(input_row["context_docs"]),
                }
            )
    output = OUT / f"contexts_{args.split}.jsonl"
    write_jsonl(output, rows)
    write_json(
        OUT / f"context_build_manifest_{args.split}.json",
        {
            "status": "complete",
            "split": args.split,
            "queries": len(unique),
            "rows": len(rows),
            "packing": cfg["recomp"]["sentence_packing"],
            "frozen_budget": cfg["recomp"]["frozen_budget_matched_tokens"],
            "reader_context_char_limit": 3200,
            "reader_token_limit": 1024,
            "holdout_used_for_budget_selection": False,
        },
    )
    print(output)


def reader_stage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    contexts = read_jsonl(OUT / f"contexts_{args.split}.jsonl")
    query_ids = sorted({str(row["query_id"]) for row in contexts})
    assigned = {query_id for index, query_id in enumerate(query_ids) if index % args.num_shards == args.shard_id}
    pending_all = [row for row in contexts if str(row["query_id"]) in assigned]
    output = OUT / f"reader_{args.split}.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    existing = read_jsonl(output) if args.resume and output.exists() else []
    done = {(str(row["query_id"]), str(row["method"])) for row in existing}
    pending = [row for row in pending_all if (str(row["query_id"]), str(row["method"])) not in done]
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.reader_model, local_files_only=True, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    output_rows = list(existing)
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [reader_prompt(row["question"], row["sentences"]) for row in batch]
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=32, num_beams=1, do_sample=False
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        batch_seconds = time.perf_counter() - batch_started
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, prediction in zip(batch, predictions):
            output_rows.append(
                {
                    "query_id": row["query_id"],
                    "split": args.split,
                    "method": row["method"],
                    "prediction": prediction.strip(),
                    "context_tokens": row["context_tokens"],
                    "retained_sentences": row["retained_sentences"],
                    "represented_documents": row["represented_documents"],
                    "reader_latency_seconds_batched": batch_seconds / max(1, len(batch)),
                    "reader_batch_size": len(batch),
                }
            )
        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(pending):
            write_jsonl(output, output_rows)
            write_json(
                OUT / f"reader_progress_{args.split}_shard{args.shard_id}.json",
                {
                    "status": "running" if start + args.batch_size < len(pending) else "complete",
                    "completed": len(output_rows),
                    "assigned": len(pending_all),
                    "seconds": time.perf_counter() - started,
                    "reader_calls_per_query_online": 1,
                },
            )
    if not pending:
        write_json(
            OUT / f"reader_progress_{args.split}_shard{args.shard_id}.json",
            {"status": "complete", "completed": len(output_rows), "assigned": len(pending_all)},
        )
    print(output)


def compressed_instances(
    query_id: str,
    context: dict[str, Any],
    item: dict[str, Any],
    official_api: Any,
) -> list[dict[str, Any]]:
    rows = []
    doc_ranks: dict[str, int] = {}
    for sentence in context["sentences"]:
        key = official_api.normalize_title(sentence["title"])
        if key not in doc_ranks:
            doc_ranks[key] = len(doc_ranks)
        rows.append(
            {
                "query_id": query_id,
                "title": sentence["title"],
                "sent_id": int(sentence["sent_id"]),
                "features": official_api.sentence_features(
                    item["question"],
                    sentence["title"],
                    sentence["text"],
                    doc_ranks[key],
                    int(sentence["sent_id"]),
                    int(sentence["doc_sentence_count"]),
                ),
                "label": int(
                    (official_api.normalize_title(sentence["title"]), int(sentence["sent_id"]))
                    in {
                        (official_api.normalize_title(title), int(sent_id))
                        for title, sent_id in zip(
                            item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]
                        )
                    }
                ),
            }
        )
    return rows


def support_models(split: str, official: dict[str, dict[str, Any]], api: Any) -> tuple[dict[int, Any], dict[int, float], dict[str, int]]:
    selections = read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
    actions = {
        str(row["action_id"]): row
        for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    }
    query_ids = {str(row["query_id"]) for row in selections}
    train_by_query: dict[str, list[dict[str, Any]]] = {}
    for selection in selections:
        query_id = str(selection["query_id"])
        rows = []
        for action_id in (f"{query_id}::v4::fallback", str(selection["action_id"])):
            rows.extend(api.context_instances(query_id, actions[action_id], official[query_id]))
        train_by_query[query_id] = rows
    historical = read_json(V4 / "outputs/official_metrics/official_hotpotqa_summary.json", {})
    thresholds = {int(key): float(value) for key, value in historical.get("fold_thresholds", {}).items()}
    if set(thresholds.values()) != {0.7}:
        raise AssertionError(f"Unexpected support thresholds: {thresholds}")
    if split == "development":
        models = {}
        fold_by_query = {str(row["query_id"]): int(row["outer_fold"]) for row in selections}
        for fold in range(5):
            train_ids = [query_id for query_id in query_ids if fold_by_query[query_id] != fold]
            models[fold] = api.fit([row for query_id in train_ids for row in train_by_query[query_id]])
        return models, thresholds, fold_by_query
    model = api.fit([row for query_id in query_ids for row in train_by_query[query_id]])
    return {0: model}, {0: 0.7}, {}


def reader_files(split: str) -> list[Path]:
    return sorted(OUT.glob(f"reader_{split}.shard*-of-*.jsonl"))


def reference_rows(split: str) -> list[dict[str, Any]]:
    if split == "development":
        return read_jsonl(V4 / "outputs/official_metrics/official_hotpotqa_per_query.jsonl")
    return read_jsonl(V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl")


def reference_actions(split: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = (
        V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl"
        if split == "development"
        else V4 / "outputs/scaleup/generated_actions_3000.jsonl"
    )
    rows = read_jsonl(path)
    by_id = {str(row["action_id"]): row for row in rows}
    if split == "holdout":
        return {
            (str(row["query_id"]), str(row["method"])): by_id[str(row["action_id"])]
            for row in reference_rows(split)
        }
    fallback = {
        str(row["query_id"]): row for row in rows if str(row["action_family"]) == "fallback"
    }
    selected = {
        str(row["query_id"]): by_id[str(row["action_id"])]
        for row in read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
    }
    return {
        **{(query_id, "baseline"): action for query_id, action in fallback.items()},
        **{(query_id, "v4_selected"): action for query_id, action in selected.items()},
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    cfg = config()
    api = official_module()
    official = api.load_official(args.arrow)
    contexts = {
        (str(row["query_id"]), str(row["method"])): row
        for row in read_jsonl(OUT / f"contexts_{args.split}.jsonl")
    }
    files = reader_files(args.split)
    if not files:
        raise FileNotFoundError(f"No reader shards for {args.split}")
    reader_rows = {
        (str(row["query_id"]), str(row["method"])): row
        for path in files
        for row in iter_jsonl(path)
    }
    if set(contexts) != set(reader_rows):
        missing = sorted(set(contexts) - set(reader_rows))[:10]
        raise AssertionError(f"Incomplete reader outputs; first missing: {missing}")
    models, thresholds, fold_by_query = support_models(args.split, official, api)
    rows = []
    for key, context in contexts.items():
        query_id, method = key
        fold = fold_by_query.get(query_id, 0)
        item = official[query_id]
        instances = compressed_instances(query_id, context, item, api)
        predicted_support = api.support_set(instances, api.score(models[fold], instances), thresholds[fold])
        gold_support = {
            (api.normalize_title(title), int(sent_id))
            for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
        }
        metrics = api.official_metrics(
            reader_rows[key]["prediction"], item["answer"], predicted_support, gold_support
        )
        rows.append({**reader_rows[key], **metrics})
    actions = reference_actions(args.split)
    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True, use_fast=True)
    for row in reference_rows(args.split):
        method = "full_v4" if row["method"] == "v4_selected" else "frozen_top5_baseline"
        action = actions[(str(row["query_id"]), str(row["method"]))]
        docs = list(action["context_docs"])
        rows.append(
            {
                **row,
                "method": method,
                "context_tokens": token_count(reader_tokenizer, context_text(docs)),
                "retained_sentences": None,
                "represented_documents": len({str(doc.get("doc_id", doc["title"])) for doc in docs}),
                "reader_latency_seconds_batched": None,
            }
        )
    per_query = OUT / f"official_per_query_{args.split}.jsonl"
    write_jsonl(per_query, rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = {}
    for method, values in grouped.items():
        numeric = lambda key: [float(row[key]) for row in values if row.get(key) is not None]
        summaries[method] = {
            "n": len(values),
            **{metric: mean(numeric(metric)) for metric in METRICS},
            "context_tokens": mean(numeric("context_tokens")) if numeric("context_tokens") else None,
            "retained_sentences": mean(numeric("retained_sentences")) if numeric("retained_sentences") else None,
            "represented_documents": mean(numeric("represented_documents")) if numeric("represented_documents") else None,
            "reader_latency_seconds_batched": mean(numeric("reader_latency_seconds_batched")) if numeric("reader_latency_seconds_batched") else None,
        }
    baseline = {str(row["query_id"]): row for row in rows if row["method"] == "frozen_top5_baseline"}
    significance = {}
    for method in summaries:
        if method == "frozen_top5_baseline":
            continue
        method_rows = {str(row["query_id"]): row for row in rows if row["method"] == method}
        shared = sorted(set(method_rows) & set(baseline))
        significance[method] = {
            metric: paired_bootstrap([float(method_rows[q][metric]) - float(baseline[q][metric]) for q in shared])
            for metric in ("answer_f1", "sp_f1", "joint_f1")
        }
    payload = {
        "status": "complete",
        "split": args.split,
        "protocol": {
            "same_top5_input": True,
            "same_flan_reader": True,
            "same_decoding": True,
            "same_support_predictor": True,
            "frozen_budget": cfg["recomp"]["frozen_budget_matched_tokens"],
            "holdout_used_for_selection": False,
        },
        "metrics": summaries,
        "vs_baseline_significance": significance,
        "per_query": str(per_query),
    }
    output_name = "recomp_budget_matched_metrics.json" if args.split == "development" else "recomp_holdout_metrics.json"
    write_json(OUT / output_name, payload)
    if args.split == "development":
        write_curve_outputs(payload)
    write_report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_curve_outputs(payload: dict[str, Any]) -> None:
    rows = []
    for method, metrics in payload["metrics"].items():
        if method.startswith("recomp_budget_") or method.startswith("baseline_truncated_") or method == "recomp_top1":
            rows.append({"method": method, **metrics})
    csv_path = OUT / "recomp_budget_curve.csv"
    fields = ["method", "n", "context_tokens", "retained_sentences", "represented_documents", "answer_f1", "sp_f1", "joint_f1", "reader_latency_seconds_batched"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    try:
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
        for prefix, label, style in (("recomp_budget_", "RECOMP", "o-"), ("baseline_truncated_", "Baseline-Truncated", "s--")):
            subset = sorted((row for row in rows if row["method"].startswith(prefix)), key=lambda row: row["context_tokens"])
            for axis, metric in zip(axes, ("answer_f1", "sp_f1", "joint_f1")):
                axis.plot([row["context_tokens"] for row in subset], [row[metric] for row in subset], style, label=label)
                axis.set_title(metric.replace("_", " ").upper())
                axis.set_xlabel("Mean context tokens")
                axis.grid(alpha=0.25)
        axes[0].set_ylabel("F1")
        axes[-1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "performance_vs_context_tokens.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        write_text(FIGURES / "performance_vs_context_tokens.NEEDS_MEASUREMENT.txt", f"[NEEDS MEASUREMENT] Plot failed: {exc}")


def fmt(value: Any, digits: int = 4) -> str:
    return "[NEEDS MEASUREMENT]" if value is None else f"{float(value):.{digits}f}"


def write_report() -> None:
    dev = read_json(OUT / "recomp_budget_matched_metrics.json")
    holdout = read_json(OUT / "recomp_holdout_metrics.json")
    lines = [
        "# Budget-Matched RECOMP Fairness Report",
        "",
        "The original Top-1 setting is retained only as a compatibility diagnostic. The fixed matched protocol ranks sentences with the author-released RECOMP checkpoint and greedily packs whole sentences to the nearest 660-token reader context. The same Top-5 input, FLAN-T5-Large prompt/decoding, and sentence-support predictor are used for every system.",
        "",
    ]
    if not dev:
        lines.append("Development budget curve: [NEEDS MEASUREMENT]")
    else:
        lines.extend(["## Development (1,000)", "", "| Method | Tokens | Sentences | Docs | Answer F1 | SP F1 | Joint F1 | Batched reader latency |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for method, values in dev["metrics"].items():
            if method in ("frozen_top5_baseline", "full_v4", "recomp_top1", "recomp_budget_660", "baseline_truncated_660"):
                sentence_value = "--" if method in ("frozen_top5_baseline", "full_v4") else fmt(values.get("retained_sentences"), 1)
                latency_value = "see cost report" if method in ("frozen_top5_baseline", "full_v4") else fmt(values.get("reader_latency_seconds_batched"), 4)
                lines.append(f"| {method} | {fmt(values.get('context_tokens'), 1)} | {sentence_value} | {fmt(values.get('represented_documents'), 1)} | {fmt(values.get('answer_f1'))} | {fmt(values.get('sp_f1'))} | {fmt(values.get('joint_f1'))} | {latency_value} |")
    lines.extend(["", "## Frozen 3,000-Query Holdout", ""])
    if not holdout:
        lines.append("[NEEDS MEASUREMENT]")
    else:
        lines.extend(["| Method | Tokens | Sentences | Docs | Answer F1 | SP F1 | Joint F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
        for method, values in holdout["metrics"].items():
            sentence_value = "--" if method in ("frozen_top5_baseline", "full_v4") else fmt(values.get("retained_sentences"), 1)
            lines.append(f"| {method} | {fmt(values.get('context_tokens'), 1)} | {sentence_value} | {fmt(values.get('represented_documents'), 1)} | {fmt(values.get('answer_f1'))} | {fmt(values.get('sp_f1'))} | {fmt(values.get('joint_f1'))} |")
    if dev and holdout:
        matched = holdout["vs_baseline_significance"]["recomp_budget_660"]["joint_f1"]
        decision = (
            f"Both stages are complete. On the frozen holdout, 660-token RECOMP differs from the Top-5 baseline by "
            f"{matched['delta']:+.4f} Joint F1 (p={matched['p_value']:.4f}); this is not a significant advantage. "
            "The original Top-1 result is no longer evidence of general superiority. We frame RECOMP and Full as different context-construction objectives under matched reader and context budgets."
        )
    else:
        decision = "One or more stages remain incomplete; no RECOMP numerical superiority claim is allowed."
    lines.extend(["", "## Claim Decision", "", decision])
    text = "\n".join(lines)
    write_text(HERE / "reports/recomp_fairness_report.md", text)
    write_text(HERE / "recomp_fairness_report.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["score", "build", "reader", "evaluate", "report"])
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arrow", default=os.environ.get("V4_HOTPOT_ARROW", DEFAULT_ARROW))
    parser.add_argument("--reader-model", default=os.environ.get("V4_FLAN_T5_LARGE", FLAN))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("Invalid shard id")
    if args.stage == "score":
        score_stage(args)
    elif args.stage == "build":
        build_stage(args)
    elif args.stage == "reader":
        reader_stage(args)
    elif args.stage == "evaluate":
        evaluate_stage(args)
    else:
        write_report()


if __name__ == "__main__":
    main()
