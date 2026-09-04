#!/usr/bin/env python3
"""Independent CrossEncoder-Top5 post-hoc secondary baseline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import joblib

from sigirap_common import (
    FIGURES,
    FLAN,
    HERE,
    METRICS,
    OUTPUTS,
    REPORTS,
    SPLITS,
    TABLES,
    ensure_layout,
    flan_prompt,
    iter_jsonl,
    load_module,
    metric_means,
    paired_bootstrap,
    read_json,
    read_jsonl,
    source_rows,
    stable_shard,
    token_count_approx,
    write_json,
    write_jsonl,
)


RERANKER_DIR = OUTPUTS / "reranker"
VARIANTS = ("ce_score_order", "ce_baseline_stable")


def action_path(split: str) -> Path:
    return RERANKER_DIR / f"ce_actions_{split}.jsonl"


def reader_path(split: str, shard_index: int, num_shards: int) -> Path:
    return RERANKER_DIR / f"ce_reader_{split}.shard{shard_index:02d}-of-{num_shards:02d}.jsonl"


def doc_rank(doc: dict[str, Any]) -> tuple[int, str]:
    return int(doc.get("source_rank", 10**6)), str(doc["doc_id"])


def build_actions(args: argparse.Namespace) -> None:
    splits = (args.split,) if args.split != "all" else tuple(SPLITS)
    manifest: dict[str, Any] = {
        "status": "complete",
        "method": "independent CrossEncoder-Top5",
        "variants": list(VARIANTS),
        "uses_pair_features": False,
        "uses_missing_hop_features": False,
        "uses_selector_probabilities": False,
        "uses_gold_or_reader_outcomes": False,
        "document_budget": 5,
        "candidate_pool": "same frozen approximately Top-10 pool",
        "splits": {},
    }
    for split in splits:
        started = time.perf_counter()
        cache = joblib.load(SPLITS[split]["cache"])
        rows: list[dict[str, Any]] = []
        for query_id, query in cache["queries"].items():
            details = query["doc_feature_details"]
            docs = {str(doc["doc_id"]): doc for doc in query["docs"]}
            selected_ids = sorted(
                docs,
                key=lambda doc_id: (-float(details[doc_id]["cross_encoder_relevance"]), doc_rank(docs[doc_id])),
            )[:5]
            orders = {
                "ce_score_order": selected_ids,
                "ce_baseline_stable": sorted(selected_ids, key=lambda doc_id: doc_rank(docs[doc_id])),
            }
            for variant, ordered_ids in orders.items():
                context_docs = [docs[doc_id] for doc_id in ordered_ids]
                rows.append({
                    "split": split,
                    "query_id": str(query_id),
                    "question": str(query["question"]),
                    "action_id": f"{query_id}::{variant}",
                    "method": variant,
                    "context_doc_ids": ordered_ids,
                    "context_titles": [str(doc["title"]) for doc in context_docs],
                    "context_docs": context_docs,
                    "cross_encoder_scores": [
                        float(details[doc_id]["cross_encoder_relevance"]) for doc_id in ordered_ids
                    ],
                    "documents_retained": len(context_docs),
                    "context_tokens_approx": token_count_approx(context_docs),
                    "cross_encoder_calls": len(docs),
                    "reader_calls": 1,
                })
        write_jsonl(action_path(split), rows)
        manifest["splits"][split] = {
            "n_queries": len(cache["queries"]),
            "n_actions": len(rows),
            "build_seconds": time.perf_counter() - started,
            "cross_encoder_checkpoint": str(cache["cross_encoder_path"]),
        }
    write_json(RERANKER_DIR / "ce_action_build_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def official_source(split: str) -> tuple[Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    oracle = load_module(HERE / "01_oracle_action_set_diagnostic.py", "sigirap_oracle_for_ce")
    api, official = oracle.load_official()
    return api, official, source_rows(split, official)


def run_reader(args: argparse.Namespace) -> None:
    api, _, source = official_source(args.split)
    output = reader_path(args.split, args.shard_index, args.num_shards)
    done = {str(row["action_id"]) for row in read_jsonl(output)} if args.resume else set()
    if output.exists() and not args.resume:
        output.unlink()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(args.model_path or FLAN)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16
    ).to(torch.device(args.device))
    model.eval()
    manifest = {
        "status": "running",
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": args.device,
        "batch_size": args.batch_size,
        "model_path": str(model_path),
        "prompt_frozen": True,
        "context_character_cap": 3200,
        "generation": {"max_new_tokens": 32, "num_beams": 1, "do_sample": False},
        "started_at_epoch": time.time(),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)

    batch: list[dict[str, Any]] = []
    completed = len(done)

    def flush(values: list[dict[str, Any]]) -> int:
        if not values:
            return 0
        prompts = [flan_prompt(source[row["query_id"]]["question"], row["context_docs"]) for row in values]
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(torch.device(args.device))
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows = []
        for action, prediction in zip(values, predictions):
            gold_answer = str(source[action["query_id"]]["answer"])
            answer = api.answer_metrics(prediction.strip(), gold_answer)
            rows.append({
                "split": args.split,
                "query_id": action["query_id"],
                "action_id": action["action_id"],
                "method": action["method"],
                "prediction": prediction.strip(),
                "answer_em": float(answer["em"]),
                "answer_f1": float(answer["f1"]),
            })
        write_jsonl(output, rows, mode="a")
        return len(rows)

    for action in iter_jsonl(action_path(args.split)):
        query_id = str(action["query_id"])
        if stable_shard(query_id, args.num_shards) != args.shard_index:
            continue
        if str(action["action_id"]) in done:
            continue
        batch.append(action)
        if len(batch) >= args.batch_size:
            completed += flush(batch)
            batch = []
    completed += flush(batch)
    manifest.update({"status": "complete", "completed": completed})
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_reader(split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(RERANKER_DIR.glob(f"ce_reader_{split}.shard*-of-*.jsonl")):
        for row in iter_jsonl(path):
            rows[str(row["action_id"])] = row
    expected = sum(1 for _ in iter_jsonl(action_path(split)))
    if len(rows) != expected:
        raise AssertionError(f"Incomplete CE reader output for {split}: {len(rows)} != {expected}")
    return rows


def existing_primary_rows(split: str) -> list[dict[str, Any]]:
    if split == "development1000":
        path = SPLITS[split]["actions"].parent.parent / "official_metrics/official_hotpotqa_per_query.jsonl"
        mapping = {"baseline": "baseline", "v4_selected": "full"}
    elif split == "holdout3000":
        path = SPLITS[split]["actions"].parent / "official_metrics/flan_per_query.jsonl"
        mapping = {"baseline": "baseline", "v4_selected": "full"}
    else:
        path = SPLITS[split]["actions"].parent / "official_per_query_3405.jsonl"
        mapping = {"frozen_top5_baseline": "baseline", "full_v4": "full"}
    rows = []
    for row in iter_jsonl(path):
        if str(row["method"]) in mapping:
            rows.append({**row, "method": mapping[str(row["method"])]})
    return rows


def score_ce_actions(
    split: str,
    api: Any,
    official: dict[str, dict[str, Any]],
    fold_models: dict[int, Any],
    global_model: Any,
) -> list[dict[str, Any]]:
    readers = load_reader(split)
    selections = {str(row["query_id"]): row for row in iter_jsonl(SPLITS[split]["selections"])}
    rows: list[dict[str, Any]] = []
    for action in iter_jsonl(action_path(split)):
        query_id = str(action["query_id"])
        gold = official[query_id]
        instances = api.context_instances(query_id, action, gold)
        if split == "development1000":
            fold = int(selections[query_id]["outer_fold"])
            support_model = fold_models[fold]
        else:
            support_model = global_model
        predicted_support = api.support_set(instances, api.score(support_model, instances), 0.7)
        gold_support = {
            (api.normalize_title(title), int(sentence_id))
            for title, sentence_id in zip(gold["supporting_facts"]["title"], gold["supporting_facts"]["sent_id"])
        }
        outcome = readers[str(action["action_id"])]
        metrics = api.official_metrics(outcome["prediction"], gold["answer"], predicted_support, gold_support)
        rows.append({
            "split": split,
            "query_id": query_id,
            "method": action["method"],
            "action_id": action["action_id"],
            "prediction": outcome["prediction"],
            "documents_retained": action["documents_retained"],
            "context_tokens": action["context_tokens_approx"],
            "reader_calls": 1,
            "cross_encoder_calls": action["cross_encoder_calls"],
            **{metric: float(metrics[metric]) for metric in METRICS},
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh_report(payload: dict[str, Any]) -> None:
    chosen = payload["development_variant_choice"]["chosen_variant"]
    latency_path = RERANKER_DIR / "ce_reranker_latency.json"
    latency = read_json(latency_path) if latency_path.exists() else {
        "status": "pending_direct_measurement",
        "baseline_mean_ms": 140.8817,
        "full_mean_ms": 213.4842,
    }
    table = [
        "# Independent Relevance Reranker Comparison",
        "",
        f"Development selected `{chosen}`. This choice was frozen before reading either holdout result.",
        "",
        "| Split | System | Answer F1 | SP F1 | Joint F1 | Delta Joint vs baseline | 95% CI | Paired p | Docs | Context tokens | CE calls | Reader calls | End-to-end ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, result in payload["splits"].items():
        for method in ("baseline", chosen, "full"):
            values = result["metrics"][method]
            comparison = result["vs_baseline"][method] if method != "baseline" else None
            ci = "reference" if comparison is None else f"[{comparison['joint_f1']['ci95_low']:+.4f}, {comparison['joint_f1']['ci95_high']:+.4f}]"
            delta = 0.0 if comparison is None else comparison["joint_f1"]["mean"]
            p_value = "--" if comparison is None else f"{comparison['joint_f1']['p_value']:.4f}"
            resource = result["resources"][method]
            if method == "baseline":
                end_to_end = f"{latency.get('baseline_mean_ms', 140.8817):.2f}"
            elif method == "full":
                end_to_end = f"{latency.get('full_mean_ms', 213.4842):.2f}"
            elif latency.get("status") == "complete":
                end_to_end = f"{latency['end_to_end_ms']['mean']:.2f}"
            else:
                end_to_end = "--"
            table.append(
                f"| {SPLITS[split]['label']} | {method} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | "
                f"{values['joint_f1']:.4f} | {delta:+.4f} | {ci} | {p_value} | "
                f"{resource['documents_retained']:.2f} | {resource['context_tokens']:.1f} | "
                f"{resource['cross_encoder_calls']:.1f} | {resource['reader_calls']:.1f} | {end_to_end} |"
            )
    (TABLES / "independent_reranker_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    report = [
        "# Strong Independent Reranker Baseline",
        "",
        "## Protocol",
        "",
        "CrossEncoder-Top5 independently scores each document in the same frozen approximately Top-10 candidate pool and retains five documents. It uses the same frozen cross-encoder checkpoint, 3,200-character cap, FLAN-T5-Large prompt and decoding, sentence-support predictor, and official metrics as Full. It excludes pair features, missing-hop features, outcome models, selector probabilities, gold labels, and reader outcomes at inference.",
        "",
        f"The ordering variant `{chosen}` was chosen using the 1,000-query nested development split only. Both 3,000- and 3,405-query results are frozen evaluations of that choice. Because this baseline was added after the primary study, it is a **post-hoc secondary baseline analysis**, not a pre-specified confirmatory comparison.",
        "",
        "## Results",
        "",
    ]
    for split, result in payload["splits"].items():
        baseline = result["metrics"]["baseline"]["joint_f1"]
        ce = result["metrics"][chosen]["joint_f1"]
        full = result["metrics"]["full"]["joint_f1"]
        versus_full = result["chosen_vs_full"]["joint_f1"]
        report.extend([
            f"### {SPLITS[split]['label']}",
            "",
            f"Baseline / CrossEncoder / Full Joint F1: {baseline:.4f} / {ce:.4f} / {full:.4f}. CrossEncoder minus Full is {versus_full['mean']:+.4f} (95% CI [{versus_full['ci95_low']:+.4f}, {versus_full['ci95_high']:+.4f}], paired p={versus_full['p_value']:.4f}).",
            "",
        ])
    if latency.get("status") == "complete":
        latency_text = (
            f"Direct same-machine CrossEncoder-Top5 mean/median/P95 latency is "
            f"{latency['end_to_end_ms']['mean']:.2f}/{latency['end_to_end_ms']['median']:.2f}/"
            f"{latency['end_to_end_ms']['p95']:.2f} ms/query; CrossEncoder scoring alone averages "
            f"{latency['cross_encoder_ms']['mean']:.2f} ms/query. The direct scoring path reproduces "
            f"{latency['context_match_rate_against_cached_selection']:.1%} of cached Top-5 contexts."
        )
    else:
        latency_text = "Direct CrossEncoder-Top5 latency remains pending; cached action-build time is not substituted."
    report.extend([
        "## Cost boundary",
        "",
        f"Latency artifact status: `{latency.get('status')}`. Frozen Top-5 and Full remain fixed at {latency.get('baseline_mean_ms', 140.8817):.2f} and {latency.get('full_mean_ms', 213.4842):.2f} ms/query. {latency_text}",
        "",
        "## Claim boundary",
        "",
        "This analysis asks whether independent document relevance can recover the pair-complementary result under one frozen pool and reader. It does not establish universal superiority over neural reranking or over other retrievers.",
    ])
    (REPORTS / "strong_baseline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def finalize(_: argparse.Namespace) -> None:
    oracle = load_module(HERE / "01_oracle_action_set_diagnostic.py", "sigirap_oracle_for_ce_finalize")
    api, official = oracle.load_official()
    fold_models, global_model = oracle.build_support_models(api, official)
    all_rows: list[dict[str, Any]] = []
    split_ce: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = score_ce_actions(split, api, official, fold_models, global_model)
        split_ce[split] = rows
        all_rows.extend(rows)

    development_means = {
        variant: metric_means(row for row in split_ce["development1000"] if row["method"] == variant)
        for variant in VARIANTS
    }
    chosen = max(VARIANTS, key=lambda variant: (development_means[variant]["joint_f1"], variant == "ce_score_order"))
    payload: dict[str, Any] = {
        "status": "complete",
        "analysis_label": "post-hoc secondary baseline analysis",
        "development_variant_choice": {
            "chosen_variant": chosen,
            "criterion": "highest development official Joint F1; score-order is the predeclared tie break",
            "development_metrics": development_means,
            "holdout_outcomes_used": False,
        },
        "fairness": {
            "same_candidate_pool": True,
            "same_document_budget": 5,
            "same_character_cap": 3200,
            "same_flan_reader_prompt_and_decoding": True,
            "same_support_predictor": True,
            "same_official_metrics": True,
            "pair_features_used": False,
            "gold_or_reader_outcomes_used_at_inference": False,
        },
        "splits": {},
    }
    per_query_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        primary = existing_primary_rows(split)
        combined = primary + split_ce[split]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in combined:
            grouped[str(row["method"])].append(row)
            by_query[str(row["query_id"])][str(row["method"])] = row
            per_query_rows.append({"split": split, **row})
        metrics = {method: metric_means(rows) for method, rows in grouped.items()}
        vs_baseline: dict[str, dict[str, Any]] = {}
        for method in ("full",) + VARIANTS:
            vs_baseline[method] = {
                metric: paired_bootstrap([
                    values[method][metric] - values["baseline"][metric] for values in by_query.values()
                ])
                for metric in ("answer_f1", "sp_f1", "joint_f1")
            }
        chosen_vs_full = {
            metric: paired_bootstrap([
                values[chosen][metric] - values["full"][metric] for values in by_query.values()
            ])
            for metric in ("answer_f1", "sp_f1", "joint_f1")
        }
        resources = {}
        for method, rows in grouped.items():
            if method in VARIANTS:
                resources[method] = {
                    "documents_retained": mean(float(row["documents_retained"]) for row in rows),
                    "context_tokens": mean(float(row["context_tokens"]) for row in rows),
                    "reader_calls": 1.0,
                    "cross_encoder_calls": mean(float(row["cross_encoder_calls"]) for row in rows),
                }
            elif method == "baseline":
                resources[method] = {"documents_retained": 5.0, "context_tokens": 668.7, "reader_calls": 1.0, "cross_encoder_calls": 0.0}
            else:
                resources[method] = {"documents_retained": 5.0, "context_tokens": 662.5, "reader_calls": 1.0, "cross_encoder_calls": 10.0}
        payload["splits"][split] = {
            "n_queries": len(by_query),
            "metrics": metrics,
            "vs_baseline": vs_baseline,
            "chosen_vs_full": chosen_vs_full,
            "resources": resources,
        }
    write_json(RERANKER_DIR / "ce_reranker_metrics.json", payload)
    write_csv(RERANKER_DIR / "ce_reranker_per_query.csv", per_query_rows)
    if not (RERANKER_DIR / "ce_reranker_latency.json").exists():
        write_json(RERANKER_DIR / "ce_reranker_latency.json", {
            "status": "pending_direct_measurement",
            "baseline_mean_ms": 140.88169157900847,
            "full_mean_ms": 213.48419773590285,
            "do_not_substitute_cached_build_time": True,
        })
    refresh_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def run_latency(args: argparse.Namespace) -> None:
    payload = read_json(RERANKER_DIR / "ce_reranker_metrics.json")
    chosen = payload["development_variant_choice"]["chosen_variant"]
    cache = joblib.load(SPLITS["development1000"]["cache"])
    query_items = list(cache["queries"].items())[: args.warmup_queries + args.measured_queries]

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cross_encoder = CrossEncoder(
        str(cache["cross_encoder_path"]),
        device=args.device,
        local_files_only=True,
        max_length=512,
    )
    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_path or FLAN), local_files_only=True, use_fast=True)
    reader = AutoModelForSeq2SeqLM.from_pretrained(
        Path(args.model_path or FLAN), local_files_only=True, torch_dtype=torch.float16
    ).to(torch.device(args.device))
    reader.eval()

    total_times: list[float] = []
    ce_times: list[float] = []
    reader_times: list[float] = []
    context_matches = 0
    frozen_actions = {
        str(row["query_id"]): row for row in iter_jsonl(action_path("development1000")) if row["method"] == chosen
    }
    for index, (query_id, query) in enumerate(query_items):
        docs = list(query["docs"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Match the exact text serialization used when the frozen semantic cache
        # was built; even punctuation changes can alter CrossEncoder ordering.
        scores = cross_encoder.predict(
            [(query["question"], f"{doc['title']}. {doc['text']}") for doc in docs],
            batch_size=10,
            show_progress_bar=False,
        )
        torch.cuda.synchronize()
        after_ce = time.perf_counter()
        ranked = sorted(
            zip(docs, np.asarray(scores).tolist()),
            key=lambda pair: (-float(pair[1]), doc_rank(pair[0])),
        )[:5]
        selected = [doc for doc, _ in ranked]
        if chosen == "ce_baseline_stable":
            selected.sort(key=doc_rank)
        context_matches += int(
            [str(doc["doc_id"]) for doc in selected] == frozen_actions[str(query_id)]["context_doc_ids"]
        )
        prompt = flan_prompt(query["question"], selected)
        encoded = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(torch.device(args.device))
        with torch.inference_mode():
            reader.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if index >= args.warmup_queries:
            ce_times.append((after_ce - started) * 1000)
            reader_times.append((ended - after_ce) * 1000)
            total_times.append((ended - started) * 1000)
    latency = {
        "status": "complete",
        "variant": chosen,
        "device": args.device,
        "warmup_queries": args.warmup_queries,
        "measured_queries": args.measured_queries,
        "same_machine_and_batch_one": True,
        "model_loading_excluded": True,
        "cross_encoder_input_matches_frozen_cache": True,
        "cross_encoder_calls_per_query": 10,
        "reader_calls_per_query": 1,
        "context_match_rate_against_cached_selection": context_matches / len(query_items),
        "cross_encoder_ms": {"mean": mean(ce_times), "median": median(ce_times), "p95": percentile(ce_times, 0.95)},
        "reader_and_serialization_ms": {"mean": mean(reader_times), "median": median(reader_times), "p95": percentile(reader_times, 0.95)},
        "end_to_end_ms": {"mean": mean(total_times), "median": median(total_times), "p95": percentile(total_times, 0.95)},
        "baseline_mean_ms": 140.88169157900847,
        "full_mean_ms": 213.48419773590285,
    }
    write_json(RERANKER_DIR / "ce_reranker_latency.json", latency)
    refresh_report(payload)
    print(json.dumps(latency, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "reader", "finalize", "latency"), required=True)
    parser.add_argument("--split", choices=("all",) + tuple(SPLITS), default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--measured-queries", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    ensure_layout()
    arguments = parse_args()
    if arguments.stage == "build":
        build_actions(arguments)
    elif arguments.stage == "reader":
        if arguments.split == "all":
            raise ValueError("Reader stage requires one split")
        run_reader(arguments)
    elif arguments.stage == "finalize":
        finalize(arguments)
    else:
        run_latency(arguments)
