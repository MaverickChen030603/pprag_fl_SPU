#!/usr/bin/env python3
"""Faithfully run the author-released RECOMP HotpotQA extractive compressor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any

from completion_common import FAITHFUL, V4_ROOT, add_v4_import_path, ensure_layout, load_module, read_jsonl, write_json, write_jsonl


RECOMP_MODEL = "fangyuan/hotpotqa_extractive_compressor"
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
RECOMP_REPO = "https://github.com/carriex/recomp"
RECOMP_COMMIT = "51d4432"
METRICS = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]


def mean_pooling(token_embeddings: Any, mask: Any) -> Any:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def reader_prompt(question: str, title: str, sentence: str) -> str:
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n[1] {title}: {sentence}\n\nAnswer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressor-device", default="cuda:2")
    parser.add_argument("--reader-device", default="cuda:3")
    parser.add_argument("--reader-batch-size", type=int, default=16)
    args = parser.parse_args()
    ensure_layout()
    add_v4_import_path()
    official = load_module(V4_ROOT / "08_run_official_hotpot_evaluation.py", "v4_recomp_official")
    hotpot = official.load_official(official.DEFAULT_ARROW)
    selections = read_jsonl(V4_ROOT / "outputs/nested_selector/v4_nested_per_query.jsonl")
    query_ids = [str(row["query_id"]) for row in selections]
    baseline_actions = {
        str(row["action_id"]): row
        for row in read_jsonl(V4_ROOT / "outputs/generated_actions/v4_outer_test_actions.jsonl")
        if row["action_family"] == "fallback"
    }

    import torch
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

    compressor_tokenizer = AutoTokenizer.from_pretrained(RECOMP_MODEL)
    compressor = AutoModel.from_pretrained(RECOMP_MODEL).to(torch.device(args.compressor_device))
    compressor.eval()
    selected_sentences = []
    for index, query_id in enumerate(query_ids):
        item = hotpot[query_id]
        by_title = {
            official.normalize_title(title): (str(title), [str(value) for value in sentences])
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])
        }
        candidates = []
        for title in baseline_actions[f"{query_id}::v4::fallback"]["context_titles"]:
            record = by_title.get(official.normalize_title(str(title)))
            if record is None:
                continue
            canonical_title, sentences = record
            candidates.extend({"title": canonical_title, "sent_id": sent_id, "text": sentence} for sent_id, sentence in enumerate(sentences))
        if not candidates:
            raise AssertionError(f"No RECOMP sentence candidates for {query_id}")
        texts = [str(item["question"])] + [row["text"] for row in candidates]
        encoded = compressor_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(args.compressor_device)
        with torch.inference_mode():
            outputs = compressor(**encoded)
            embeddings = mean_pooling(outputs[0], encoded["attention_mask"]).detach().cpu()
        scores = [float(embeddings[0] @ embeddings[candidate_index + 1]) for candidate_index in range(len(candidates))]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best = dict(candidates[best_index])
        best.update({
            "query_id": query_id,
            "question": str(item["question"]),
            "answer": str(item["answer"]),
            "compressor_score": scores[best_index],
            "candidate_sentence_count": len(candidates),
            "input_document_count": len(baseline_actions[f"{query_id}::v4::fallback"]["context_titles"]),
        })
        selected_sentences.append(best)
        if (index + 1) % 100 == 0:
            write_json(FAITHFUL / "compressor_progress.json", {"completed": index + 1, "total": len(query_ids)})
    write_jsonl(FAITHFUL / "recomp_selected_sentences.jsonl", selected_sentences)

    reader_tokenizer = AutoTokenizer.from_pretrained(FLAN, local_files_only=True, use_fast=True)
    reader = AutoModelForSeq2SeqLM.from_pretrained(FLAN, local_files_only=True, torch_dtype=torch.float16).to(args.reader_device)
    reader.eval()
    predictions = []
    for start in range(0, len(selected_sentences), args.reader_batch_size):
        batch = selected_sentences[start : start + args.reader_batch_size]
        prompts = [reader_prompt(row["question"], row["title"], row["text"]) for row in batch]
        encoded = reader_tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.reader_device)
        with torch.inference_mode():
            generated = reader.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        values = reader_tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, prediction in zip(batch, values):
            gold_support = {
                (official.normalize_title(title), int(sentence_id))
                for title, sentence_id in zip(
                    hotpot[row["query_id"]]["supporting_facts"]["title"],
                    hotpot[row["query_id"]]["supporting_facts"]["sent_id"],
                )
            }
            pred_support = {(official.normalize_title(row["title"]), int(row["sent_id"]))}
            metrics = official.official_metrics(prediction.strip(), row["answer"], pred_support, gold_support)
            predictions.append({
                "query_id": row["query_id"],
                "prediction": prediction.strip(),
                "selected_title": row["title"],
                "selected_sent_id": row["sent_id"],
                "compressor_score": row["compressor_score"],
                **metrics,
            })
    write_jsonl(FAITHFUL / "recomp_official_per_query.jsonl", predictions)

    baseline_rows = {
        str(row["query_id"]): row
        for row in read_jsonl(V4_ROOT / "outputs/official_metrics/official_hotpotqa_per_query.jsonl")
        if row["method"] == "baseline"
    }
    v4_rows = {
        str(row["query_id"]): row
        for row in read_jsonl(V4_ROOT / "outputs/official_metrics/official_hotpotqa_per_query.jsonl")
        if row["method"] == "v4_selected"
    }
    recomp_rows = {str(row["query_id"]): row for row in predictions}
    summaries = {
        "baseline": {metric: mean(float(baseline_rows[qid][metric]) for qid in query_ids) for metric in METRICS},
        "recomp": {metric: mean(float(recomp_rows[qid][metric]) for qid in query_ids) for metric in METRICS},
        "v4": {metric: mean(float(v4_rows[qid][metric]) for qid in query_ids) for metric in METRICS},
    }
    significance = {
        metric: official.paired_bootstrap([recomp_rows[qid][metric] - baseline_rows[qid][metric] for qid in query_ids])
        for metric in METRICS
    }
    result = {
        "status": "complete",
        "classification": "faithful_method_reproduction_with_standardized_reader_adaptation",
        "faithful_external_baseline": True,
        "method": "RECOMP extractive compressor",
        "official_repository": RECOMP_REPO,
        "official_repository_commit": RECOMP_COMMIT,
        "author_released_checkpoint": RECOMP_MODEL,
        "paper_hyperparameters": {"input_documents": 5, "selected_sentences": 1},
        "dataset": "same HotpotQA 1,000-query development evaluation as V4",
        "context_budget": "same frozen Top-5 baseline documents as compressor input",
        "reader_adaptation": "FLAN-T5-Large replaces paper FLAN-UL2 so all systems use the V4 reader",
        "reader_prompt_or_decoding_tuned": False,
        "target_holdout_tuning": False,
        "support_metric_extension": "the selected RECOMP sentence is used as the predicted support fact",
        "metrics": summaries,
        "recomp_vs_baseline_deltas": {metric: summaries["recomp"][metric] - summaries["baseline"][metric] for metric in METRICS},
        "v4_vs_recomp_deltas": {metric: summaries["v4"][metric] - summaries["recomp"][metric] for metric in METRICS},
        "recomp_vs_baseline_significance": significance,
    }
    write_json(FAITHFUL / "faithful_baseline_results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
