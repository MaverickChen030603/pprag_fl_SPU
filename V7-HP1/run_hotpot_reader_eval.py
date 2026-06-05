#!/usr/bin/env python3
"""Reader/generator answer EM/F1 on top retrieved Hotpot contexts for V7-HP1."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

HELPER_PATH = Path(__file__).with_name("run_hotpot_official_eval.py")
spec = importlib.util.spec_from_file_location("hotpot_official_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(helper)  # type: ignore[union-attr]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--rawdata-path", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--prefer-official", action="store_true")
    p.add_argument("--reader-model", default="google-t5/t5-small")
    p.add_argument("--retrieval-topk", type=int, default=5)
    p.add_argument("--support-topk", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--reader-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--reader-max-input-length", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--local-reader-only", action="store_true")
    return p.parse_args()


def build_prompt(question: str, ranked_docs: List[Dict[str, Any]], topk: int) -> str:
    context_parts = []
    for doc in ranked_docs[:topk]:
        title = str(doc.get("title", ""))
        text = str(doc.get("text", doc.get("content", "")))
        context_parts.append(f"[{title}] {text}")
    context = " ".join(context_parts)
    return f"question: {question} context: {context} answer:"


def generate_answers(tokenizer, model, prompts: List[str], device: torch.device, batch_size: int, max_input_length: int, max_new_tokens: int) -> List[str]:
    outputs: List[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [o.strip() for o in outputs]


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    examples, mode = helper.load_official_examples(Path(args.rawdata_path), args.max_examples, args.prefer_official)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = helper.latest_hf_model(run_dir)
    metadata = helper.load_run_metadata(run_dir)
    if model_dir is None:
        result = {"status": "skipped", "reason": "no_hf_model", "run_dir": str(run_dir)}
        (output_dir / "reader_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ret_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    ret_model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)

    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=args.local_reader_only)
    reader_model = AutoModelForSeq2SeqLM.from_pretrained(args.reader_model, local_files_only=args.local_reader_only).to(device)

    prepared = []
    all_doc_texts: List[str] = []
    skipped_empty_context = 0
    for ex in examples:
        docs = helper.build_sentence_docs(ex)
        if not docs:
            skipped_empty_context += 1
            continue
        start = len(all_doc_texts)
        all_doc_texts.extend(d["content"] for d in docs)
        prepared.append((ex, docs, start, len(all_doc_texts)))

    if not prepared:
        result = {"status": "completed", "n": 0, "dataset_mode": mode, "metrics": {}, "skipped_empty_context": skipped_empty_context}
        (output_dir / "reader_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    doc_emb = helper.encode_texts(ret_tokenizer, ret_model, all_doc_texts, device, args.batch_size, args.max_length)
    query_emb = helper.encode_texts(ret_tokenizer, ret_model, [ex["question"] for ex, _, _, _ in prepared], device, args.batch_size, args.max_length)

    ranked_payload = []
    prompts: List[str] = []
    for qi, (ex, docs, start, end) in enumerate(prepared):
        scores = torch.matmul(doc_emb[start:end], query_emb[qi]).tolist()
        ranked = [d | {"score": float(score)} for d, score in sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)]
        ranked_payload.append((ex, ranked))
        prompts.append(build_prompt(ex["question"], ranked, args.retrieval_topk))

    predictions = generate_answers(
        reader_tokenizer,
        reader_model,
        prompts,
        device,
        args.reader_batch_size,
        args.reader_max_input_length,
        args.max_new_tokens,
    )

    totals = defaultdict(float)
    per_query = []
    hotpot_pred = {"answer": {}, "sp": {}}
    for (ex, ranked), pred_answer in zip(ranked_payload, predictions):
        pred_sp = [[d["title"], int(d["sent_id"])] for d in ranked[:args.support_topk]]
        ans_em = helper.exact_match_score(pred_answer, ex["answer"])
        ans_f1, ans_prec, ans_rec = helper.f1_score(pred_answer, ex["answer"])
        sp_em, sp_f1, sp_prec, sp_rec = helper.sp_metrics(pred_sp, ex["supporting_facts"])
        joint_em, joint_f1 = helper.joint_metrics(ans_em, ans_f1, sp_em, sp_f1)
        answer_access = float(any(helper.normalize_answer(ex["answer"]) in helper.normalize_answer(d["content"]) for d in ranked[:args.retrieval_topk]))
        gold_titles = {str(t) for t, _ in ex["supporting_facts"]}
        pred_titles = {str(t) for t, _ in pred_sp}
        support_title_recall = len(gold_titles & pred_titles) / len(gold_titles) if gold_titles else 0.0
        gold_set = {(str(t), int(i)) for t, i in ex["supporting_facts"]}
        pred_set = {(str(t), int(i)) for t, i in pred_sp}
        metrics = {
            "answer_em": ans_em,
            "answer_f1": ans_f1,
            "answer_precision": ans_prec,
            "answer_recall": ans_rec,
            "sp_em": sp_em,
            "sp_f1": sp_f1,
            "sp_precision": sp_prec,
            "sp_recall": sp_rec,
            "joint_em": joint_em,
            "joint_f1": joint_f1,
            "answer_access_at_k": answer_access,
            "support_title_recall_at_k": support_title_recall,
            "all_gold_sp_retrieved_at_k": float(gold_set.issubset(pred_set)),
        }
        for k, v in metrics.items():
            totals[k] += float(v)
        rec = {
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "pred_answer": pred_answer,
            "gold_sp": ex["supporting_facts"],
            "pred_sp": pred_sp,
            "top_context": ranked[:args.retrieval_topk],
            "metrics": metrics,
            "source": ex.get("source", ""),
        }
        per_query.append(rec)
        hotpot_pred["answer"][ex["id"]] = pred_answer
        hotpot_pred["sp"][ex["id"]] = pred_sp

    count = len(per_query)
    means = {k: v / count for k, v in totals.items()}
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "retriever_model_dir": str(model_dir),
        "reader_model": args.reader_model,
        "output_dir": str(output_dir),
        "n": count,
        "skipped_empty_context": skipped_empty_context,
        "dataset_mode": mode,
        "metrics": means,
        "method": metadata.get("selection_strategy"),
        "profile": (metadata.get("option") or {}).get("v7_agent_profile"),
        "suite_tag": metadata.get("suite_tag"),
        "task_name": metadata.get("task_name"),
        "seed": metadata.get("seed"),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "reader_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "per_query_reader.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_query) + "\n", encoding="utf-8")
    (output_dir / "hotpot_reader_predictions.json").write_text(json.dumps(hotpot_pred, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
