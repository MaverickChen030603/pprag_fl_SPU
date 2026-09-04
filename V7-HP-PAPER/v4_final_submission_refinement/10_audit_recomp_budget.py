#!/usr/bin/env python3
"""Measure RECOMP and V4 context budgets under the frozen FLAN tokenizer."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FEDE4RAG_ROOT", HERE.parents[1])).resolve()
V4 = ROOT / "V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
COMPLETION = ROOT / "V7-HP-PAPER/v4_submission_completion"
FLAN = Path(
    os.environ.get(
        "V4_FLAN_T5_LARGE",
        "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/"
        "snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    )
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def context_text(docs: list[dict]) -> str:
    return "\n".join(
        f"[{index}] {doc['title']}: {doc['text']}"
        for index, doc in enumerate(docs, start=1)
    )


def full_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:3200]}\n\nAnswer:"
    )


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FLAN, local_files_only=True, use_fast=True)
    actions = read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    action_by_id = {str(row["action_id"]): row for row in actions}
    baselines = {
        str(row["query_id"]): row
        for row in actions
        if row["action_family"] == "fallback"
    }
    selected = {
        str(row["query_id"]): row
        for row in read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
    }
    recomp = {
        str(row["query_id"]): row
        for row in read_jsonl(COMPLETION / "outputs/faithful_baseline/recomp_selected_sentences.jsonl")
    }
    query_ids = sorted(recomp)
    if set(query_ids) != set(baselines) or set(query_ids) != set(selected):
        raise AssertionError("RECOMP, baseline, and selector query sets differ")

    rows = []
    for query_id in query_ids:
        baseline = baselines[query_id]
        selected_action = action_by_id[str(selected[query_id]["action_id"])]
        recomp_row = recomp[query_id]
        question = str(recomp_row["question"])
        baseline_context_raw = context_text(baseline["context_docs"])
        selected_context_raw = context_text(selected_action["context_docs"])
        recomp_context_raw = f"[1] {recomp_row['title']}: {recomp_row['text']}"
        baseline_context = baseline_context_raw[:3200]
        selected_context = selected_context_raw[:3200]
        recomp_context = recomp_context_raw[:3200]

        def tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=True))

        rows.append(
            {
                "query_id": query_id,
                "baseline_doc_count": len(baseline["context_docs"]),
                "selected_doc_count": len(selected_action["context_docs"]),
                "recomp_sentence_count": 1,
                "candidate_sentence_count": int(recomp_row["candidate_sentence_count"]),
                "baseline_context_tokens": tokens(baseline_context),
                "selected_context_tokens": tokens(selected_context),
                "recomp_context_tokens": tokens(recomp_context),
                "baseline_prompt_tokens": tokens(full_prompt(question, baseline_context_raw)),
                "selected_prompt_tokens": tokens(full_prompt(question, selected_context_raw)),
                "recomp_prompt_tokens": tokens(full_prompt(question, recomp_context_raw)),
                "baseline_char_truncated": len(baseline_context_raw) > 3200,
                "selected_char_truncated": len(selected_context_raw) > 3200,
                "recomp_char_truncated": len(recomp_context_raw) > 3200,
            }
        )

    summary = {
        "status": "complete",
        "tokenizer": "google/flan-t5-large",
        "tokenizer_path": str(FLAN),
        "n_queries": len(rows),
        "same_input_top5": True,
        "same_output_token_budget": False,
        "reader_context_char_limit": 3200,
        "reader_token_limit": 1024,
        "averages": {
            key: mean(float(row[key]) for row in rows)
            for key in [
                "baseline_doc_count",
                "selected_doc_count",
                "recomp_sentence_count",
                "candidate_sentence_count",
                "baseline_context_tokens",
                "selected_context_tokens",
                "recomp_context_tokens",
                "baseline_prompt_tokens",
                "selected_prompt_tokens",
                "recomp_prompt_tokens",
            ]
        },
        "compression_ratios": {
            "recomp_to_baseline_context_tokens": mean(
                row["recomp_context_tokens"] / max(1, row["baseline_context_tokens"])
                for row in rows
            ),
            "recomp_to_v4_selected_context_tokens": mean(
                row["recomp_context_tokens"] / max(1, row["selected_context_tokens"])
                for row in rows
            ),
        },
        "char_truncation_rates": {
            name: mean(float(row[f"{name}_char_truncated"]) for row in rows)
            for name in ["baseline", "selected", "recomp"]
        },
        "document_count_distributions": {
            "baseline": dict(Counter(row["baseline_doc_count"] for row in rows)),
            "v4_selected": dict(Counter(row["selected_doc_count"] for row in rows)),
        },
        "interpretation": (
            "RECOMP and V4 receive the same baseline Top-5 pool, but they do not expose the "
            "reader to the same output budget. RECOMP emits one sentence while V4 preserves a "
            "five-document bounded context. The comparison therefore tests compatibility with "
            "the standardized reader under the evaluated Top-1 setting, not universal method superiority."
        ),
    }
    output = HERE / "recomp_budget_audit.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
