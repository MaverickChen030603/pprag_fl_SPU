#!/usr/bin/env python3
"""Shared frozen-artifact utilities for the SIGIR-AP strengthening analyses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
V4 = PAPER_ROOT / "opportunity_aware_semantic_generation_v4"
V5 = PAPER_ROOT / "review_driven_revision_v5"
COMPLETION = PAPER_ROOT / "v4_submission_completion"
OUTPUTS = HERE / "outputs"
REPORTS = HERE / "reports"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

FLAN = Path(
    "/home/iiserver31/.cache/huggingface/hub/"
    "models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
)
DEFAULT_ARROW = Path(
    "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/"
    "1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
)
METRICS = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")
SEED = 20260715

SPLITS: dict[str, dict[str, Any]] = {
    "development1000": {
        "label": "Development (nested 1,000)",
        "n": 1000,
        "prefix": "v4",
        "actions": V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl",
        "selections": V4 / "outputs/nested_selector/v4_nested_per_query.jsonl",
        "reader_outcomes": V4 / "outputs/action_outcomes/v4_action_outputs.jsonl",
        "cache": V4 / "outputs/semantic_generator/semantic_feature_cache.joblib",
        "source": None,
        "diagnostic_status": "mechanism diagnostic",
    },
    "holdout3000": {
        "label": "Original holdout (3,000)",
        "n": 3000,
        "prefix": "v4scale",
        "actions": V4 / "outputs/scaleup/generated_actions_3000.jsonl",
        "selections": V4 / "outputs/scaleup/frozen_selector_selections_3000.jsonl",
        "cache": V4 / "outputs/scaleup/semantic_feature_cache_3000.joblib",
        "source": V4 / "outputs/scaleup/same_source_hotpot_validation_3000.json",
        "diagnostic_status": "post-hoc outcome-aware diagnostic",
    },
    "revision3405": {
        "label": "Revision holdout (3,405)",
        "n": 3405,
        "prefix": "v4revision",
        "actions": V5 / "outputs/lite_model/revision_holdout/full_v4_actions_3405.jsonl",
        "selections": V5 / "outputs/lite_model/revision_holdout/full_v4_selections_3405.jsonl",
        "cache": V5 / "outputs/lite_model/revision_holdout/full_v4_semantic_feature_cache_3405.joblib",
        "source": V5 / "outputs/lite_model/revision_holdout/hotpot_revision_holdout_3405.json",
        "diagnostic_status": "post-hoc outcome-aware diagnostic",
    },
}


def ensure_layout() -> None:
    for path in (
        OUTPUTS / "oracle",
        OUTPUTS / "reranker",
        OUTPUTS / "pareto",
        OUTPUTS / "2wiki_analysis",
        OUTPUTS / "intervention_profile",
        TABLES,
        FIGURES,
        REPORTS,
        HERE / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_api() -> Any:
    v4_path = str(V4)
    if v4_path not in sys.path:
        sys.path.insert(0, v4_path)
    return load_module(V4 / "08_run_official_hotpot_evaluation.py", "sigirap_official")


def source_rows(split: str, official: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    config = SPLITS[split]
    if config["source"] is None:
        if official is None:
            raise ValueError("Development source requires the official Hotpot dataset")
        return {query_id: row for query_id, row in official.items() if query_id in selection_ids(split)}
    rows = read_json(config["source"])
    return {str(row.get("_id", row.get("id", row.get("query_id")))): row for row in rows}


def selection_ids(split: str) -> set[str]:
    return {str(row["query_id"]) for row in iter_jsonl(SPLITS[split]["selections"])}


def baseline_action_id(split: str, query_id: str) -> str:
    return f"{query_id}::{SPLITS[split]['prefix']}::fallback"


def stable_shard(query_id: str, num_shards: int) -> int:
    digest = hashlib.md5(query_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def flan_prompt(question: str, docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    context = "\n".join(
        f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1)
    )
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"
    )


def paired_bootstrap(differences: list[float], rounds: int = 5000) -> dict[str, float]:
    if not differences:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value": 1.0}
    rng = random.Random(SEED)
    n = len(differences)
    samples = [mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(rounds)]
    samples.sort()
    lower = samples[int(0.025 * rounds)]
    upper = samples[min(rounds - 1, int(0.975 * rounds))]
    p_value = min(
        1.0,
        2 * min(sum(value <= 0 for value in samples) / rounds, sum(value >= 0 for value in samples) / rounds),
    )
    return {"mean": mean(differences), "ci95_low": lower, "ci95_high": upper, "p_value": p_value}


def metric_means(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = list(rows)
    return {metric: mean(float(row[metric]) for row in values) for metric in METRICS}


def win_loss_tie(values: Iterable[float], eps: float = 1e-12) -> dict[str, int]:
    differences = list(values)
    return {
        "wins": sum(value > eps for value in differences),
        "losses": sum(value < -eps for value in differences),
        "ties": sum(abs(value) <= eps for value in differences),
    }


def answer_scores(prediction: str, gold: str) -> tuple[float, float]:
    api = official_api()
    metrics = api.answer_metrics(prediction, gold)
    return float(metrics["em"]), float(metrics["f1"])


def title_proxy(context_titles: list[str], supporting_titles: list[str]) -> tuple[float, float]:
    def norm(value: str) -> str:
        return " ".join(str(value).lower().split())

    predicted = {norm(value) for value in context_titles}
    gold = {norm(value) for value in supporting_titles}
    overlap = len(predicted & gold)
    recall = overlap / len(gold) if gold else 0.0
    precision = overlap / len(predicted) if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, f1


def token_count_approx(docs: list[dict[str, Any]], max_chars: int = 3200) -> int:
    context = "\n".join(f"{doc['title']}: {doc['text']}" for doc in docs)[:max_chars]
    return len(context.split())


def normalize_answer(text: str) -> list[str]:
    api = official_api()
    return api.normalize_answer(text).split()


def answer_precision_recall(prediction: str, gold: str) -> tuple[float, float]:
    pred, truth = normalize_answer(prediction), normalize_answer(gold)
    overlap = sum((Counter(pred) & Counter(truth)).values())
    return (
        overlap / len(pred) if pred else float(pred == truth),
        overlap / len(truth) if truth else float(pred == truth),
    )
