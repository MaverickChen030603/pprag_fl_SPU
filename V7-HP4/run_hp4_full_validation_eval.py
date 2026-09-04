from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_reader_helpers():
    helper_path = Path("V7-HP4/run_hp4_reader_counterfactual_eval.py")
    spec = importlib.util.spec_from_file_location("v7_hp4_reader_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load HP4 reader helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_READER = _load_reader_helpers()
Reader = _READER.Reader
build_retrieval_item = _READER.build_retrieval_item
evaluate_dataset = _READER.evaluate_dataset
materialize_dev = _READER.materialize_dev
summarize = _READER.summarize


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _context_to_reference(context: Any) -> str:
    parts: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        for title, sents in zip(titles, sentences):
            if isinstance(sents, list):
                text = " ".join(str(s) for s in sents)
            else:
                text = str(sents)
            parts.append(f"[{title}] {text}")
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                body = item[1]
                text = " ".join(str(s) for s in body) if isinstance(body, list) else str(body)
                parts.append(f"[{title}] {text}")
            elif isinstance(item, dict):
                title = item.get("title", "")
                body = item.get("sentences", item.get("text", ""))
                text = " ".join(str(s) for s in body) if isinstance(body, list) else str(body)
                parts.append(f"[{title}] {text}")
    return " ".join(parts)


def _support_titles(item: dict[str, Any]) -> list[str]:
    if "supporting_titles" in item:
        return [str(t) for t in item.get("supporting_titles", [])]
    facts = item.get("supporting_facts", [])
    titles: list[str] = []
    if isinstance(facts, dict):
        facts = facts.get("title", [])
    for fact in facts:
        title = fact[0] if isinstance(fact, (list, tuple)) and fact else fact
        title = str(title)
        if title and title not in titles:
            titles.append(title)
    return titles


def _normalize_hotpot_item(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    reference = str(item.get("reference", "") or "")
    if not reference and "context" in item:
        reference = _context_to_reference(item.get("context"))
    support_titles = _support_titles(item)
    if not reference or not support_titles:
        return None
    return {
        "_id": str(item.get("_id", item.get("id", idx))),
        "question": str(item.get("question", "")),
        "answer": item.get("answer", ""),
        "supporting_titles": support_titles,
        "reference": reference,
    }


def load_or_build_validation_split(
    preferred_path: Path,
    output_path: Path,
    max_examples: int,
    seed: int,
    fallback_path: Path,
) -> tuple[Path, str, int]:
    if preferred_path.exists():
        payload = _load_json(preferred_path)
        return preferred_path, "existing_preferred", len(payload)
    if output_path.exists():
        payload = _load_json(output_path)
        return output_path, "existing_generated", len(payload)

    candidates: list[dict[str, Any]] = []
    source = ""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        for idx, item in enumerate(ds):
            normalized = _normalize_hotpot_item(dict(item), idx)
            if normalized is not None:
                candidates.append(normalized)
        source = "huggingface:hotpot_qa/distractor/validation"
    except Exception as exc:
        source = f"fallback:{fallback_path} ({type(exc).__name__}: {exc})"
        raw = _load_json(fallback_path)
        for idx, item in enumerate(raw):
            normalized = _normalize_hotpot_item(item, idx)
            if normalized is not None:
                candidates.append(normalized)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    split = candidates[:max_examples]
    if len(split) < min(max_examples, 500):
        raise RuntimeError(f"Only built {len(split)} validation examples from {source}")
    _save_json(output_path, split)
    _save_json(output_path.with_suffix(".meta.json"), {
        "source": source,
        "seed": seed,
        "requested_examples": max_examples,
        "actual_examples": len(split),
    })
    return output_path, source, len(split)


def paired_rows(rows: list[dict[str, Any]], metric: str) -> list[float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("mode") in {"baseline_uniform", "hp4_soft_agent"}:
            by_id[str(row.get("id"))][str(row.get("mode"))] = row
    diffs = []
    for modes in by_id.values():
        if "baseline_uniform" not in modes or "hp4_soft_agent" not in modes:
            continue
        diffs.append(float(modes["hp4_soft_agent"].get(metric, 0.0)) - float(modes["baseline_uniform"].get(metric, 0.0)))
    return diffs


def permutation_p_value(diffs: list[float], rounds: int = 10000, seed: int = 13) -> dict[str, float]:
    if not diffs:
        return {"n": 0, "mean_diff": 0.0, "p_value_two_sided": 1.0, "ci95_low": 0.0, "ci95_high": 0.0}
    observed = mean(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(rounds):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(mean(signed)) >= abs(observed):
            extreme += 1
    boots = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return {
        "n": len(diffs),
        "mean_diff": observed,
        "p_value_two_sided": (extreme + 1.0) / (rounds + 1.0),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def write_report(path: Path, summary: dict[str, Any], significance: dict[str, Any], data_source: str, rows_path: Path) -> None:
    def table() -> str:
        lines = [
            "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ["baseline_uniform", "hp4_soft_agent"]:
            m = summary.get(mode, {})
            lines.append(
                f"| {mode} | {m.get('n', 0)} | {m.get('answer_access_at_k', 0):.4f} | "
                f"{m.get('support_recall_at_k', 0):.4f} | {m.get('sp_f1', 0):.4f} | "
                f"{m.get('answer_em', 0):.4f} | {m.get('answer_f1', 0):.4f} | {m.get('joint_f1', 0):.4f} |"
            )
        return "\n".join(lines)

    joint_gap = summary.get("hp4_soft_agent", {}).get("joint_f1", 0.0) - summary.get("baseline_uniform", {}).get("joint_f1", 0.0)
    sp_gap = summary.get("hp4_soft_agent", {}).get("sp_f1", 0.0) - summary.get("baseline_uniform", {}).get("sp_f1", 0.0)
    text = f"""# V7-HP4 Phase 2 Full Validation Reader Significance Report

Data source: `{data_source}`

Rows: `{rows_path}`

## Full Validation Metrics

{table()}

## Paired Significance

| metric | n | mean_gap | p_value_two_sided | bootstrap_ci95 |
| --- | ---: | ---: | ---: | ---: |
| joint_f1 | {significance['joint_f1']['n']} | {significance['joint_f1']['mean_diff']:+.4f} | {significance['joint_f1']['p_value_two_sided']:.6f} | [{significance['joint_f1']['ci95_low']:+.4f}, {significance['joint_f1']['ci95_high']:+.4f}] |
| sp_f1 | {significance['sp_f1']['n']} | {significance['sp_f1']['mean_diff']:+.4f} | {significance['sp_f1']['p_value_two_sided']:.6f} | [{significance['sp_f1']['ci95_low']:+.4f}, {significance['sp_f1']['ci95_high']:+.4f}] |

## Interpretation

- joint_f1 aggregate gap: {joint_gap:+.4f}
- sp_f1 aggregate gap: {sp_gap:+.4f}
- The paired test uses query-level soft-agent minus baseline deltas with random sign-flip permutation and bootstrap confidence intervals.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_full_validation")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--permutation-rounds", type=int, default=10000)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dev_path, data_source, split_n = load_or_build_validation_split(
        Path(args.preferred_dev),
        Path(args.generated_dev),
        args.max_dev,
        args.seed,
        Path(args.fallback_dev),
    )
    examples = materialize_dev(dev_path, args.max_dev)
    if len(examples) < min(args.max_dev, split_n) * 0.8:
        raise RuntimeError(f"Only materialized {len(examples)} examples from {dev_path}")

    reader = Reader(args.reader_model, args.device, args.reader_batch_size)
    rows, summary = evaluate_dataset(examples, reader, args.top_k, args.alpha, counterfactual=False)
    rows_path = out / "full_validation_reader_rows.json"
    summary_path = out / "full_validation_reader_summary.json"
    sig_path = out / "full_validation_significance.json"
    _save_json(rows_path, rows)
    _save_json(summary_path, summary)
    significance = {
        "joint_f1": permutation_p_value(paired_rows(rows, "joint_f1"), args.permutation_rounds, args.seed + 1),
        "sp_f1": permutation_p_value(paired_rows(rows, "sp_f1"), args.permutation_rounds, args.seed + 2),
    }
    _save_json(sig_path, significance)
    report = Path(args.report_dir) / "v7_hp4_phase2_full_validation_significance_latest.md"
    write_report(report, summary, significance, data_source, rows_path)
    print(json.dumps({
        "examples": len(examples),
        "data_source": data_source,
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "significance_path": str(sig_path),
        "report_path": str(report),
        "summary": summary,
        "significance": significance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
