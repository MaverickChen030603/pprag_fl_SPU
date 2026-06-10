from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from search_space import TrialConfig, suggest_trial_config


ROOT_DIR = Path(__file__).resolve().parents[1]
V6_HP1_DIR = ROOT_DIR / "V6-HP1"
OUTPUT_ROOT = ROOT_DIR / "V6-HP1-OPTUNA" / "outputs"
TRIAL_RESULTS_ROOT = OUTPUT_ROOT / "trial_results"
RAG_ROOT = OUTPUT_ROOT / "rag_eval"

if str(V6_HP1_DIR) not in sys.path:
    sys.path.insert(0, str(V6_HP1_DIR))

from report_generator import summarize_downstream_run  # noqa: E402
from summarize_results import summarize_run  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latest_hf_model(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("retriever_hf_*"))
    return candidates[-1] if candidates else None


def _find_single_run_dir(suite_root: Path) -> Path:
    runs = sorted(path.parent for path in suite_root.rglob("run_metadata.json"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one run under {suite_root}, found {len(runs)}")
    return runs[0]


def _command_for_trial(
    *,
    python_bin: str,
    trial_number: int,
    cfg: TrialConfig,
    args: Any,
) -> list[str]:
    suite_tag = f"{args.suite_prefix}_t{trial_number:04d}"
    command = [
        python_bin,
        str(V6_HP1_DIR / "run_upstream.py"),
        "--strategy",
        "hypernet_v6",
        "--topk",
        str(cfg.topk),
        "--warmup",
        str(cfg.warmup),
        "--rounds",
        str(args.rounds),
        "--clients",
        str(args.clients),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--experiment-name",
        args.experiment_name,
        "--suite-tag",
        suite_tag,
        "--task-name",
        args.task_name,
        "--rawdata-path",
        args.rawdata_path,
        "--rag-dataset",
        "hotpot_qa",
        "--rag-hotpot-split",
        args.hotpot_split,
        "--rag-hotpot-max-examples",
        str(args.hotpot_max_examples),
        "--score-mode",
        cfg.score_mode,
        "--budget-mode",
        cfg.budget_mode,
        "--history-window",
        str(cfg.history_window),
        "--hard-query-scale",
        str(cfg.hard_query_scale),
        "--hard-client-threshold",
        str(cfg.hard_client_threshold),
        "--adaptive-expand-threshold",
        str(cfg.adaptive_expand_threshold),
        "--adaptive-shrink-threshold",
        str(cfg.adaptive_shrink_threshold),
        "--utility-expand-threshold",
        str(cfg.utility_expand_threshold),
    ]
    if not cfg.use_hard_query_weighting:
        command.append("--disable-hard-query-weighting")
    if not cfg.use_utility_memory:
        command.append("--disable-utility-memory")
    if cfg.layerwise_budget:
        command.append("--layerwise-budget")
    return command


def _run_rag_eval(*, python_bin: str, model_dir: Path, output_dir: Path, args: Any) -> None:
    command = [
        python_bin,
        str(V6_HP1_DIR / "run_rag_eval.py"),
        "--model",
        str(model_dir),
        "--script",
        args.rag_script,
        "--output-dir",
        str(output_dir),
        "--python",
        python_bin,
        "--dataset",
        "hotpot_qa",
        "--hotpot-split",
        args.hotpot_split,
        "--hotpot-max-examples",
        str(args.eval_examples),
    ]
    if args.save_per_query:
        command.append("--save-per-query")
    subprocess.run(command, cwd=str(ROOT_DIR), check=True)


def compute_objective(metrics: dict[str, Any], payload: float, payload_penalty: float) -> float:
    mrr = float(metrics.get("mrr", 0.0))
    ndcg = float(metrics.get("ndcg", metrics.get("NDCG", 0.0)))
    f1 = float(metrics.get("f1", metrics.get("F1", 0.0)))
    em = float(metrics.get("em", metrics.get("EM", 0.0)))
    recall_3 = float(metrics.get("recall_3", 0.0))
    quality = 0.30 * mrr + 0.25 * ndcg + 0.20 * f1 + 0.15 * em + 0.10 * recall_3
    return quality - payload_penalty * payload


def objective_factory(args: Any):
    python_bin = args.python

    def objective(trial: Any) -> float:
        cfg = suggest_trial_config(trial)
        suite_tag = f"{args.suite_prefix}_t{trial.number:04d}"
        suite_root = ROOT_DIR / "V6-HP1" / "outputs" / args.experiment_name / suite_tag
        trial_result_dir = TRIAL_RESULTS_ROOT / f"trial_{trial.number:04d}"
        rag_output_dir = RAG_ROOT / f"trial_{trial.number:04d}"
        command = _command_for_trial(python_bin=python_bin, trial_number=trial.number, cfg=cfg, args=args)

        trial_result_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            trial_result_dir / "trial_config.json",
            {
                "trial_number": trial.number,
                "trial_params": asdict(cfg),
                "upstream_command": command,
                "suite_root": str(suite_root),
            },
        )

        subprocess.run(command, cwd=str(ROOT_DIR), check=True)
        run_dir = _find_single_run_dir(suite_root)
        upstream_summary = summarize_run(run_dir)
        model_dir = _latest_hf_model(run_dir)
        if model_dir is None:
            raise RuntimeError(f"No exported HF retriever found under {run_dir}")

        _run_rag_eval(python_bin=python_bin, model_dir=model_dir, output_dir=rag_output_dir, args=args)
        downstream_summary = summarize_downstream_run(rag_output_dir, run_dir, str(run_dir.relative_to(suite_root)))
        metrics = downstream_summary.get("metrics", {})
        payload = float(upstream_summary.get("overall_payload_ratio", 1.0))
        score = compute_objective(metrics, payload, args.payload_penalty)

        result = {
            "trial_number": trial.number,
            "objective": score,
            "trial_params": asdict(cfg),
            "run_dir": str(run_dir),
            "rag_output_dir": str(rag_output_dir),
            "upstream": upstream_summary,
            "downstream": downstream_summary,
            "metrics": metrics,
            "payload_penalty": args.payload_penalty,
        }
        _write_json(trial_result_dir / "trial_result.json", result)
        for key, value in metrics.items():
            try:
                trial.set_user_attr(f"metric_{key}", float(value))
            except (TypeError, ValueError):
                trial.set_user_attr(f"metric_{key}", value)
        trial.set_user_attr("overall_payload_ratio", payload)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("rag_output_dir", str(rag_output_dir))
        return score

    return objective
