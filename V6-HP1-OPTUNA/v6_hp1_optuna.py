from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "V6-HP1-OPTUNA" / "outputs_stage2"


def parse_args() -> argparse.Namespace:
    default_python = Path.home() / "anaconda3" / "envs" / "supv2" / "bin" / "python"
    parser = argparse.ArgumentParser(description="Run second-pass narrowed Optuna search for V6-HP1 HotpotQA.")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--study-name", default="v6hp1_hotpot_stage2_low_payload")
    parser.add_argument("--storage", default="")
    parser.add_argument("--sampler-seed", type=int, default=20260612)
    parser.add_argument("--python", default=str(default_python if default_python.exists() else sys.executable))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--experiment-name", default="pprag_fl_v6_hp1_optuna_stage2")
    parser.add_argument("--suite-prefix", default="v6hp1_optuna_stage2")
    parser.add_argument("--task-name", default="num5_dir_a03_imb00_ts0_v6hp1")
    parser.add_argument("--rawdata-path", default=str(ROOT_DIR / "FedE" / "select_data_hotpot_train_5000.json"))
    parser.add_argument("--hotpot-split", default="validation")
    parser.add_argument("--hotpot-max-examples", type=int, default=1500)
    parser.add_argument("--eval-subset-type", "--eval_subset_type", choices=["all", "hard_only"], default="hard_only")
    parser.add_argument("--eval-num-examples", "--eval_num_examples", type=int, default=1000)
    parser.add_argument("--hard-query-subset", "--hard_query_subset", default=str(ROOT_DIR / "V6-HP1" / "data" / "hotpot_hard_query_subset.json"))
    parser.add_argument("--rag-script", default="main_100_test.py")
    parser.add_argument("--payload-penalty-mode", choices=["fixed", "search"], default="search")
    parser.add_argument("--payload-penalty", type=float, default=0.25)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save-per-query", action="store_true")
    return parser.parse_args()


def _trial_rows(study) -> list[dict]:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                **trial.params,
                **trial.user_attrs,
            }
        )
    return rows


def _write_outputs(study, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _trial_rows(study)
    (output_root / "optuna_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_root / "optuna_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete_trials = [trial for trial in study.trials if trial.value is not None]
    best = max(complete_trials, key=lambda trial: trial.value) if complete_trials else None
    lines = ["# V6-HP1 Optuna Stage2 Search Report", ""]
    lines.append(f"- Trials: {len(study.trials)}")
    lines.append(f"- Completed trials: {len(complete_trials)}")
    if best is not None:
        lines.append(f"- Best trial: {best.number}")
        lines.append(f"- Best objective: {best.value:.6f}")
        lines.append(f"- Best payload: {best.user_attrs.get('overall_payload_ratio', '')}")
        lines.append(f"- Best payload penalty: {best.user_attrs.get('payload_penalty', '')}")
        lines.append("")
        lines.append("## Best Parameters")
        lines.append("")
        for key, value in best.params.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
        lines.append("Fixed search constraints: `topk=2`, `warmup=0`, `use_utility_memory=False`.")
        lines.append("")
        lines.append("## Best Metrics")
        lines.append("")
        for key, value in best.user_attrs.items():
            if key.startswith("metric_"):
                lines.append(f"- `{key.removeprefix('metric_')}`: `{value}`")
    (output_root / "optuna_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install it with `pip install optuna`.") from exc

    from optuna_objective_stage2 import objective_factory

    output_root = Path(args.output_root).expanduser().resolve()
    optuna_root = output_root / "optuna"
    optuna_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{optuna_root / 'v6hp1_optuna_stage2.db'}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(subprocess.CalledProcessError, RuntimeError),
    )
    _write_outputs(study, output_root)
    complete_trials = [trial for trial in study.trials if trial.value is not None]
    if complete_trials:
        best = max(complete_trials, key=lambda trial: trial.value)
        print(f"Best trial: {best.number} value={best.value:.6f}")
    else:
        print("No completed trials yet; wrote failure summary for inspection.")
    print(f"Wrote Optuna stage2 outputs under {output_root}")


if __name__ == "__main__":
    main()
