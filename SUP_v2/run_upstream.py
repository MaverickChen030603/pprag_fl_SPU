from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
FEDE_DIR = REPO_ROOT / "FedE"

if str(FEDE_DIR) not in sys.path:
    sys.path.insert(0, str(FEDE_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from experiment_config import UpstreamConfig
from metrics import ensure_dir, write_json
from report_generator import write_single_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUP_v2 upstream FL experiment.")
    parser.add_argument("--strategy", default="hypernet", choices=["full", "random", "static_top", "delta_norm", "hypernet"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_sup_v2")
    parser.add_argument("--task-name", default="num5_alpha05_sup_v2")
    parser.add_argument("--estimate-encryption", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UpstreamConfig:
    return UpstreamConfig(
        experiment_name=args.experiment_name,
        task_name=args.task_name,
        num_clients=args.clients,
        num_rounds=args.rounds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gpu=args.gpu,
        seed=args.seed,
        selection_strategy=args.strategy,
        topk_blocks=args.topk,
        warmup_rounds=args.warmup,
        estimate_encryption=args.estimate_encryption,
    )


def ensure_task(config: UpstreamConfig) -> None:
    import flgo

    task_config = {
        "benchmark": {"name": "flgo.benchmark.fedrag_classification"},
        "partitioner": {"name": "IDPartitioner", "para": {"num_clients": config.num_clients}},
    }
    if config.task_path.exists():
        return
    old_cwd = os.getcwd()
    os.chdir(FEDE_DIR)
    try:
        flgo.gen_task(task_config, task_path=str(config.task_path))
    finally:
        os.chdir(old_cwd)


def run(config: UpstreamConfig):
    import flgo
    import fedrag_selective_upload

    ensure_dir(config.output_dir)
    write_json(config.output_dir / "upstream_config.json", config.to_dict())
    ensure_task(config)
    runner = flgo.init(
        task=str(config.task_path),
        algorithm=fedrag_selective_upload,
        option=config.to_flgo_option(),
    )
    runner.run()
    return runner


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.dry_run:
        print(config.to_dict())
        return
    run(config)
    report_path = write_single_run_report(config, config.output_dir)
    print(f"Analysis report written to {report_path}")


if __name__ == "__main__":
    main()
