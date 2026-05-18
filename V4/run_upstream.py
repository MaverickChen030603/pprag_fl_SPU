from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
FEDE_DIR = REPO_ROOT / "FedE"

if str(FEDE_DIR) not in sys.path:
    sys.path.insert(0, str(FEDE_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from experiment_config import UpstreamConfig, build_task_name
from metrics import ensure_dir, write_json
from report_generator import write_single_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 upstream FL experiment.")
    parser.add_argument(
        "--strategy",
        default="hypernet_v4",
        choices=["full", "random", "static_top", "delta_norm", "hypernet_v2", "hypernet_v3", "hypernet_v4"],
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v4")
    parser.add_argument("--suite-tag", default="v4_adhoc")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--partitioner", default="DirichletPartitioner", choices=["IDPartitioner", "DirichletPartitioner"])
    parser.add_argument("--dir-alpha", type=float, default=0.3)
    parser.add_argument("--dir-error-bar", type=float, default=1e-3)
    parser.add_argument("--imbalance", type=float, default=0.0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--overwrite-task", action="store_true")
    parser.add_argument("--estimate-encryption", action="store_true")
    parser.add_argument("--score-mode", choices=["importance", "value", "downstream_value"], default="downstream_value")
    parser.add_argument("--budget-mode", choices=["fixed", "adaptive", "adaptive_v4"], default="adaptive_v4")
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--disable-client-embedding", action="store_true")
    parser.add_argument("--disable-history-features", action="store_true")
    parser.add_argument("--disable-block-embedding", action="store_true")
    parser.add_argument("--adaptive-min-topk", type=int, default=1)
    parser.add_argument("--adaptive-max-topk", type=int, default=7)
    parser.add_argument("--adaptive-scale", type=float, default=1.0)
    parser.add_argument("--layerwise-budget", action="store_true")
    parser.add_argument("--disable-hard-query-weighting", action="store_true")
    parser.add_argument("--disable-utility-memory", action="store_true")
    parser.add_argument("--hard-query-scale", type=float, default=1.0)
    parser.add_argument("--hard-client-threshold", type=float, default=0.55)
    parser.add_argument("--hard-client-bonus-topk", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UpstreamConfig:
    task_name = args.task_name or build_task_name(
        num_clients=args.clients,
        partitioner_name=args.partitioner,
        dirichlet_alpha=args.dir_alpha,
        imbalance=args.imbalance,
        task_seed=args.task_seed,
    )
    return UpstreamConfig(
        experiment_name=args.experiment_name,
        suite_tag=args.suite_tag,
        task_name=task_name,
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
        partitioner_name=args.partitioner,
        dirichlet_alpha=args.dir_alpha,
        imbalance=args.imbalance,
        dirichlet_error_bar=args.dir_error_bar,
        task_seed=args.task_seed,
        overwrite_task=args.overwrite_task,
        score_mode=args.score_mode,
        budget_mode=args.budget_mode,
        history_window=args.history_window,
        use_client_embedding=not args.disable_client_embedding,
        use_history_features=not args.disable_history_features,
        use_block_embedding=not args.disable_block_embedding,
        adaptive_min_topk=args.adaptive_min_topk,
        adaptive_max_topk=args.adaptive_max_topk,
        adaptive_scale=args.adaptive_scale,
        layerwise_budget=args.layerwise_budget,
        use_hard_query_weighting=not args.disable_hard_query_weighting,
        use_utility_memory=not args.disable_utility_memory,
        hard_query_scale=args.hard_query_scale,
        hard_client_threshold=args.hard_client_threshold,
        hard_client_bonus_topk=args.hard_client_bonus_topk,
    )


def ensure_task(config: UpstreamConfig) -> None:
    import flgo
    import flgo.benchmark.partition as fbp

    task_config = {"benchmark": {"name": "flgo.benchmark.fedrag_classification"}}
    if config.partitioner_name == "DirichletPartitioner":
        task_config["partitioner"] = {
            "name": fbp.DirichletPartitioner,
            "para": {
                "num_clients": config.num_clients,
                "alpha": config.dirichlet_alpha,
                "error_bar": config.dirichlet_error_bar,
                "imbalance": config.imbalance,
            },
        }
    else:
        task_config["partitioner"] = {"name": fbp.IDPartitioner, "para": {"num_clients": config.num_clients}}
    if config.task_path.exists() and not config.overwrite_task:
        return
    old_cwd = os.getcwd()
    os.chdir(FEDE_DIR)
    try:
        flgo.gen_task(task_config, task_path=str(config.task_path), seed=config.task_seed, overwrite=config.overwrite_task)
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
    print(f"V4 analysis report written to {report_path}")


if __name__ == "__main__":
    main()
