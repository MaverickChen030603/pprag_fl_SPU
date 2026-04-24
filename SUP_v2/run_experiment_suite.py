from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, List

from experiment_config import (
    UpstreamConfig,
    comparison_suite,
    encryption_ablation_suite,
    topk_ablation_suite,
    warmup_ablation_suite,
)
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUP_v2 experiment suites.")
    parser.add_argument("--suite", default="comparison", choices=["comparison", "topk", "warmup", "encryption", "all"])
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_sup_v2")
    parser.add_argument("--task-name", default="num5_alpha05_sup_v2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> UpstreamConfig:
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
    )


def build_suite(args: argparse.Namespace) -> List[UpstreamConfig]:
    base = base_config(args)
    if args.suite == "comparison":
        return comparison_suite(base)
    if args.suite == "topk":
        return topk_ablation_suite(replace(base, selection_strategy="hypernet"))
    if args.suite == "warmup":
        return warmup_ablation_suite(replace(base, selection_strategy="hypernet"))
    if args.suite == "encryption":
        return encryption_ablation_suite(replace(base, selection_strategy="hypernet"))
    configs = []
    configs.extend(comparison_suite(base))
    configs.extend(topk_ablation_suite(replace(base, selection_strategy="hypernet")))
    configs.extend(warmup_ablation_suite(replace(base, selection_strategy="hypernet")))
    configs.extend(encryption_ablation_suite(replace(base, selection_strategy="hypernet")))
    dedup = {}
    for config in configs:
        dedup[(config.selection_strategy, config.topk_blocks, config.warmup_rounds, config.estimate_encryption)] = config
    return list(dedup.values())


def main() -> None:
    args = parse_args()
    configs = build_suite(args)
    manifest = [config.to_dict() for config in configs]
    if configs:
        ensure_dir(configs[0].output_dir.parent)
        write_json(configs[0].output_dir.parent / f"{args.suite}_manifest.json", manifest)
    if args.dry_run:
        for index, config in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] {config.to_dict()}")
        return
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] Running {config.selection_strategy} k={config.topk_blocks} warmup={config.warmup_rounds}")
        run(config)
    report_path = write_suite_report(args.suite, configs)
    print(f"Suite analysis report written to {report_path}")


if __name__ == "__main__":
    main()
