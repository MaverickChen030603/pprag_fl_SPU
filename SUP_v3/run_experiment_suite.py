from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, List

from experiment_config import UpstreamConfig, build_task_name
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUP_v3 experiment suites.")
    parser.add_argument(
        "--suite",
        default="stability",
        choices=["smoke", "stability", "budget", "heterogeneity", "topk", "warmup", "all_v2"],
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_sup_v3")
    parser.add_argument("--partitioner", default="DirichletPartitioner", choices=["IDPartitioner", "DirichletPartitioner"])
    parser.add_argument("--dir-alpha", type=float, default=0.3)
    parser.add_argument("--imbalance", type=float, default=0.0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--seed-list", default="0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def base_config(args: argparse.Namespace) -> UpstreamConfig:
    task_name = build_task_name(
        num_clients=args.clients,
        partitioner_name=args.partitioner,
        dirichlet_alpha=args.dir_alpha,
        imbalance=args.imbalance,
        task_seed=args.task_seed,
    )
    return UpstreamConfig(
        experiment_name=args.experiment_name,
        suite_tag=args.suite,
        task_name=task_name,
        num_clients=args.clients,
        num_rounds=args.rounds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gpu=args.gpu,
        partitioner_name=args.partitioner,
        dirichlet_alpha=args.dir_alpha,
        imbalance=args.imbalance,
        task_seed=args.task_seed,
    )


def smoke_suite(base: UpstreamConfig) -> list[UpstreamConfig]:
    return [
        replace(base, seed=0, num_rounds=1, selection_strategy="hypernet", topk_blocks=3, warmup_rounds=0),
    ]


def stability_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="full", topk_blocks=0, warmup_rounds=1),
                replace(base, seed=seed, selection_strategy="random", topk_blocks=3, warmup_rounds=1),
                replace(base, seed=seed, selection_strategy="delta_norm", topk_blocks=3, warmup_rounds=1),
                replace(base, seed=seed, selection_strategy="hypernet", topk_blocks=3, warmup_rounds=1),
            ]
        )
    return configs


def budget_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    method_grid = [
        ("random", 1, 1),
        ("random", 3, 1),
        ("delta_norm", 1, 1),
        ("delta_norm", 3, 1),
        ("hypernet", 1, 1),
        ("hypernet", 3, 1),
        ("hypernet", 5, 1),
    ]
    for seed in seeds:
        for strategy, topk, warmup in method_grid:
            configs.append(replace(base, seed=seed, selection_strategy=strategy, topk_blocks=topk, warmup_rounds=warmup))
    return configs


def heterogeneity_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for alpha in (0.5, 0.3, 0.1):
        scenario_task = build_task_name(
            num_clients=base.num_clients,
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
            imbalance=base.imbalance,
            task_seed=base.task_seed,
        )
        for seed in seeds:
            configs.extend(
                [
                    replace(
                        base,
                        seed=seed,
                        task_name=scenario_task,
                        partitioner_name="DirichletPartitioner",
                        dirichlet_alpha=alpha,
                        selection_strategy="random",
                        topk_blocks=3,
                        warmup_rounds=1,
                    ),
                    replace(
                        base,
                        seed=seed,
                        task_name=scenario_task,
                        partitioner_name="DirichletPartitioner",
                        dirichlet_alpha=alpha,
                        selection_strategy="delta_norm",
                        topk_blocks=3,
                        warmup_rounds=1,
                    ),
                    replace(
                        base,
                        seed=seed,
                        task_name=scenario_task,
                        partitioner_name="DirichletPartitioner",
                        dirichlet_alpha=alpha,
                        selection_strategy="hypernet",
                        topk_blocks=3,
                        warmup_rounds=1,
                    ),
                ]
            )
    return configs


def topk_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [
        replace(base, seed=seed, selection_strategy="hypernet", topk_blocks=topk, warmup_rounds=1)
        for seed in seeds
        for topk in (1, 3, 5, 7)
    ]


def warmup_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [
        replace(base, seed=seed, selection_strategy="hypernet", topk_blocks=3, warmup_rounds=warmup)
        for seed in seeds
        for warmup in (0, 1, 2, 5)
    ]


def deduplicate(configs: list[UpstreamConfig]) -> list[UpstreamConfig]:
    dedup: dict[tuple, UpstreamConfig] = {}
    for config in configs:
        key = (
            config.task_name,
            config.seed,
            config.selection_strategy,
            config.topk_blocks,
            config.warmup_rounds,
            config.estimate_encryption,
        )
        dedup[key] = config
    return list(dedup.values())


def build_suite(args: argparse.Namespace) -> List[UpstreamConfig]:
    base = base_config(args)
    seeds = parse_seed_list(args.seed_list)
    if args.suite == "smoke":
        return smoke_suite(base)
    if args.suite == "stability":
        return stability_suite(base, seeds)
    if args.suite == "budget":
        return budget_suite(base, seeds)
    if args.suite == "heterogeneity":
        return heterogeneity_suite(base, seeds)
    if args.suite == "topk":
        return topk_suite(base, seeds)
    if args.suite == "warmup":
        return warmup_suite(base, seeds)
    configs: list[UpstreamConfig] = []
    configs.extend(stability_suite(base, seeds))
    configs.extend(budget_suite(base, seeds))
    configs.extend(heterogeneity_suite(base, seeds))
    configs.extend(topk_suite(base, seeds))
    configs.extend(warmup_suite(base, seeds))
    return deduplicate(configs)


def main() -> None:
    args = parse_args()
    configs = build_suite(args)
    manifest = [config.to_dict() for config in configs]
    if configs:
        ensure_dir(configs[0].suite_root)
        write_json(configs[0].suite_root / f"{args.suite}_manifest.json", manifest)
    if args.dry_run:
        for index, config in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] {config.to_dict()}")
        return
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] Running {config.selection_strategy} "
            f"seed={config.seed} topk={config.topk_blocks} warmup={config.warmup_rounds} task={config.task_name}"
        )
        run(config)
    report_path = write_suite_report(args.suite, configs)
    print(f"Suite analysis report written to {report_path}")


if __name__ == "__main__":
    main()
