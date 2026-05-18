from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, List

from experiment_config import UpstreamConfig, build_task_name, default_seed_list
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 experiment suites.")
    parser.add_argument(
        "--suite",
        default="v4_main",
        choices=[
            "smoke",
            "v4_main",
            "v4_budget",
            "v4_heterogeneity",
            "v4_hardquery",
            "v4_ablation_signal",
            "v4_ablation_budget",
            "v4_explain",
            "all_v4",
        ],
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v4")
    parser.add_argument("--partitioner", default="DirichletPartitioner", choices=["IDPartitioner", "DirichletPartitioner"])
    parser.add_argument("--dir-alpha", type=float, default=0.3)
    parser.add_argument("--imbalance", type=float, default=0.0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--seed-list", default="0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    if not raw.strip():
        return default_seed_list()
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def base_config(args: argparse.Namespace) -> UpstreamConfig:
    task_name = build_task_name(
        num_clients=args.clients,
        partitioner_name=args.partitioner,
        dirichlet_alpha=args.dir_alpha,
        imbalance=args.imbalance,
        task_seed=args.task_seed,
        suffix="v4",
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
    return [replace(base, seed=0, num_rounds=1, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=0)]


def v4_main_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="full", topk_blocks=0, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="random", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="delta_norm", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive"),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4"),
            ]
        )
    return configs


def v4_budget_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    method_grid = [
        ("random", 1, "importance", "fixed"),
        ("random", 3, "importance", "fixed"),
        ("delta_norm", 1, "importance", "fixed"),
        ("delta_norm", 3, "importance", "fixed"),
        ("hypernet_v3", 1, "value", "adaptive"),
        ("hypernet_v3", 3, "value", "adaptive"),
        ("hypernet_v4", 1, "downstream_value", "adaptive_v4"),
        ("hypernet_v4", 3, "downstream_value", "adaptive_v4"),
        ("hypernet_v4", 5, "downstream_value", "adaptive_v4"),
    ]
    for seed in seeds:
        for strategy, topk, score_mode, budget_mode in method_grid:
            configs.append(
                replace(
                    base,
                    seed=seed,
                    selection_strategy=strategy,
                    topk_blocks=topk,
                    warmup_rounds=1,
                    score_mode=score_mode,
                    budget_mode=budget_mode,
                    use_history_features=strategy.startswith("hypernet"),
                )
            )
    return configs


def v4_heterogeneity_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for alpha in (0.5, 0.3, 0.1, 0.05):
        scenario_task = build_task_name(
            num_clients=base.num_clients,
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
            imbalance=base.imbalance,
            task_seed=base.task_seed,
            suffix="v4",
        )
        for seed in seeds:
            configs.extend(
                [
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="random", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="delta_norm", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive"),
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4"),
                ]
            )
    return configs


def v4_hardquery_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive", use_hard_query_weighting=False),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", use_hard_query_weighting=True, hard_query_scale=1.0),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", use_hard_query_weighting=True, hard_query_scale=1.5),
            ]
        )
    return configs


def v4_ablation_signal_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4"),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive_v4"),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", use_hard_query_weighting=False),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", use_utility_memory=False),
            ]
        )
    return configs


def v4_ablation_budget_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", layerwise_budget=False),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive", layerwise_budget=False),
                replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", layerwise_budget=False),
            ]
        )
    return configs


def v4_explain_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [
        replace(base, seed=seed, selection_strategy="hypernet_v4", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v4", layerwise_budget=True)
        for seed in seeds
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
            config.score_mode,
            config.budget_mode,
            config.use_client_embedding,
            config.use_history_features,
            config.use_hard_query_weighting,
            config.use_utility_memory,
            config.layerwise_budget,
        )
        dedup[key] = config
    return list(dedup.values())


def build_suite(args: argparse.Namespace) -> List[UpstreamConfig]:
    base = base_config(args)
    seeds = parse_seed_list(args.seed_list)
    if args.suite == "smoke":
        return smoke_suite(base)
    if args.suite == "v4_main":
        return v4_main_suite(base, seeds)
    if args.suite == "v4_budget":
        return v4_budget_suite(base, seeds)
    if args.suite == "v4_heterogeneity":
        return v4_heterogeneity_suite(base, seeds)
    if args.suite == "v4_hardquery":
        return v4_hardquery_suite(base, seeds)
    if args.suite == "v4_ablation_signal":
        return v4_ablation_signal_suite(base, seeds)
    if args.suite == "v4_ablation_budget":
        return v4_ablation_budget_suite(base, seeds)
    if args.suite == "v4_explain":
        return v4_explain_suite(base, seeds)
    configs: list[UpstreamConfig] = []
    configs.extend(v4_main_suite(base, seeds))
    configs.extend(v4_budget_suite(base, seeds))
    configs.extend(v4_heterogeneity_suite(base, seeds))
    configs.extend(v4_hardquery_suite(base, seeds))
    configs.extend(v4_ablation_signal_suite(base, seeds))
    configs.extend(v4_ablation_budget_suite(base, seeds))
    configs.extend(v4_explain_suite(base, seeds))
    return deduplicate(configs)


def is_completed(config: UpstreamConfig) -> bool:
    output_dir = config.output_dir
    if not (output_dir / "run_metadata.json").exists():
        return False
    if (output_dir / "final_artifacts.json").exists():
        return True
    round_log = output_dir / "round_logs.jsonl"
    if not round_log.exists():
        return False
    with round_log.open("r", encoding="utf-8") as handle:
        rounds = sum(1 for line in handle if line.strip())
    return rounds >= int(config.num_rounds)


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
        if is_completed(config):
            print(
                f"[{index}/{len(configs)}] Skipping completed {config.selection_strategy} seed={config.seed} "
                f"topk={config.topk_blocks} warmup={config.warmup_rounds} task={config.task_name} "
                f"score={config.score_mode} budget={config.budget_mode}"
            )
            continue
        print(
            f"[{index}/{len(configs)}] Running {config.selection_strategy} seed={config.seed} "
            f"topk={config.topk_blocks} warmup={config.warmup_rounds} task={config.task_name} "
            f"score={config.score_mode} budget={config.budget_mode}"
        )
        run(config)
    report_path = write_suite_report(args.suite, configs)
    print(f"V4 suite analysis report written to {report_path}")


if __name__ == "__main__":
    main()
