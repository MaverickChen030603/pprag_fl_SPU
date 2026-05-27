from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from experiment_config import UpstreamConfig, build_task_name, default_seed_list
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V6 experiment suites.")
    parser.add_argument(
        "--suite",
        default="v6_main",
        choices=[
            "smoke",
            "v6_main",
            "v6_budget_aligned",
            "v6_heterogeneity",
            "v6_hardquery",
            "v6_ablation_signal",
            "v6_ablation_budget",
            "v6_explain",
            "all_v6",
        ],
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v6")
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
        suffix="v6",
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


def smoke_suite(base: UpstreamConfig, _seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [
        replace(
            base,
            suite_tag="smoke",
            seed=0,
            num_rounds=1,
            selection_strategy="hypernet_v6",
            topk_blocks=3,
            warmup_rounds=0,
            score_mode="downstream_value",
            budget_mode="fixed",
        )
    ]


def v6_main_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="full", topk_blocks=0, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="random", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="delta_norm", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive"),
                replace(
                    base,
                    seed=seed,
                    selection_strategy="hypernet_v6",
                    topk_blocks=3,
                    warmup_rounds=1,
                    score_mode="downstream_value",
                    budget_mode="fixed",
                    hard_budget_only=True,
                ),
            ]
        )
    return configs


def v6_budget_aligned_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    method_grid = [
        ("random", 3, "importance", "fixed", {}),
        ("delta_norm", 3, "importance", "fixed", {}),
        ("hypernet_v3", 3, "value", "adaptive", {}),
        (
            "hypernet_v6",
            3,
            "downstream_value",
            "fixed",
            {
                "adaptive_expand_threshold": 0.82,
                "utility_expand_threshold": 1.38,
                "hard_client_threshold": 0.70,
                "hard_client_bonus_topk": 0,
                "hard_budget_only": True,
            },
        ),
    ]
    for seed in seeds:
        for strategy, topk, score_mode, budget_mode, extra in method_grid:
            configs.append(
                replace(
                    base,
                    suite_tag="v6_budget_aligned",
                    seed=seed,
                    selection_strategy=strategy,
                    topk_blocks=topk,
                    warmup_rounds=1,
                    score_mode=score_mode,
                    budget_mode=budget_mode,
                    use_history_features=strategy.startswith("hypernet"),
                    **extra,
                )
            )
    return configs


def v6_heterogeneity_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for alpha in (0.5, 0.3, 0.1, 0.05):
        scenario_task = build_task_name(
            num_clients=base.num_clients,
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
            imbalance=base.imbalance,
            task_seed=base.task_seed,
            suffix="v6",
        )
        for seed in seeds:
            configs.extend(
                [
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="random", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="delta_norm", topk_blocks=3, warmup_rounds=1, score_mode="importance", budget_mode="fixed"),
                    replace(base, seed=seed, task_name=scenario_task, partitioner_name="DirichletPartitioner", dirichlet_alpha=alpha, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive"),
                    replace(
                        base,
                        seed=seed,
                        task_name=scenario_task,
                        partitioner_name="DirichletPartitioner",
                        dirichlet_alpha=alpha,
                        selection_strategy="hypernet_v6",
                        topk_blocks=3,
                        warmup_rounds=1,
                        score_mode="downstream_value",
                        budget_mode="fixed" if alpha >= 0.1 else "adaptive_v6",
                        hard_budget_only=True,
                    ),
                ]
            )
    return configs


def v6_hardquery_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v3", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="adaptive", use_hard_query_weighting=False),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", use_hard_query_weighting=True, hard_query_scale=1.25),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v6", use_hard_query_weighting=True, hard_query_scale=1.5, adaptive_expand_threshold=0.80, utility_expand_threshold=1.40),
            ]
        )
    return configs


def v6_ablation_signal_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="value", budget_mode="fixed"),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", use_hard_query_weighting=False),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", use_utility_memory=False),
            ]
        )
    return configs


def v6_ablation_budget_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        configs.extend(
            [
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", layerwise_budget=False),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="adaptive_v6", layerwise_budget=False),
                replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", layerwise_budget=True),
            ]
        )
    return configs


def v6_explain_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [
        replace(base, seed=seed, selection_strategy="hypernet_v6", topk_blocks=3, warmup_rounds=1, score_mode="downstream_value", budget_mode="fixed", layerwise_budget=True)
        for seed in seeds
    ]


def deduplicate(configs: list[UpstreamConfig]) -> list[UpstreamConfig]:
    dedup: dict[tuple, UpstreamConfig] = {}
    for config in configs:
        key = (
            config.suite_tag,
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
            config.adaptive_expand_threshold,
            config.utility_expand_threshold,
            config.hard_client_threshold,
            config.hard_budget_only,
        )
        dedup[key] = config
    return list(dedup.values())


def is_completed(config: UpstreamConfig) -> bool:
    final_artifacts = config.output_dir / "final_artifacts.json"
    if final_artifacts.exists():
        return True
    run_metadata = config.output_dir / "run_metadata.json"
    if not run_metadata.exists():
        return False
    try:
        import json

        metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
    except Exception:
        return False
    completed_rounds = int(metadata.get("completed_rounds", 0) or 0)
    target_rounds = int(metadata.get("target_rounds", config.num_rounds) or config.num_rounds)
    return completed_rounds >= target_rounds


def build_suite(args: argparse.Namespace) -> list[UpstreamConfig]:
    seeds = parse_seed_list(args.seed_list)
    base = base_config(args)
    suite_configs: list[UpstreamConfig] = []
    if args.suite == "all_v6":
        suite_configs.extend(v6_main_suite(replace(base, suite_tag="v6_main"), seeds))
        suite_configs.extend(v6_budget_aligned_suite(replace(base, suite_tag="v6_budget_aligned"), seeds))
        suite_configs.extend(v6_heterogeneity_suite(replace(base, suite_tag="v6_heterogeneity"), seeds))
        suite_configs.extend(v6_hardquery_suite(replace(base, suite_tag="v6_hardquery"), seeds))
        suite_configs.extend(v6_ablation_signal_suite(replace(base, suite_tag="v6_ablation_signal"), seeds))
        suite_configs.extend(v6_ablation_budget_suite(replace(base, suite_tag="v6_ablation_budget"), seeds))
        suite_configs.extend(v6_explain_suite(replace(base, suite_tag="v6_explain"), seeds))
    else:
        builders = {
            "smoke": smoke_suite,
            "v6_main": v6_main_suite,
            "v6_budget_aligned": v6_budget_aligned_suite,
            "v6_heterogeneity": v6_heterogeneity_suite,
            "v6_hardquery": v6_hardquery_suite,
            "v6_ablation_signal": v6_ablation_signal_suite,
            "v6_ablation_budget": v6_ablation_budget_suite,
            "v6_explain": v6_explain_suite,
        }
        suite_configs.extend(builders[args.suite](base, seeds))
    return deduplicate(suite_configs)


def main() -> int:
    args = parse_args()
    suite_configs = build_suite(args)
    manifest_root = ensure_dir(base_config(args).suite_root)
    write_json(manifest_root / "suite_manifest.json", [config.to_dict() for config in suite_configs])
    if args.dry_run:
        for config in suite_configs:
            print(config.output_dir)
        return 0

    results = []
    total = len(suite_configs)
    for index, config in enumerate(suite_configs, start=1):
        if is_completed(config):
            print(f"[{index}/{total}] Skipping completed {config.selection_strategy} seed={config.seed} task={config.task_name}")
            results.append({"status": "skipped", "output_dir": str(config.output_dir), "config": config.to_dict()})
            continue
        print(
            f"[{index}/{total}] Running {config.selection_strategy} "
            f"seed={config.seed} topk={config.topk_blocks} warmup={config.warmup_rounds} task={config.task_name}"
        )
        result = run(config)
        results.append(result)

    write_json(manifest_root / "suite_results.json", results)
    report_path = write_suite_report(args.suite, suite_configs)
    print(f"V6 suite analysis report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
