from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from experiment_config import UpstreamConfig, build_task_name, default_seed_list
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run


BASELINE_METHODS = ["random", "delta_norm", "hypernet_v6", "adaptive_v6"]
AGENT_METHODS = ["agent_rule_v7", "agent_bandit_v7", "agent_policy_v7"]
OPTIONAL_AGENT_METHODS = ["agent_llm_planner_v7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V7 Agentic Federated RAG experiment suites.")
    parser.add_argument(
        "--suite",
        default="v7_main",
        choices=[
            "smoke",
            "v7_main",
            "v7_budget_aligned",
            "v7_heterogeneity",
            "v7_hardquery",
            "v7_ablation_signal",
            "v7_ablation_agent_level",
            "v7_cost_efficiency",
            "v7_explain",
            "all_v7",
        ],
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v7")
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
        suffix="v7",
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


def method_config(base: UpstreamConfig, seed: int, method: str, **overrides) -> UpstreamConfig:
    common = {
        "seed": seed,
        "method_name": method,
        "topk_blocks": 3,
        "warmup_rounds": 1,
        "score_mode": "downstream_value",
        "budget_mode": "fixed",
        "hard_budget_only": True,
    }
    if method == "random":
        common.update(selection_strategy="random", agent_profile="baseline_random", score_mode="importance")
    elif method == "delta_norm":
        common.update(selection_strategy="delta_norm", agent_profile="baseline_delta_norm", score_mode="importance")
    elif method == "hypernet_v6":
        common.update(selection_strategy="hypernet_v6", agent_profile="baseline_hypernet_v6")
    elif method == "adaptive_v6":
        common.update(selection_strategy="hypernet_v6", agent_profile="baseline_adaptive_v6", budget_mode="adaptive_v6")
    elif method == "agent_rule_v7":
        common.update(selection_strategy="hypernet_v6", agent_profile="rule_memory_hardquery")
    elif method == "agent_bandit_v7":
        common.update(selection_strategy="hypernet_v6", agent_profile="bandit_ucb_memory", history_window=7)
    elif method == "agent_policy_v7":
        common.update(selection_strategy="hypernet_v6", agent_profile="policy_feature_selector", layerwise_budget=True)
    elif method == "agent_llm_planner_v7":
        common.update(selection_strategy="hypernet_v6", agent_profile="llm_planner_overlay", layerwise_budget=True)
    elif method == "full_upload":
        common.update(selection_strategy="full", agent_profile="baseline_full_upload", topk_blocks=0, score_mode="importance")
    else:
        raise ValueError(f"Unknown V7 method: {method}")
    common.update(overrides)
    return replace(base, **common)


def smoke_suite(base: UpstreamConfig, _seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [method_config(replace(base, suite_tag="smoke", num_rounds=1), 0, "agent_rule_v7", warmup_rounds=0)]


def v7_main_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINE_METHODS + AGENT_METHODS
    return [method_config(base, seed, method) for seed in seeds for method in methods]


def v7_budget_aligned_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINE_METHODS + AGENT_METHODS
    strict = {
        "budget_mode": "fixed",
        "adaptive_expand_threshold": 0.82,
        "utility_expand_threshold": 1.38,
        "hard_client_threshold": 0.70,
        "hard_client_bonus_topk": 0,
        "hard_budget_only": True,
    }
    configs = []
    for seed in seeds:
        for method in methods:
            extra = dict(strict)
            if method == "adaptive_v6":
                extra["agent_profile"] = "baseline_adaptive_v6_fixed_payload"
            configs.append(method_config(base, seed, method, **extra))
    return configs


def v7_heterogeneity_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = ["random", "delta_norm", "adaptive_v6"] + AGENT_METHODS
    configs: list[UpstreamConfig] = []
    for alpha in (0.5, 0.3, 0.1, 0.05):
        scenario_task = build_task_name(
            num_clients=base.num_clients,
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
            imbalance=base.imbalance,
            task_seed=base.task_seed,
            suffix="v7",
        )
        scenario_base = replace(
            base,
            task_name=scenario_task,
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
        )
        for seed in seeds:
            for method in methods:
                configs.append(method_config(scenario_base, seed, method, hard_client_threshold=0.68 if alpha >= 0.1 else 0.62))
    return configs


def v7_hardquery_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = ["hypernet_v6", "adaptive_v6"] + AGENT_METHODS
    return [
        method_config(base, seed, method, use_hard_query_weighting=True, hard_query_scale=1.5)
        for seed in seeds
        for method in methods
    ]


def v7_ablation_signal_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    variants = [
        ("full_agent", {}),
        ("no_downstream_proxy", {"score_mode": "value"}),
        ("no_hard_query", {"use_hard_query_weighting": False}),
        ("no_client_rarity", {"use_client_embedding": False}),
        ("no_memory", {"use_history_features": False, "use_utility_memory": False}),
        ("no_redundancy_penalty", {"use_block_embedding": False}),
        ("no_instability_penalty", {"history_window": 1}),
        ("current_round_only", {"history_window": 1, "use_utility_memory": False}),
    ]
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        for method in ("agent_rule_v7", "agent_policy_v7"):
            for variant, extra in variants:
                configs.append(method_config(base, seed, method, agent_profile=f"{method}:{variant}", **extra))
    return configs


def v7_ablation_agent_level_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    levels = [
        ("passive_selector", "delta_norm", {}),
        ("reactive_agent", "agent_rule_v7", {"history_window": 1, "use_utility_memory": False}),
        ("memory_agent", "agent_rule_v7", {}),
        ("planning_agent", "agent_policy_v7", {}),
        ("llm_planner_agent", "agent_llm_planner_v7", {}),
    ]
    return [
        method_config(base, seed, method, method_name=level, agent_profile=level, **extra)
        for seed in seeds
        for level, method, extra in levels
    ]


def v7_cost_efficiency_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = AGENT_METHODS + OPTIONAL_AGENT_METHODS
    return [method_config(base, seed, method) for seed in seeds for method in methods]


def v7_explain_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = AGENT_METHODS + OPTIONAL_AGENT_METHODS
    return [
        method_config(base, seed, method, layerwise_budget=True, use_hard_query_weighting=True)
        for seed in seeds
        for method in methods
    ]


def deduplicate(configs: list[UpstreamConfig]) -> list[UpstreamConfig]:
    dedup: dict[tuple, UpstreamConfig] = {}
    for config in configs:
        key = (
            config.suite_tag,
            config.task_name,
            config.seed,
            config.method_label,
            config.agent_profile,
            config.selection_strategy,
            config.topk_blocks,
            config.warmup_rounds,
            config.score_mode,
            config.budget_mode,
            config.use_client_embedding,
            config.use_history_features,
            config.use_block_embedding,
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
    builders = {
        "smoke": smoke_suite,
        "v7_main": v7_main_suite,
        "v7_budget_aligned": v7_budget_aligned_suite,
        "v7_heterogeneity": v7_heterogeneity_suite,
        "v7_hardquery": v7_hardquery_suite,
        "v7_ablation_signal": v7_ablation_signal_suite,
        "v7_ablation_agent_level": v7_ablation_agent_level_suite,
        "v7_cost_efficiency": v7_cost_efficiency_suite,
        "v7_explain": v7_explain_suite,
    }
    if args.suite == "all_v7":
        suite_configs: list[UpstreamConfig] = []
        for suite in [
            "v7_main",
            "v7_budget_aligned",
            "v7_heterogeneity",
            "v7_hardquery",
            "v7_ablation_signal",
            "v7_ablation_agent_level",
            "v7_cost_efficiency",
            "v7_explain",
        ]:
            suite_configs.extend(builders[suite](replace(base, suite_tag=suite), seeds))
        return deduplicate(suite_configs)
    return deduplicate(builders[args.suite](base, seeds))


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
            print(f"[{index}/{total}] Skipping completed {config.method_label} seed={config.seed} task={config.task_name}")
            results.append({"status": "skipped", "output_dir": str(config.output_dir), "config": config.to_dict()})
            continue
        print(
            f"[{index}/{total}] Running {config.method_label} "
            f"seed={config.seed} topk={config.topk_blocks} warmup={config.warmup_rounds} task={config.task_name}"
        )
        results.append(run(config))

    write_json(manifest_root / "suite_results.json", results)
    report_path = write_suite_report(args.suite, suite_configs)
    print(f"V7 suite analysis report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
