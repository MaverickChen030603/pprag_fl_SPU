from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from experiment_config import UpstreamConfig, build_task_name, default_seed_list
from metrics import ensure_dir, write_json
from report_generator import write_suite_report
from run_upstream import run

BASELINES = ["hypernet_v6", "adaptive_v6"]
CORE_AGENTS = []
AGENTS = ["agent_tail_v7hp2", "agent_memory_v7hp2", "agent_tail_reader_v7hp2", "agent_memory_reader_v7hp2"]
ORACLE = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V7-HP3 HotpotQA multihop agent-gap experiments.")
    parser.add_argument(
        "--suite",
        default="hp2_multihop_hard",
        choices=[
            "smoke",
            "hp2_multihop_hard",
            "hp2_rare_bridge_tail",
            "hp2_budget_aligned",
            "hp2_ablation_signal",
            "hp2_reader_aligned",
            "hp3_reset_hard",
            "all_hp2",
        ],
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v7_hp3")
    parser.add_argument("--partitioner", default="DirichletPartitioner", choices=["IDPartitioner", "DirichletPartitioner"])
    parser.add_argument("--dir-alpha", type=float, default=0.3)
    parser.add_argument("--imbalance", type=float, default=0.0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--seed-list", default="0,1,2")
    parser.add_argument("--rawdata-path", default=str(Path(__file__).resolve().parents[1] / "FedE" / "select_data_hotpot_train_5000.json"))
    parser.add_argument("--rag-dataset", default="hotpot_qa")
    parser.add_argument("--rag-hotpot-split", default="validation")
    parser.add_argument("--rag-hotpot-max-examples", type=int, default=1000)
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
        suffix="v7hp3",
    )
    return UpstreamConfig(
        experiment_name=args.experiment_name,
        suite_tag=args.suite,
        task_name=task_name,
        rawdata_path=str(Path(args.rawdata_path).expanduser().resolve()),
        rag_dataset=args.rag_dataset,
        rag_hotpot_split=args.rag_hotpot_split,
        rag_hotpot_max_examples=args.rag_hotpot_max_examples,
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
        "hard_client_bonus_topk": 0,
        "use_hard_query_weighting": True,
        "use_utility_memory": True,
    }
    if method == "hypernet_v6":
        common.update(selection_strategy="hypernet_v6", agent_profile="hp2_baseline_hypernet_v6")
    elif method == "adaptive_v6":
        common.update(selection_strategy="hypernet_v6", agent_profile="hp2_baseline_adaptive_v6", budget_mode="adaptive_v6")
    elif method == "agent_rule_v7":
        common.update(selection_strategy="agent_rule_v7", agent_profile="hp2_rule_memory_hardquery", agent_strategy_mode="hard_query_focused")
    elif method == "agent_bandit_v7":
        common.update(selection_strategy="agent_bandit_v7", agent_profile="hp2_bandit_ucb_memory", history_window=7, agent_strategy_mode="stability_focused")
    elif method == "agent_tail_v7hp2":
        common.update(selection_strategy="agent_tail_v7hp2", agent_profile="hp2_multihop_tail_agent", hard_client_bonus_topk=1)
    elif method == "agent_memory_v7hp2":
        common.update(selection_strategy="agent_memory_v7hp2", agent_profile="hp2_memory_bridge_agent", history_window=7)
    elif method == "agent_tail_reader_v7hp2":
        common.update(selection_strategy="agent_tail_v7hp2", agent_profile="hp2_reader_tail_agent", hard_client_bonus_topk=1, use_reader_feedback=True, reader_feedback_weight=0.90, reader_feedback_scale=2.50, reader_feedback_mode="step", reader_positive_reward=10.0, reader_negative_reward=-5.0, reader_reward_top_fraction=0.25)
    elif method == "agent_memory_reader_v7hp2":
        common.update(selection_strategy="agent_memory_v7hp2", agent_profile="hp2_reader_memory_agent", history_window=7, use_reader_feedback=True, reader_feedback_weight=0.90, reader_feedback_scale=2.50, reader_feedback_mode="step", reader_positive_reward=10.0, reader_negative_reward=-5.0, reader_reward_top_fraction=0.25)
    elif method == "agent_oracle_v7hp2":
        common.update(selection_strategy="agent_oracle_v7hp2", agent_profile="hp2_oracle_multihop_upperbound", hard_client_bonus_topk=2)
    else:
        raise ValueError(f"Unknown HP3 method: {method}")
    common.update(overrides)
    return replace(base, **common)


def smoke_suite(base: UpstreamConfig, _seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [method_config(replace(base, suite_tag="smoke", num_rounds=1), 0, "agent_tail_v7hp2", warmup_rounds=0)]


def hp2_multihop_hard_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS + ORACLE
    return [
        method_config(
            replace(base, suite_tag="hp2_multihop_hard"),
            seed,
            method,
            hard_query_scale=8.0,
            hard_client_threshold=0.32,
            hard_client_bonus_topk=1 if method.startswith("agent_") else 0,
            adaptive_expand_threshold=0.90,
            utility_expand_threshold=1.70,
        )
        for seed in seeds
        for method in methods
    ]


def hp2_rare_bridge_tail_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS + ORACLE
    configs: list[UpstreamConfig] = []
    for alpha in (0.1, 0.05):
        scenario = replace(
            base,
            suite_tag="hp2_rare_bridge_tail",
            task_name=build_task_name(base.num_clients, "DirichletPartitioner", alpha, base.imbalance, base.task_seed, suffix="v7hp3"),
            partitioner_name="DirichletPartitioner",
            dirichlet_alpha=alpha,
        )
        for seed in seeds:
            for method in methods:
                configs.append(
                    method_config(
                        scenario,
                        seed,
                        method,
                        hard_query_scale=7.0,
                        hard_client_threshold=0.28,
                        hard_client_bonus_topk=1 if method.startswith("agent_") else 0,
                    )
                )
    return configs


def hp2_budget_aligned_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS
    scenario = replace(base, suite_tag="hp2_budget_aligned")
    return [
        method_config(
            scenario,
            seed,
            method,
            topk_blocks=3,
            hard_query_scale=6.0,
            hard_client_threshold=0.36,
            hard_client_bonus_topk=0,
            adaptive_expand_threshold=0.95,
            utility_expand_threshold=1.90,
        )
        for seed in seeds
        for method in methods
    ]


def hp2_reader_aligned_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS
    scenario = replace(base, suite_tag="hp2_reader_aligned")
    return [
        method_config(
            scenario,
            seed,
            method,
            topk_blocks=3,
            hard_query_scale=6.0,
            hard_client_threshold=0.36,
            hard_client_bonus_topk=0 if "reader" not in method else 1,
            adaptive_expand_threshold=0.95,
            utility_expand_threshold=1.90,
        )
        for seed in seeds
        for method in methods
    ]



def hp3_reset_hard_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = ["hypernet_v6", "adaptive_v6", "agent_tail_reader_v7hp2", "agent_memory_reader_v7hp2"]
    scenario = replace(base, suite_tag="hp3_reset_hard")
    return [
        method_config(
            scenario,
            seed,
            method,
            topk_blocks=3,
            hard_query_scale=10.0,
            hard_client_threshold=0.28,
            hard_client_bonus_topk=1 if method.startswith("agent_") else 0,
            adaptive_expand_threshold=0.90,
            utility_expand_threshold=1.45,
        )
        for seed in seeds
        for method in methods
    ]

def hp2_ablation_signal_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    variants = [
        ("full", {}),
        ("no_memory", {"use_history_features": False, "use_utility_memory": False}),
        ("no_hard_query", {"use_hard_query_weighting": False}),
        ("no_rarity", {"use_client_embedding": False}),
    ]
    scenario = replace(base, suite_tag="hp2_ablation_signal")
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        for variant, extra in variants:
            configs.append(
                method_config(
                    scenario,
                    seed,
                    "agent_tail_v7hp2",
                    agent_profile=f"agent_tail_v7hp2:{variant}",
                    hard_query_scale=7.0,
                    hard_client_threshold=0.32,
                    hard_client_bonus_topk=1,
                    **extra,
                )
            )
    return configs


def deduplicate(configs: list[UpstreamConfig]) -> list[UpstreamConfig]:
    dedup = {}
    for c in configs:
        key = (
            c.suite_tag,
            c.task_name,
            c.seed,
            c.method_label,
            c.agent_profile,
            c.selection_strategy,
            c.topk_blocks,
            c.warmup_rounds,
            c.score_mode,
            c.budget_mode,
            c.use_client_embedding,
            c.use_history_features,
            c.use_block_embedding,
            c.use_hard_query_weighting,
            c.hard_query_scale,
            c.use_utility_memory,
            c.hard_client_bonus_topk,
            c.rawdata_path,
        )
        dedup[key] = c
    return list(dedup.values())


def is_completed(config: UpstreamConfig) -> bool:
    return (config.output_dir / "final_artifacts.json").exists()


def build_suite(args: argparse.Namespace) -> list[UpstreamConfig]:
    seeds = parse_seed_list(args.seed_list)
    base = base_config(args)
    builders = {
        "smoke": smoke_suite,
        "hp2_multihop_hard": hp2_multihop_hard_suite,
        "hp2_rare_bridge_tail": hp2_rare_bridge_tail_suite,
        "hp2_budget_aligned": hp2_budget_aligned_suite,
        "hp2_reader_aligned": hp2_reader_aligned_suite,
        "hp2_ablation_signal": hp2_ablation_signal_suite,
        "hp3_reset_hard": hp3_reset_hard_suite,
    }
    if args.suite == "all_hp2":
        configs: list[UpstreamConfig] = []
        for suite in ["hp2_reader_aligned", "hp2_budget_aligned", "hp2_ablation_signal"]:
            configs.extend(builders[suite](replace(base, suite_tag=suite), seeds))
        return deduplicate(configs)
    return deduplicate(builders[args.suite](base, seeds))


def main() -> int:
    args = parse_args()
    configs = build_suite(args)
    manifest_root = ensure_dir(base_config(args).suite_root)
    write_json(manifest_root / "suite_manifest.json", [config.to_dict() for config in configs])
    if args.dry_run:
        for config in configs:
            print(config.output_dir)
        return 0
    results = []
    total = len(configs)
    for index, config in enumerate(configs, start=1):
        if is_completed(config):
            print(f"[{index}/{total}] Skipping completed {config.method_label} seed={config.seed} task={config.task_name}")
            results.append({"status": "skipped", "output_dir": str(config.output_dir), "config": config.to_dict()})
            continue
        print(f"[{index}/{total}] Running {config.method_label} seed={config.seed} task={config.task_name}")
        results.append(run(config))
    write_json(manifest_root / "suite_results.json", results)
    report_path = write_suite_report(args.suite, configs)
    print(f"V7-HP3 suite analysis report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
