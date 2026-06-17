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
CORE_AGENTS = ["agent_rule_v7", "agent_bandit_v7"]
V7AGENT2_METHODS = ["agent_rule_v7_no_prior", "agent_rule_v7_no_coverage", "agent_rule_v7_no_memory", "agent_bandit_v7_early", "agent_rule_v7_dynamic"]
MAIN_AGENTS = ["agent_rule_v7"]
AGENTS = CORE_AGENTS + ["agent_tail_v7hp1", "agent_memory_v7hp1"]
ORACLE = ["agent_oracle_v7hp1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated V7-agent-BSP Bandit Slot Planning experiments.")
    parser.add_argument(
        "--suite",
        default="hp1_multihop_hard",
        choices=[
            "smoke",
            "hp1_multihop_hard",
            "hp1_rare_bridge_tail",
            "hp1_budget_aligned",
            "hp1_ablation_signal",
            "all_hp1",
            "v7agentpm_ablation",
            "v7agentpm_bandit",
            "v7agentpm_dynamic",
            "v7pm_all",
            "v7pm_bandit_slot",
            "v7pm_dynamic_ablation",
            "v7pm_memory_ablation",
            "v7pm_main",
            "v7bsp_main",
            "v7bsp_bsp_methods",
            "v7bsp_memory_ablation",
            "v7bsp_all",
        ],
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name", default="pprag_fl_v7agentbsp")
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
        suffix="v7hp1",
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
        common.update(selection_strategy="hypernet_v6", agent_profile="hp1_baseline_hypernet_v6")
    elif method == "adaptive_v6":
        common.update(selection_strategy="hypernet_v6", agent_profile="hp1_baseline_adaptive_v6", budget_mode="adaptive_v6")
    elif method == "agent_rule_v7":
        common.update(selection_strategy="agent_rule_v7", agent_profile="v7agent_rule_early_hardquery", agent_strategy_mode="hard_query_focused", history_window=7)
    elif method == "agent_bandit_v7":
        common.update(selection_strategy="agent_bandit_v7", agent_profile="hp1_bandit_ucb_memory", history_window=7, agent_strategy_mode="stability_focused")
    elif method == "agent_rule_v7_no_prior":
        common.update(selection_strategy="agent_rule_v7", agent_profile="v7agentpm_no_prior", agent_strategy_mode="hard_query_focused", history_window=7, use_early_prior=False, use_coverage_replace=False)
    elif method == "agent_rule_v7_no_coverage":
        common.update(selection_strategy="agent_rule_v7", agent_profile="v7agentpm_no_coverage", agent_strategy_mode="hard_query_focused", history_window=7, use_early_prior=True, use_coverage_replace=False)
    elif method == "agent_rule_v7_no_memory":
        common.update(selection_strategy="agent_rule_v7", agent_profile="v7agentpm_no_memory", agent_strategy_mode="hard_query_focused", history_window=7, use_early_prior=True, use_coverage_replace=True, use_memory_ema=False)
    elif method == "agent_bandit_v7_early":
        common.update(selection_strategy="agent_bandit_v7", agent_profile="v7agentpm_bandit_early", history_window=7, agent_strategy_mode="hard_query_focused", early_coverage_weight=0.3, use_early_prior=True, use_coverage_replace=True)
    elif method == "agent_rule_v7_dynamic":
        common.update(selection_strategy="agent_rule_v7", agent_profile="v7agentpm_rule_dynamic", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True)

    elif method == "agent_pm_dynamic_full":
        common.update(selection_strategy="agent_pm_dynamic_full", agent_profile="v7pm_planning_memory_full", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_pm_dynamic_no_memory":
        common.update(selection_strategy="agent_pm_dynamic_no_memory", agent_profile="v7pm_no_memory", agent_strategy_mode="hard_query_focused", history_window=1, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=False, use_pm_failure_memory=False, use_pm_rarity_memory=False, use_pm_instability_penalty=False, use_bridge_guard=True)
    elif method == "agent_pm_dynamic_no_failure_memory":
        common.update(selection_strategy="agent_pm_dynamic_no_failure_memory", agent_profile="v7pm_no_failure_memory", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=False, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_pm_dynamic_no_rarity_memory":
        common.update(selection_strategy="agent_pm_dynamic_no_rarity_memory", agent_profile="v7pm_no_rarity_memory", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=False, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_pm_dynamic_no_instability_penalty":
        common.update(selection_strategy="agent_pm_dynamic_no_instability_penalty", agent_profile="v7pm_no_instability", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=False, use_bridge_guard=True)
    elif method == "agent_pm_dynamic_no_utility_ema":
        common.update(selection_strategy="agent_pm_dynamic_no_utility_ema", agent_profile="v7pm_no_utility_ema", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=False, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method.startswith("agent_fixed_slot_"):
        slot = int(method.rsplit("_", 1)[1])
        common.update(selection_strategy=method, agent_profile=f"v7pm_fixed_slot_{slot}", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=False, fixed_early_slots=slot, use_early_prior=True, use_coverage_replace=True, use_bridge_guard=True)
    elif method == "agent_dynamic_slot":
        common.update(selection_strategy="agent_dynamic_slot", agent_profile="v7pm_dynamic_slot", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_bridge_guard=True)
    elif method == "agent_dynamic_no_hardness":
        common.update(selection_strategy="agent_dynamic_no_hardness", agent_profile="v7pm_dynamic_no_hardness", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, disable_dynamic_hardness=True, use_bridge_guard=True)
    elif method == "agent_dynamic_no_rarity":
        common.update(selection_strategy="agent_dynamic_no_rarity", agent_profile="v7pm_dynamic_no_rarity", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_pm_rarity_memory=False, use_bridge_guard=True)
    elif method == "agent_dynamic_no_bridge_guard":
        common.update(selection_strategy="agent_dynamic_no_bridge_guard", agent_profile="v7pm_dynamic_no_bridge_guard", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_bridge_guard=False)
    elif method == "agent_pm_bandit_slot":
        common.update(selection_strategy="agent_pm_bandit_slot", agent_profile="v7pm_bandit_slot_ucb_proxy", agent_strategy_mode="hard_query_focused", history_window=9, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method in {"agent_bsp_bandit_strict", "agent_bsp_bandit_retrieval", "agent_bsp_bandit_reader"}:
        reward = method.rsplit("_", 1)[-1]
        common.update(selection_strategy=method, agent_profile=f"v7bsp_bandit_{reward}", agent_strategy_mode="hard_query_focused", history_window=7, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=False, use_pm_failure_memory=False, use_pm_rarity_memory=False, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method in {"agent_bsp_memory_bandit_strict", "agent_bsp_memory_bandit_retrieval", "agent_bsp_memory_bandit_reader"}:
        reward = method.rsplit("_", 1)[-1]
        common.update(selection_strategy=method, agent_profile=f"v7bsp_memory_bandit_{reward}", agent_strategy_mode="hard_query_focused", history_window=11, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_bsp_memory_bandit_no_failure_state":
        common.update(selection_strategy=method, agent_profile="v7bsp_no_failure_state", agent_strategy_mode="hard_query_focused", history_window=11, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=False, use_pm_rarity_memory=True, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_bsp_memory_bandit_no_rarity_state":
        common.update(selection_strategy=method, agent_profile="v7bsp_no_rarity_state", agent_strategy_mode="hard_query_focused", history_window=11, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=False, use_pm_instability_penalty=True, use_bridge_guard=True)
    elif method == "agent_bsp_memory_bandit_no_instability_state":
        common.update(selection_strategy=method, agent_profile="v7bsp_no_instability_state", agent_strategy_mode="hard_query_focused", history_window=11, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=True, use_pm_failure_memory=True, use_pm_rarity_memory=True, use_pm_instability_penalty=False, use_bridge_guard=True)
    elif method == "agent_bsp_memory_bandit_no_history_state":
        common.update(selection_strategy=method, agent_profile="v7bsp_no_history_state", agent_strategy_mode="hard_query_focused", history_window=1, use_history_features=False, use_utility_memory=False, use_dynamic_slots=True, use_early_prior=True, use_coverage_replace=True, use_memory_ema=False, use_pm_failure_memory=False, use_pm_rarity_memory=False, use_pm_instability_penalty=False, use_bridge_guard=True)
    elif method == "agent_tail_v7hp1":
        common.update(selection_strategy="agent_tail_v7hp1", agent_profile="hp1_multihop_tail_agent", hard_client_bonus_topk=1)
    elif method == "agent_memory_v7hp1":
        common.update(selection_strategy="agent_memory_v7hp1", agent_profile="hp1_memory_bridge_agent", history_window=7)
    elif method == "agent_oracle_v7hp1":
        common.update(selection_strategy="agent_oracle_v7hp1", agent_profile="hp1_oracle_multihop_upperbound", hard_client_bonus_topk=2)
    else:
        raise ValueError(f"Unknown HP1 method: {method}")
    common.update(overrides)
    return replace(base, **common)


def smoke_suite(base: UpstreamConfig, _seeds: Iterable[int]) -> list[UpstreamConfig]:
    return [method_config(replace(base, suite_tag="smoke", num_rounds=1), 0, "agent_tail_v7hp1", warmup_rounds=0)]


def hp1_multihop_hard_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS + ORACLE
    return [
        method_config(
            replace(base, suite_tag="hp1_multihop_hard"),
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


def hp1_rare_bridge_tail_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + AGENTS + ORACLE
    configs: list[UpstreamConfig] = []
    for alpha in (0.1, 0.05):
        scenario = replace(
            base,
            suite_tag="hp1_rare_bridge_tail",
            task_name=build_task_name(base.num_clients, "DirichletPartitioner", alpha, base.imbalance, base.task_seed, suffix="v7hp1"),
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


def hp1_budget_aligned_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    methods = BASELINES + MAIN_AGENTS + ["agent_bandit_v7"]
    scenario = replace(base, suite_tag="hp1_budget_aligned")
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


def hp1_ablation_signal_suite(base: UpstreamConfig, seeds: Iterable[int]) -> list[UpstreamConfig]:
    variants = [
        ("full", {}),
        ("no_memory", {"use_history_features": False, "use_utility_memory": False}),
        ("no_hard_query", {"use_hard_query_weighting": False}),
        ("no_rarity", {"use_client_embedding": False}),
    ]
    scenario = replace(base, suite_tag="hp1_ablation_signal")
    configs: list[UpstreamConfig] = []
    for seed in seeds:
        for variant, extra in variants:
            configs.append(
                method_config(
                    scenario,
                    seed,
                    "agent_tail_v7hp1",
                    agent_profile=f"agent_tail_v7hp1:{variant}",
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
            c.use_early_prior,
            c.use_coverage_replace,
            c.use_memory_ema,
            c.early_coverage_weight,
            c.use_dynamic_slots,
            c.difficulty_threshold_low,
            c.difficulty_threshold_high,
            c.fixed_early_slots,
            c.use_pm_failure_memory,
            c.use_pm_rarity_memory,
            c.use_pm_instability_penalty,
            c.use_bridge_guard,
            c.disable_dynamic_hardness,
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
        "hp1_multihop_hard": hp1_multihop_hard_suite,
        "hp1_rare_bridge_tail": hp1_rare_bridge_tail_suite,
        "hp1_budget_aligned": hp1_budget_aligned_suite,
        "hp1_ablation_signal": hp1_ablation_signal_suite,
        "v7agentpm_ablation": lambda b, seeds: [method_config(replace(b, suite_tag="v7agentpm_ablation"), seed, method) for seed in seeds for method in ["agent_rule_v7", "agent_rule_v7_no_prior", "agent_rule_v7_no_coverage", "agent_rule_v7_no_memory"]],
        "v7agentpm_bandit": lambda b, seeds: [method_config(replace(b, suite_tag="v7agentpm_bandit"), seed, "agent_bandit_v7_early", early_coverage_weight=w) for seed in seeds for w in (0.1, 0.3, 0.5)],
        "v7agentpm_dynamic": lambda b, seeds: [method_config(replace(b, suite_tag="v7agentpm_dynamic"), seed, "agent_rule_v7_dynamic") for seed in seeds],
        "v7pm_main": lambda b, seeds: [method_config(replace(b, suite_tag="v7pm_main"), seed, method) for seed in seeds for method in ["hypernet_v6", "adaptive_v6", "agent_rule_v7", "agent_rule_v7_dynamic", "agent_pm_dynamic_full"]],
        "v7pm_memory_ablation": lambda b, seeds: [method_config(replace(b, suite_tag="v7pm_memory_ablation"), seed, method) for seed in seeds for method in ["agent_pm_dynamic_full", "agent_pm_dynamic_no_memory", "agent_pm_dynamic_no_failure_memory", "agent_pm_dynamic_no_rarity_memory", "agent_pm_dynamic_no_instability_penalty", "agent_pm_dynamic_no_utility_ema"]],
        "v7pm_dynamic_ablation": lambda b, seeds: [method_config(replace(b, suite_tag="v7pm_dynamic_ablation"), seed, method) for seed in seeds for method in ["agent_fixed_slot_0", "agent_fixed_slot_1", "agent_fixed_slot_2", "agent_fixed_slot_3", "agent_dynamic_slot", "agent_dynamic_no_hardness", "agent_dynamic_no_rarity", "agent_dynamic_no_bridge_guard"]],
        "v7pm_bandit_slot": lambda b, seeds: [method_config(replace(b, suite_tag="v7pm_bandit_slot"), seed, method) for seed in seeds for method in ["agent_rule_v7_dynamic", "agent_pm_dynamic_full", "agent_pm_bandit_slot"]],
        "v7bsp_main": lambda b, seeds: [method_config(replace(b, suite_tag="v7bsp_main"), seed, method) for seed in seeds for method in ["hypernet_v6", "adaptive_v6", "agent_rule_v7", "agent_rule_v7_dynamic", "agent_pm_dynamic_full", "agent_pm_dynamic_no_memory", "agent_pm_bandit_slot", "agent_bsp_memory_bandit_strict", "agent_bsp_memory_bandit_retrieval", "agent_bsp_memory_bandit_reader"]],
        "v7bsp_bsp_methods": lambda b, seeds: [method_config(replace(b, suite_tag="v7bsp_bsp_methods"), seed, method) for seed in seeds for method in ["agent_bsp_bandit_strict", "agent_bsp_bandit_retrieval", "agent_bsp_bandit_reader", "agent_bsp_memory_bandit_strict", "agent_bsp_memory_bandit_retrieval", "agent_bsp_memory_bandit_reader"]],
        "v7bsp_memory_ablation": lambda b, seeds: [method_config(replace(b, suite_tag="v7bsp_memory_ablation"), seed, method) for seed in seeds for method in ["agent_bsp_memory_bandit_retrieval", "agent_bsp_memory_bandit_no_failure_state", "agent_bsp_memory_bandit_no_rarity_state", "agent_bsp_memory_bandit_no_instability_state", "agent_bsp_memory_bandit_no_history_state"]],
    }
    if args.suite == "v7pm_all":
        configs: list[UpstreamConfig] = []
        for suite in ["v7pm_main", "v7pm_memory_ablation", "v7pm_dynamic_ablation", "v7pm_bandit_slot"]:
            configs.extend(builders[suite](replace(base, suite_tag=suite), seeds))
        return deduplicate(configs)
    if args.suite == "v7bsp_all":
        configs: list[UpstreamConfig] = []
        for suite in ["v7bsp_main", "v7bsp_bsp_methods", "v7bsp_memory_ablation"]:
            configs.extend(builders[suite](replace(base, suite_tag=suite), seeds))
        return deduplicate(configs)
    if args.suite == "all_hp1":
        configs: list[UpstreamConfig] = []
        for suite in ["hp1_multihop_hard", "hp1_rare_bridge_tail", "hp1_budget_aligned", "hp1_ablation_signal"]:
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
    print(f"V7-agent-BSP suite analysis report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
