from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
FEDE_DIR = REPO_ROOT / "FedE"
RAGTEST_DIR = REPO_ROOT / "RAGTest"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
REPORT_ROOT = REPO_ROOT / "实验分析报告" / "V7-agent-BSP-DIAG"


def build_task_name(
    num_clients: int,
    partitioner_name: str,
    dirichlet_alpha: float,
    imbalance: float,
    task_seed: int,
    suffix: str = "v7",
) -> str:
    partitioner_tag = partitioner_name.lower()
    if partitioner_tag.startswith("dir"):
        alpha_tag = str(dirichlet_alpha).replace(".", "")
        imbalance_tag = str(imbalance).replace(".", "")
        return f"num{num_clients}_dir_a{alpha_tag}_imb{imbalance_tag}_ts{task_seed}_{suffix}"
    return f"num{num_clients}_id_ts{task_seed}_{suffix}"


@dataclass(frozen=True)
class UpstreamConfig:
    experiment_name: str = "pprag_fl_v7agentbspdiag"
    task_name: str = "num5_dir_a03_imb00_ts0_v7hp1"
    suite_tag: str = "hp1_multihop_hard"
    rawdata_path: str = str(FEDE_DIR / "select_data_hotpot_train_5000.json")
    rag_dataset: str = "hotpot_qa"
    rag_hotpot_split: str = "validation"
    rag_hotpot_max_examples: int = 1000
    num_clients: int = 5
    num_rounds: int = 25
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    gpu: int = 0
    seed: int = 0
    method_name: str = ""
    agent_profile: str = "baseline"
    selection_strategy: str = "hypernet_v6"
    topk_blocks: int = 3
    warmup_rounds: int = 1
    block_strategy: str = "bert"
    always_upload: tuple[str, ...] = ("pooler",)
    hn_lr: float = 5e-3
    hn_embedding_dim: int = 64
    hn_hidden_dim: int = 128
    estimate_encryption: bool = False
    encryption_expansion: float = 8.0
    partitioner_name: str = "DirichletPartitioner"
    dirichlet_alpha: float = 0.3
    imbalance: float = 0.0
    dirichlet_error_bar: float = 1e-3
    task_seed: int = 0
    overwrite_task: bool = False
    score_mode: str = "downstream_value"
    budget_mode: str = "adaptive_v6"
    history_window: int = 5
    use_client_embedding: bool = True
    use_history_features: bool = True
    use_block_embedding: bool = True
    adaptive_min_topk: int = 1
    adaptive_max_topk: int = 7
    adaptive_scale: float = 1.0
    layerwise_budget: bool = False
    use_hard_query_weighting: bool = True
    use_utility_memory: bool = True
    hard_query_scale: float = 1.0
    hard_client_threshold: float = 0.68
    hard_client_bonus_topk: int = 0
    adaptive_expand_threshold: float = 0.78
    adaptive_shrink_threshold: float = 0.48
    utility_expand_threshold: float = 1.30
    hard_budget_only: bool = True
    agent_strategy_mode: str = "stability_focused"
    use_early_prior: bool = True
    use_coverage_replace: bool = True
    use_memory_ema: bool = True
    early_coverage_weight: float = 0.0
    use_dynamic_slots: bool = False
    difficulty_threshold_low: float = 0.3
    difficulty_threshold_high: float = 0.7
    fixed_early_slots: int = -1
    use_pm_failure_memory: bool = True
    use_pm_rarity_memory: bool = True
    use_pm_instability_penalty: bool = True
    use_bridge_guard: bool = True
    disable_dynamic_hardness: bool = False

    def to_flgo_option(self) -> Dict:
        gpu_option = [] if self.gpu < 0 else [self.gpu]
        return {
            "num_rounds": self.num_rounds,
            "num_epochs": self.num_epochs,
            "gpu": gpu_option,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "method_name": self.method_label,
            "v7_agent_profile": self.agent_profile,
            "selection_strategy": self.selection_strategy,
            "selective_topk_blocks": self.topk_blocks,
            "selective_warmup_rounds": self.warmup_rounds,
            "selective_block_strategy": self.block_strategy,
            "selective_always_upload": list(self.always_upload),
            "hn_lr": self.hn_lr,
            "hn_embedding_dim": self.hn_embedding_dim,
            "hn_hidden_dim": self.hn_hidden_dim,
            "estimate_encryption": self.estimate_encryption,
            "encryption_expansion": self.encryption_expansion,
            "output_dir": str(self.output_dir),
            "suite_tag": self.suite_tag,
            "task_name": self.task_name,
            "rawdata_path": self.rawdata_path,
            "rag_dataset": self.rag_dataset,
            "rag_hotpot_split": self.rag_hotpot_split,
            "rag_hotpot_max_examples": self.rag_hotpot_max_examples,
            "score_mode": self.score_mode,
            "budget_mode": self.budget_mode,
            "history_window": self.history_window,
            "use_client_embedding": self.use_client_embedding,
            "use_history_features": self.use_history_features,
            "use_block_embedding": self.use_block_embedding,
            "adaptive_min_topk": self.adaptive_min_topk,
            "adaptive_max_topk": self.adaptive_max_topk,
            "adaptive_scale": self.adaptive_scale,
            "layerwise_budget": self.layerwise_budget,
            "use_hard_query_weighting": self.use_hard_query_weighting,
            "use_utility_memory": self.use_utility_memory,
            "hard_query_scale": self.hard_query_scale,
            "hard_client_threshold": self.hard_client_threshold,
            "hard_client_bonus_topk": self.hard_client_bonus_topk,
            "adaptive_expand_threshold": self.adaptive_expand_threshold,
            "adaptive_shrink_threshold": self.adaptive_shrink_threshold,
            "utility_expand_threshold": self.utility_expand_threshold,
            "hard_budget_only": self.hard_budget_only,
            "agent_strategy_mode": self.agent_strategy_mode,
            "use_early_prior": self.use_early_prior,
            "use_coverage_replace": self.use_coverage_replace,
            "use_memory_ema": self.use_memory_ema,
            "early_coverage_weight": self.early_coverage_weight,
            "use_dynamic_slots": self.use_dynamic_slots,
            "difficulty_threshold_low": self.difficulty_threshold_low,
            "difficulty_threshold_high": self.difficulty_threshold_high,
            "fixed_early_slots": self.fixed_early_slots,
            "use_pm_failure_memory": self.use_pm_failure_memory,
            "use_pm_rarity_memory": self.use_pm_rarity_memory,
            "use_pm_instability_penalty": self.use_pm_instability_penalty,
            "use_bridge_guard": self.use_bridge_guard,
            "disable_dynamic_hardness": self.disable_dynamic_hardness,
        }

    @property
    def task_path(self) -> Path:
        return FEDE_DIR / self.task_name

    @property
    def suite_root(self) -> Path:
        return OUTPUT_ROOT / self.experiment_name / self.suite_tag

    @property
    def output_dir(self) -> Path:
        enc_tag = "enc1" if self.estimate_encryption else "enc0"
        strategy_tag = self.method_label.replace("_", "-")
        score_tag = f"score-{self.score_mode}"
        budget_tag = f"budget-{self.budget_mode}"
        hist_tag = f"hist{self.history_window}" if self.use_history_features else "hist0"
        client_tag = "client1" if self.use_client_embedding else "client0"
        block_tag = "block1" if self.use_block_embedding else "block0"
        hard_tag = "hard1" if self.use_hard_query_weighting else "hard0"
        util_tag = "util1" if self.use_utility_memory else "util0"
        early_tag = "ep1" if self.use_early_prior else "ep0"
        cov_tag = "cov1" if self.use_coverage_replace else "cov0"
        mem_tag = "mem1" if self.use_memory_ema else "mem0"
        dyn_tag = "dyn1" if self.use_dynamic_slots else "dyn0"
        ecw_tag = f"ecw{int(round(self.early_coverage_weight * 100)):03d}"
        slot_tag = f"slot{self.fixed_early_slots}" if self.fixed_early_slots >= 0 else "slotdyn"
        pm_tag = f"pmf{int(self.use_pm_failure_memory)}r{int(self.use_pm_rarity_memory)}i{int(self.use_pm_instability_penalty)}bg{int(self.use_bridge_guard)}dh{int(self.disable_dynamic_hardness)}"
        tag = (
            f"{strategy_tag}_k{self.topk_blocks}_w{self.warmup_rounds}_s{self.seed}_{enc_tag}_"
            f"{score_tag}_{budget_tag}_{hist_tag}_{client_tag}_{block_tag}_{hard_tag}_{util_tag}_{early_tag}_{cov_tag}_{mem_tag}_{dyn_tag}_{ecw_tag}_{slot_tag}_{pm_tag}"
        )
        return self.suite_root / self.task_name / tag

    @property
    def method_label(self) -> str:
        return self.method_name or self.selection_strategy

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["task_path"] = str(self.task_path)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(frozen=True)
class RagEvalConfig:
    model_path: str
    script_name: str = "main_100_test.py"
    conda_env: str | None = None
    output_name: str = "rag_eval.log"

    @property
    def script_path(self) -> Path:
        return RAGTEST_DIR / self.script_name


def default_seed_list() -> List[int]:
    return [0, 1, 2]
