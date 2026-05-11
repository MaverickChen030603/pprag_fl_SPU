from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
FEDE_DIR = REPO_ROOT / "FedE"
RAGTEST_DIR = REPO_ROOT / "RAGTest"
OUTPUT_ROOT = REPO_ROOT / "V3" / "outputs"
REPORT_ROOT = REPO_ROOT / "实验分析报告" / "V3"


def build_task_name(
    num_clients: int,
    partitioner_name: str,
    dirichlet_alpha: float,
    imbalance: float,
    task_seed: int,
    suffix: str = "v3",
) -> str:
    partitioner_tag = partitioner_name.lower()
    if partitioner_tag.startswith("dir"):
        alpha_tag = str(dirichlet_alpha).replace(".", "")
        imbalance_tag = str(imbalance).replace(".", "")
        return f"num{num_clients}_dir_a{alpha_tag}_imb{imbalance_tag}_ts{task_seed}_{suffix}"
    return f"num{num_clients}_id_ts{task_seed}_{suffix}"


@dataclass(frozen=True)
class UpstreamConfig:
    experiment_name: str = "pprag_fl_v3"
    task_name: str = "num5_dir_a03_imb00_ts0_v3"
    suite_tag: str = "v3_main"
    num_clients: int = 5
    num_rounds: int = 25
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    gpu: int = 0
    seed: int = 0
    selection_strategy: str = "hypernet_v3"
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
    score_mode: str = "value"
    budget_mode: str = "adaptive"
    history_window: int = 5
    use_client_embedding: bool = True
    use_history_features: bool = True
    use_block_embedding: bool = True
    adaptive_min_topk: int = 1
    adaptive_max_topk: int = 7
    adaptive_scale: float = 1.0
    layerwise_budget: bool = False

    def to_flgo_option(self) -> Dict:
        return {
            "num_rounds": self.num_rounds,
            "num_epochs": self.num_epochs,
            "gpu": self.gpu,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
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
        strategy_tag = self.selection_strategy.replace("_", "-")
        score_tag = f"score-{self.score_mode}"
        budget_tag = f"budget-{self.budget_mode}"
        hist_tag = f"hist{self.history_window}" if self.use_history_features else "hist0"
        client_tag = "client1" if self.use_client_embedding else "client0"
        block_tag = "block1" if self.use_block_embedding else "block0"
        tag = (
            f"{strategy_tag}_k{self.topk_blocks}_w{self.warmup_rounds}_s{self.seed}_{enc_tag}_"
            f"{score_tag}_{budget_tag}_{hist_tag}_{client_tag}_{block_tag}"
        )
        return self.suite_root / self.task_name / tag

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
