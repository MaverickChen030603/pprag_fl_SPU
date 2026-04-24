from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
FEDE_DIR = REPO_ROOT / "FedE"
RAGTEST_DIR = REPO_ROOT / "RAGTest"
OUTPUT_ROOT = REPO_ROOT / "SUP_v2" / "outputs"
REPORT_ROOT = REPO_ROOT / "实验分析报告"


@dataclass(frozen=True)
class UpstreamConfig:
    experiment_name: str = "pprag_fl_sup_v2"
    task_name: str = "num5_alpha05_sup_v2"
    num_clients: int = 5
    num_rounds: int = 25
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    gpu: int = 0
    seed: int = 0
    selection_strategy: str = "hypernet"
    topk_blocks: int = 3
    warmup_rounds: int = 2
    block_strategy: str = "bert"
    always_upload: tuple[str, ...] = ("pooler",)
    hn_lr: float = 5e-3
    hn_embedding_dim: int = 64
    hn_hidden_dim: int = 128
    estimate_encryption: bool = False
    encryption_expansion: float = 8.0

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
        }

    @property
    def task_path(self) -> Path:
        return FEDE_DIR / self.task_name

    @property
    def output_dir(self) -> Path:
        enc_tag = "enc1" if self.estimate_encryption else "enc0"
        tag = f"{self.selection_strategy}_k{self.topk_blocks}_w{self.warmup_rounds}_s{self.seed}_{enc_tag}"
        return OUTPUT_ROOT / self.experiment_name / tag

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


def comparison_suite(base: UpstreamConfig | None = None) -> List[UpstreamConfig]:
    base = base or UpstreamConfig()
    return [
        replace(base, selection_strategy="full", topk_blocks=0),
        replace(base, selection_strategy="random"),
        replace(base, selection_strategy="static_top"),
        replace(base, selection_strategy="delta_norm"),
        replace(base, selection_strategy="hypernet"),
    ]


def topk_ablation_suite(base: UpstreamConfig | None = None) -> List[UpstreamConfig]:
    base = base or UpstreamConfig(selection_strategy="hypernet")
    return [replace(base, topk_blocks=k) for k in (1, 3, 5, 7)]


def warmup_ablation_suite(base: UpstreamConfig | None = None) -> List[UpstreamConfig]:
    base = base or UpstreamConfig(selection_strategy="hypernet")
    return [replace(base, warmup_rounds=w) for w in (0, 1, 2, 5)]


def encryption_ablation_suite(base: UpstreamConfig | None = None) -> List[UpstreamConfig]:
    base = base or UpstreamConfig(selection_strategy="hypernet")
    return [
        replace(base, estimate_encryption=False),
        replace(base, estimate_encryption=True),
    ]
