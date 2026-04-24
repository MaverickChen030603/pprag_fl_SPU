import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
FEDE_DIR = os.path.join(REPO_ROOT, "FedE")

if FEDE_DIR not in sys.path:
    sys.path.insert(0, FEDE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import flgo
import fedrag_hn_selective as fedrag_hn_selective


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


task = os.path.join(FEDE_DIR, "num5_alpha05_sup_v1")
config = {
    "benchmark": {"name": "flgo.benchmark.fedrag_classification"},
    "partitioner": {"name": "IDPartitioner", "para": {"num_clients": 5}},
}

if not os.path.exists(task):
    old_cwd = os.getcwd()
    os.chdir(FEDE_DIR)
    flgo.gen_task(config, task_path=task)
    os.chdir(old_cwd)

runner = flgo.init(
    task=task,
    algorithm=fedrag_hn_selective,
    option={
        "num_rounds": 25,
        "num_epochs": 1,
        "gpu": 0,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "hn_lr": 5e-3,
        "hn_embedding_dim": 64,
        "hn_hidden_dim": 128,
        "selective_block_strategy": "bert",
        "selective_warmup_rounds": 2,
        "selective_topk_blocks": 3,
        "selective_min_score": 0.0,
        "selective_always_upload": ["pooler"],
    },
)
runner.run()

