SUP_V1_OPTION = {
    # FedRAG baseline training options
    "num_rounds": 25,
    "num_epochs": 1,
    "gpu": 0,
    "batch_size": 8,
    "learning_rate": 1e-5,
    # Hypernetwork options
    "hn_lr": 5e-3,
    "hn_embedding_dim": 64,
    "hn_hidden_dim": 128,
    # Selective-upload options
    "selective_block_strategy": "bert",
    "selective_warmup_rounds": 2,
    "selective_topk_blocks": 3,
    "selective_min_score": 0.0,
    "selective_always_upload": ["pooler"],
}

