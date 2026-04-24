# SUP_V1: Hypernetwork-Guided Selective Upload for FedE4RAG

This folder contains a V1 prototype for adding hypernetwork-based parameter
importance estimation to the upstream FedE4RAG federated training pipeline.

The baseline `FedE/flgo/algorithm/fedrag.py` is not modified. This prototype is
kept in `SUP_V1/` so it can be reviewed and tested before being copied into the
main FLGo algorithm package.

## Goal

FedRAG currently trains local retrievers and uploads full model parameters for
server-side aggregation. SUP_V1 changes the communication path:

```text
client trains full local model
-> client computes block-level delta statistics
-> client uploads only selected block deltas
-> server sparsely aggregates uploaded deltas
-> server trains a small hypernetwork to predict future important blocks
```

The hypernetwork is used as a communication controller, not as a personalized
model generator in this V1.

## Files

- `hypernetwork.py`: block grouping, delta/stat helpers, and the
  `BlockImportanceHyperNetwork`.
- `fedrag_hn_selective.py`: FLGo-compatible `Server` and `Client` classes for
  FedRAG with selective upload.
- `main_sup_v1.py`: runnable example entrypoint mirroring `FedE/main.py`.
- `config_sup_v1_example.py`: option dictionary example.

## Block Strategy

For BERT/BGE-style retrievers, parameters are grouped into communication blocks:

```text
embeddings
encoder.layer.0
encoder.layer.1
...
encoder.layer.11
pooler
```

Only selected blocks upload parameter deltas. Non-selected blocks remain
unchanged on the server for that round.

## Hypernetwork Signal

The client returns small metadata for all blocks:

```text
l2 norm of block delta
mean l2 per parameter
max absolute delta
number of parameters
```

The server uses these statistics as supervision:

```text
HyperNetwork(client_id, block_id, previous_stats) -> importance score
target score ~= normalized block delta norm
```

At round `t`, the server predicts the upload mask using statistics learned from
previous rounds. The client only uploads the predicted blocks after local
training. This preserves communication savings because full deltas are never
sent after warmup.

## Important Options

- `selective_warmup_rounds`: full upload rounds before selective upload starts.
- `selective_topk_blocks`: number of BERT blocks uploaded per client per round.
- `selective_always_upload`: safety list for blocks that should always upload.
- `hn_lr`: hypernetwork learning rate.
- `hn_embedding_dim`: client/block embedding size.
- `hn_hidden_dim`: hypernetwork hidden size.

## Expected Behavior

With 14 blocks (`embeddings`, 12 encoder layers, `pooler`) and
`selective_topk_blocks=3`, the theoretical payload ratio after warmup should be
roughly 20-30%, depending on the exact parameter sizes of selected layers and
any `selective_always_upload` blocks.

## How To Test

Run from repository root:

```bash
python3 SUP_V1/main_sup_v1.py
```

Before running, the original FedE placeholders must still be configured:

```text
FedE/flgo/benchmark/fedrag_classification/config.py
FedE/flgo/benchmark/fedrag_classification/core.py
FedE/select_data.json
```

The current upstream code contains placeholder model paths such as
`/path/to/bge-en`, so update those paths before expecting a full training run.

## Integration Path

After validating the prototype:

1. Copy `SUP_V1/hypernetwork.py` into `FedE/flgo/algorithm/selective_hypernetwork.py`.
2. Copy `SUP_V1/fedrag_hn_selective.py` into `FedE/flgo/algorithm/fedrag_hn_selective.py`.
3. Change imports in `fedrag_hn_selective.py` from `SUP_V1` local imports to
   relative FLGo imports.
4. Use `algorithm=fedrag_hn_selective` in a new `FedE/main_sup_v1.py`.

## V1 Limitations

- Selection is block-level, not tensor-level or element-level.
- Hypernetwork decisions use previous-round statistics, avoiding a second
  communication phase in the same round.
- This prototype does not yet add LoRA/Adapter-only training, which is the
  recommended V2 for larger BERT/BGE retrievers.

