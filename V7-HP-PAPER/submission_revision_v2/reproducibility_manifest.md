# Reproducibility Manifest

## Code and data snapshot

- Remote project root: `/home/iiserver31/projects/FedE4RAG-main`
- Historical Git commit observed during packaging: `04a159a106c3761cb40f868035bbc1719dde5f0b`
- Worktree status: dirty; unrelated experimental artifacts were present
- Main dataset and action-label checksums: see `data_manifest.json` and `artifact_inventory.json`
- Exact query and fold assignments: `data_manifest.json`, `fold_manifest.json`

The dirty worktree means the commit alone is not a complete code identity. The submission package therefore checksums all result-critical scripts and artifacts. A clean archival tag remains recommended.

## Reader

- Model family: `google/flan-t5-large`
- Exact model revision: `[NEEDS SOURCE FILE]` (not logged)
- Exact tokenizer revision: `[NEEDS SOURCE FILE]` (not logged)
- Prompt: `Answer the question using only the context. Return a short answer.` followed by the question and numbered `title: text` documents
- Context truncation: 3,200 characters before tokenization
- Tokenizer maximum length: 1,024
- `max_new_tokens`: 32
- Beams: 1
- Sampling: disabled
- Reader batch size: 1
- Reader device in the archived launcher: CUDA device 0

## Retrieval/action settings

- Context budget: top-5 documents
- Hybrid retrieval alpha in archived HP4 launcher: 0.55
- Main action families: one fixed reranking and three conservative one-document insertion families
- Two-document insertion: materialized but excluded from the main selector
- Primary nested configuration: two-stage safety/positive scoring, safety threshold 0.5, positive threshold 0.1, 0.5 train-selected coverage, fallback otherwise

## Statistical settings

- Outer folds: 5, deterministic query-level MD5 ordering
- Inner nuisance folds: 5 per outer train split
- Bootstrap: paired query-level resampling
- Bootstrap repetitions: 2,000
- Bootstrap seed: 13 for the primary significance report
- Risk-coverage sweep: 0.1 to 1.0 in increments of 0.1; diagnostic only

## Environment

- OS: Ubuntu 20.04 generation, Linux 5.4 kernel snapshot
- Python: 3.10.20
- PyTorch: 2.3.1+cu121
- Transformers: 4.41.2
- Datasets: 4.8.5
- scikit-learn: 1.7.2
- NumPy: 1.26.4
- SciPy: 1.15.3
- Matplotlib: 3.10.9
- SentencePiece: 0.2.1
- GPU: 4 x NVIDIA A100-PCIE-40GB available on the packaging host
- Driver: 535.54.03

## Runtime

The historical end-to-end reader runtime was not logged in a stable artifact: `[NEEDS SOURCE FILE]`. The nested selector rerun is CPU-only and does not rerun the reader; it consumes the fixed 5,000 action-outcome table.

## Reproduction order

1. verify source sample against `data_manifest.json`;
2. verify action labels against their checksum;
3. generate nested nuisance features and selector decisions;
4. run paired significance and ablations;
5. generate risk-coverage and utility-sensitivity diagnostics;
6. regenerate manifests and inventory.

Executable commands are provided in `run_commands.sh`. The package is reproducible from the archived action table, but an exact raw-reader replay still requires pinning the missing model/tokenizer revisions.
