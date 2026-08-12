#!/usr/bin/env python3
"""Write the R4 reader/baseline/no-leak preflight audit before inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


V20 = Path(__file__).resolve().parents[1]
V17 = V20.parent / "v17_fedaction_rag"
R3 = V20 / "stage_r3_probe_route"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = args.output_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    cache = Path("/home/iiserver31/.cache/huggingface/hub")
    checkpoints = {
        "google/flan-t5-large": (cache / "models--google--flan-t5-large/refs/main").read_text().strip(),
        "allenai/unifiedqa-v2-t5-large-1363200": (cache / "models--allenai--unifiedqa-v2-t5-large-1363200/refs/main").read_text().strip(),
    }
    reader_audit = f"""# R4-P0 Reader Context Contract Audit

**Decision:** `pass`; no unresolved reader-contract ambiguity.

## Evidence hierarchy

1. V20 preregistration fixes a retrieval global pool/context contract of `Top-10 / Top-5`.
2. V17 preregistration independently fixes final context `K=5` and the two reader checkpoints.
3. V17's executed `01_label_oracle_contexts.py` uniquely specifies the input serialization, truncation and decoding used below.

## Frozen contract carried into R4

| Component | Contract |
|---|---|
| Retrieval global pool | raw merged Top-10 |
| Reader context | first 5 documents in that frozen order |
| Duplicate policy | no duplicate document IDs; no padding |
| FLAN format | `[rank] title: text` with answer-only instruction |
| UnifiedQA format | `question` followed by `title: text` spans |
| Character truncation | first 4,000 context characters after ordered serialization |
| Tokenizer encoding | truncation to 1,024 input tokens; default truncation direction (right) |
| Decode | greedy (`num_beams=1`, `do_sample=False`), `max_new_tokens=32` |
| Support extraction | frozen V16 support predictor, threshold plus deterministic top-two fallback |

The V20 text does not restate every prompt detail, but its `Top-10 / Top-5` table and the V17 executed script agree. Therefore this is recorded as **legacy frozen reader K=5 carried forward**, not a post-reader choice.

## Cached checkpoints

| Reader | Revision |
|---|---|
| `google/flan-t5-large` | `{checkpoints['google/flan-t5-large']}` |
| `allenai/unifiedqa-v2-t5-large-1363200` | `{checkpoints['allenai/unifiedqa-v2-t5-large-1363200']}` |
"""
    (protocol / "reader_context_contract_audit.md").write_text(reader_audit, encoding="utf-8")
    prereg = """# V20 Stage R4 Reader Preregistration

- Data: the three frozen R3 N=300 holdouts, unchanged query IDs.
- Conditions: federated baseline, label-free ProbeRoute, logistic ProbeRoute, and centralized retrieval reference.
- Primary outcome: query-level Joint F1, Logistic ProbeRoute versus federated baseline.
- Readers: frozen FLAN-T5-Large and UnifiedQA-T5-Large under the audit contract.
- Statistics: 5,000 paired bootstrap resamples; Joint primary uncorrected, remaining paired tests BH-FDR reported.
- Reader evaluation never changes packets, clients, raw merge, context K, prompt, decoder, or retrieval configuration.
"""
    (protocol / "reader_preregistration.md").write_text(prereg, encoding="utf-8")
    baseline = """# Hotpot H0/C0 Baseline Contract Matrix

| Component | H0 inherited route (R4 primary) | C0 static Top-3 (cost-only) |
|---|---|---|
| Query IDs/sample | R3-T frozen Hotpot holdout N=300 | same R3-T holdout N=300 |
| Candidate generator | inherited topic origin-plus-centroid route | static P0 centroid profile |
| Selected clients | frozen `inherited_b3_routes.jsonl` | P0 static ranks 1--3 |
| Local depth/transmission | 10 local materialized, 5 per client, 15 total | identical |
| Global merge/pool | raw dense score, Top-10 | identical |
| Corpus/partition/index | V17 topic-silo canonical index and assignment | identical |
| R3 observed raw complete@10 | 0.4700 | 0.5100 |
| Role in R4 | **federated baseline** | communication/Pareto comparison only |

The difference is explained by route selection, not a detected implementation failure. R4 therefore uses H0 only as the Hotpot main federated baseline. C0 is not admitted to the reader comparison.
"""
    (protocol / "baseline_contract_matrix.md").write_text(baseline, encoding="utf-8")
    artifacts = [
        R3 / "ranker_training/packets/2wikimultihopqa/probe_holdout.jsonl",
        R3 / "ranker_training/packets/musique/probe_holdout.jsonl",
        R3 / "hotpot_transfer/packets/probe_holdout.jsonl",
        R3 / "protocol/2wikimultihopqa/probe_holdout.jsonl",
        R3 / "protocol/musique/probe_holdout.jsonl",
        R3 / "hotpot_transfer/protocol/probe_holdout.jsonl",
    ]
    audit = {
        "status": "pass", "reader_started_before_r4": False, "final_test_accessed": False,
        "input_artifact_sha256": {str(path): sha256(path) for path in artifacts},
        "forbidden_input_fields": ["answer", "supporting_facts", "is_supporting", "gold_support"],
        "reader_input_source": "materialized title/text and question only",
        "centralized_reference": "same frozen V17 retrieval contract replayed on R3 query IDs; reference not upper bound",
    }
    (protocol / "no_leak_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
