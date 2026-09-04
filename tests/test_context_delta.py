from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from v7_hp4.hybrid_retriever import context_delta_audit, docs_from_micro_case


def test_topk_context_delta() -> None:
    data_path = ROOT / "data" / "v7_hp4_micro_benchmark.json"
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    assert len(cases) >= 20, f"expected at least 20 HP4 micro cases, got {len(cases)}"
    changed = 0
    for case in cases[:30]:
        docs = docs_from_micro_case(case)
        support_ids = [d.doc_id for d in docs if d.is_support]
        assert len(support_ids) == 2, f"case {case.get('id')} must have exactly two support docs"
        audit = context_delta_audit(docs, str(case["question"]), support_ids, top_k=5)
        if float(audit["overlap_at_k"]) < 1.0:
            changed += 1
        assert all(audit["target_in_high"].values()), (
            f"support docs should surface when w=1 for case {case.get('id')}: {audit}"
        )
    assert changed > 0, "Overlap@K stayed 100% for all cases; soft weights do not alter context"


if __name__ == "__main__":
    test_topk_context_delta()
    print("Top-K Context Delta Audit passed.")

