from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_generation import Document, beam_sequence_repairs, enumerate_set_repairs


def docs(count: int) -> list[Document]:
    return [
        Document(
            doc_id=f"d{index}",
            retrieval_score=1.0 - index / max(1, count),
            cross_score=(index % 5) / 4,
            bridge_score=1.0 if index in {6, 7} else 0.0,
            anchor_score=1.0 if index < 2 else 0.0,
            source_rank=index,
        )
        for index in range(count)
    ]


class EnumeratedRepairTest(unittest.TestCase):
    def test_keeps_null_action_and_unique_sequences(self) -> None:
        pool = docs(10)
        actions = enumerate_set_repairs(pool, pool[:5], top_k=16)
        self.assertEqual(16, len(actions))
        self.assertTrue(any(action.is_baseline for action in actions))
        self.assertEqual(len(actions), len({action.doc_ids for action in actions}))
        self.assertTrue(any("d6" in action.doc_ids or "d7" in action.doc_ids for action in actions))

    def test_rejects_pool_larger_than_twelve(self) -> None:
        pool = docs(13)
        with self.assertRaises(ValueError):
            enumerate_set_repairs(pool, pool[:5], top_k=8)


class BeamRepairTest(unittest.TestCase):
    def test_search_changes_complete_context(self) -> None:
        pool = docs(20)
        baseline = pool[:5]
        actions = beam_sequence_repairs(pool, baseline, beam_width=8, depth=2, output_k=16)
        self.assertTrue(any(action.is_baseline for action in actions))
        self.assertTrue(any(action.added_doc_ids for action in actions))
        self.assertEqual(len(actions), len({action.doc_ids for action in actions}))
        self.assertTrue(all(len(action.doc_ids) == 5 for action in actions))


if __name__ == "__main__":
    unittest.main()

