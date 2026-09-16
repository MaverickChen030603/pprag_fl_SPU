#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from materialize_posthoc_baselines import random_top3, raw_merge, static_top3


def records() -> list[dict[str, object]]:
    return [
        {
            "client_id": client,
            "static_candidate_rank": rank,
            "static_score": float(8 - rank),
        }
        for rank, client in enumerate((7, 2, 9, 1, 5, 3, 8, 4), start=1)
    ]


class SelectorTest(unittest.TestCase):
    def test_static_top3_obeys_rank(self) -> None:
        self.assertEqual(static_top3(records()), [7, 2, 9])

    def test_random_top3_is_deterministic_and_seeded(self) -> None:
        first = random_top3(records(), "query-1", 0)
        self.assertEqual(first, random_top3(records(), "query-1", 0))
        self.assertNotEqual(first, random_top3(records(), "query-1", 1))
        self.assertEqual(len(set(first)), 3)

    def test_raw_merge_uses_five_per_client_and_stable_tie_break(self) -> None:
        packet = {"local_dense_docs_top10": {}}
        for client in (1, 2, 3):
            packet["local_dense_docs_top10"][str(client)] = [
                {
                    "doc_id": f"d{client}-{rank}",
                    "dense_score": 1.0 if rank == 0 else 1.0 / (rank + client),
                }
                for rank in range(10)
            ]
        transmitted, merged = raw_merge(packet, [1, 2, 3])
        self.assertEqual(len(transmitted), 15)
        self.assertEqual(len(merged), 10)
        self.assertEqual(
            [row["doc_id"] for row in merged[:3]], ["d1-0", "d2-0", "d3-0"]
        )
        self.assertTrue(all(int(row["doc_id"].split("-")[1]) < 5 for row in transmitted))


if __name__ == "__main__":
    unittest.main()
