from __future__ import annotations

import unittest

from retrieval.retrieval_common import hop_count_without_labels, lexical_tokens, minmax


class RetrievalContractTests(unittest.TestCase):
    def test_musique_hop_count_comes_from_public_id(self) -> None:
        self.assertEqual(hop_count_without_labels({"query_id": "4hop__1_2_3_4"}, "musique"), 4)

    def test_minmax(self) -> None:
        self.assertEqual(minmax([2.0, 4.0]), [0.0, 1.0])

    def test_lexical_tokenization(self) -> None:
        self.assertEqual(lexical_tokens("A two-hop QA?"), ["two", "hop", "qa"])


if __name__ == "__main__":
    unittest.main()
