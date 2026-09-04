from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "protocol" / "02_freeze_data_splits.py"
SPEC = importlib.util.spec_from_file_location("freeze", SCRIPT)
freeze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(freeze)


class FreezeTest(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_disjoint(self) -> None:
        rows = [
            {"_id": f"q{index:03d}", "type": "bridge" if index % 2 else "comparison", "level": "hard" if index % 3 else "easy"}
            for index in range(50)
        ]
        sizes = {"train": 10, "development": 5, "calibration": 5, "final_test": 5}
        first = freeze.allocate_stratified(rows, "hotpotqa", sizes, 7)
        second = freeze.allocate_stratified(rows, "hotpotqa", sizes, 7)
        self.assertEqual(
            {name: [freeze.query_id(row) for row in values] for name, values in first.items()},
            {name: [freeze.query_id(row) for row in values] for name, values in second.items()},
        )
        sets = [set(freeze.query_id(row) for row in values) for values in first.values()]
        self.assertEqual(sum(map(len, sets)), len(set().union(*sets)))

    def test_strip_labels(self) -> None:
        inputs, labels = freeze.strip_labels({"_id": "q1", "question": "Q?", "answer": "A", "supporting_facts": [["T", 0]]})
        self.assertNotIn("answer", inputs)
        self.assertEqual("A", labels["answer"])
        self.assertEqual("q1", inputs["query_id"])


if __name__ == "__main__":
    unittest.main()

