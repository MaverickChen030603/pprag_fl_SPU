from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "oracle_search" / "00_v15_pilot_synergy_probe.py"
SPEC = importlib.util.spec_from_file_location("v15_probe", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AtomicDepthTests(unittest.TestCase):
    def test_replacement_and_move_require_two_steps(self) -> None:
        baseline = ("a", "b", "c", "d", "e")
        target = ("x", "a", "b", "c", "d")
        self.assertEqual(MODULE.minimum_atomic_depth(baseline, target), 2)

    def test_two_replacements(self) -> None:
        self.assertEqual(MODULE.minimum_atomic_depth(("a", "b", "c", "d", "e"), ("a", "x", "c", "y", "e")), 2)

    def test_permutation(self) -> None:
        self.assertEqual(MODULE.minimum_atomic_depth(("a", "b", "c", "d", "e"), ("e", "b", "c", "d", "a")), 1)


if __name__ == "__main__":
    unittest.main()
