from __future__ import annotations

import unittest

from oracle_search import oracle_landscape


class OracleSynergyTests(unittest.TestCase):
    def test_strict_synergy_uses_all_single_edits(self) -> None:
        rows = [
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 0, "is_baseline": True, "answer_f1": 0.2, "sp_f1": 0.2, "joint_f1": 0.1},
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 1, "trajectory_id": "s1", "answer_f1": 0.3, "sp_f1": 0.3, "joint_f1": 0.2},
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 1, "trajectory_id": "s2", "answer_f1": 0.4, "sp_f1": 0.4, "joint_f1": 0.25},
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 2, "trajectory_id": "c1", "answer_f1": 0.5, "sp_f1": 0.5, "joint_f1": 0.4},
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 3, "trajectory_id": "c2", "answer_f1": 0.45, "sp_f1": 0.45, "joint_f1": 0.35},
            {"query_id": "q1", "dataset": "hotpotqa", "reader": "r", "depth": 5, "trajectory_id": "u1", "answer_f1": 0.6, "sp_f1": 0.6, "joint_f1": 0.5},
        ]
        result = oracle_landscape.summarize_query(rows)
        self.assertAlmostEqual(result["strict_synergy_joint"], 0.15)
        self.assertEqual(result["composition_only_positive_joint"], 0)
        self.assertAlmostEqual(result["best_two_step_delta_joint"], 0.3)
        self.assertAlmostEqual(result["best_three_step_delta_joint"], 0.25)
        self.assertAlmostEqual(result["best_any_evaluated_context_delta_joint"], 0.4)


if __name__ == "__main__":
    unittest.main()
