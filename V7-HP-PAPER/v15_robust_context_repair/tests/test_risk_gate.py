from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "risk_gate"))

from gate_common import apply_threshold, best_actions, observed_risk
from importlib import import_module


class RiskGateTest(unittest.TestCase):
    def test_independent_best_action_and_exact_fallback(self) -> None:
        rows = [
            {"query_id": "q1", "action_id": "a", "predicted_utility": 0.4, "predicted_answer_harm": 0.1, "answer_drop": 0, "joint_drop": 0, "answer_delta": 0.1, "joint_delta": 0.1},
            {"query_id": "q1", "action_id": "b", "predicted_utility": 0.8, "predicted_answer_harm": 0.2, "answer_drop": 0, "joint_drop": 0, "answer_delta": 0.2, "joint_delta": 0.2},
            {"query_id": "q2", "action_id": "c", "predicted_utility": 0.3, "predicted_answer_harm": 0.8, "answer_drop": 1, "joint_drop": 1, "answer_delta": -0.2, "joint_delta": -0.2},
        ]
        self.assertEqual(["b", "c"], [row["action_id"] for row in best_actions(rows)])
        decisions = apply_threshold(rows, utility_threshold=0.5, harm_threshold=0.5)
        self.assertEqual([1, 0], [row["selected"] for row in decisions])
        self.assertEqual("exact_fallback", decisions[1]["decision"])
        self.assertEqual(0.5, observed_risk(decisions)["coverage"])


if __name__ == "__main__":
    unittest.main()

