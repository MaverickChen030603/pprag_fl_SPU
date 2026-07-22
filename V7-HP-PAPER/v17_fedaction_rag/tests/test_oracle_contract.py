from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OracleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load(ROOT / "oracle" / "03_analyze_federated_oracle.py", "analysis_impl")

    def test_fed_gain_uses_best_single_client_or_cross_action(self):
        outcomes = [
            {"trajectory_id": "a", "joint_f1": 0.6, "candidate_type": "single_client_context", "depth": 5, "is_single_cross_action": False, "is_cross_composition": False},
            {"trajectory_id": "b", "joint_f1": 0.7, "candidate_type": "trajectory", "depth": 1, "is_single_cross_action": True, "is_cross_composition": False},
            {"trajectory_id": "c", "joint_f1": 0.9, "candidate_type": "trajectory", "depth": 2, "is_single_cross_action": False, "is_cross_composition": True},
        ]
        single_client, _ = self.analysis.best_delta(outcomes, 0.5, lambda row: row["candidate_type"] == "single_client_context", "joint_f1")
        single_cross, _ = self.analysis.best_delta(outcomes, 0.5, lambda row: row["is_single_cross_action"], "joint_f1")
        composed, _ = self.analysis.best_delta(outcomes, 0.5, lambda row: row["is_cross_composition"], "joint_f1")
        self.assertAlmostEqual(composed - max(single_client, single_cross), 0.2)

    def test_bootstrap_degenerate_interval(self):
        low, high = self.analysis.bootstrap_ci([0.1] * 20, 100, 1)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.1)


if __name__ == "__main__":
    unittest.main()
