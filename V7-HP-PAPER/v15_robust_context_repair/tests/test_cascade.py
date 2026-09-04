import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_cheap_gate", ROOT / "cascade" / "07_train_cheap_gate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CheapCascadeTest(unittest.TestCase):
    def test_feature_contract_excludes_expensive_and_outcome_features(self):
        rows = [
            {
                "features": {
                    "sequence_sparse_score_mean": 2.0,
                    "sequence_dense_score_mean": 0.5,
                    "mean_query_doc_overlap": 0.2,
                    "sequence_cross_score_mean": 8.0,
                    "reader_outcome": 1.0,
                }
            }
        ]
        self.assertEqual(
            MODULE.cheap_names(rows),
            ["mean_query_doc_overlap", "sequence_dense_score_mean", "sequence_sparse_score_mean"],
        )

    def test_offline_opportunity_target_uses_robust_same_action_gain(self):
        baseline = {
            "query_id": "q1",
            "features": {"is_baseline": 1.0, "sequence_sparse_score_mean": 2.0},
            "reader_labels": {"a": {"joint_delta": 0.0}, "b": {"joint_delta": 0.0}},
        }
        repair = {
            "query_id": "q1",
            "features": {"is_baseline": 0.0, "sequence_sparse_score_mean": 1.0},
            "reader_labels": {"a": {"joint_delta": 0.2}, "b": {"joint_delta": 0.1}},
        }
        x, y, query_ids = MODULE.build_examples(
            [baseline, repair], ["sequence_sparse_score_mean"], beta=1.0, min_gain=1e-6
        )
        self.assertEqual(query_ids, ["q1"])
        self.assertEqual(x.tolist(), [[2.0]])
        self.assertEqual(y.tolist(), [1])


if __name__ == "__main__":
    unittest.main()

