from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "action_scorer"))

from model import DirectMultiReaderScorer, multitask_loss
from scorer_common import validate_feature_names


class ActionScorerTest(unittest.TestCase):
    def test_multitask_loss_is_finite_and_backpropagates(self) -> None:
        model = DirectMultiReaderScorer(input_dim=8, readers=2, hidden_dim=16, dropout=0.0)
        features = torch.randn(6, 8)
        labels = torch.zeros(6, 2, 5)
        labels[:, :, :3] = torch.randn(6, 2, 3) * 0.1
        labels[:, :, 3:] = torch.randint(0, 2, (6, 2, 2)).float()
        prediction = model(features)
        loss, parts = multitask_loss(prediction, labels, ["q1"] * 3 + ["q2"] * 3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertGreaterEqual(parts["ranking"], 0.0)

    def test_rejects_label_derived_features(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_names(["bm25", "gold_support_count"])


if __name__ == "__main__":
    unittest.main()

