from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PartitionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load(ROOT / "partitions" / "01_build_client_partitions.py", "partition_impl")

    def test_balanced_random_is_total_unique_and_balanced(self):
        labels = self.module.balanced_random(103, 20, 7)
        self.assertEqual(len(labels), 103)
        counts = [int(np.sum(labels == client)) for client in range(20)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_dirichlet_has_no_empty_client(self):
        topics = np.asarray([index % 5 for index in range(1000)], dtype=np.int32)
        labels = self.module.dirichlet_labels(topics, 20, 0.1, 9)
        self.assertEqual(set(map(int, labels)), set(range(20)))

    def test_assignment_writer_emits_one_row_per_document(self):
        docs = [{"doc_id": f"d{index}", "title": "", "text": ""} for index in range(7)]
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.jsonl"
            self.module.write_assignments(path, docs, labels)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), len(docs))


if __name__ == "__main__":
    unittest.main()
