from __future__ import annotations

import importlib.util
import sqlite3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RetrievalContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load(ROOT / "retrieval" / "01_generate_federated_pools.py", "retrieval_impl")

    def test_local_sparse_search_never_crosses_client_boundary(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT, text TEXT)")
        connection.execute(
            "CREATE VIRTUAL TABLE docs_fts USING fts5(title, text, content='docs', content_rowid='id')"
        )
        connection.executemany(
            "INSERT INTO docs(id,doc_id,title,text) VALUES (?,?,?,?)",
            [
                (1, "a", "Alpha", "shared bridge token"),
                (2, "b", "Beta", "shared bridge token"),
                (3, "c", "Gamma", "unrelated text"),
            ],
        )
        connection.execute("INSERT INTO docs_fts(docs_fts) VALUES ('rebuild')")
        self.module.install_client_assignments(connection, {"a": 0, "b": 1, "c": 1})
        client_zero = self.module.local_sparse_search(connection, "shared bridge", 0, 5)
        client_one = self.module.local_sparse_search(connection, "shared bridge", 1, 5)
        self.assertEqual({row["doc_id"] for row in client_zero}, {"a"})
        self.assertEqual({row["doc_id"] for row in client_one}, {"b"})
        self.assertTrue(all(row["client_id"] == 0 for row in client_zero))
        self.assertTrue(all(row["client_id"] == 1 for row in client_one))
        connection.close()


if __name__ == "__main__":
    unittest.main()
