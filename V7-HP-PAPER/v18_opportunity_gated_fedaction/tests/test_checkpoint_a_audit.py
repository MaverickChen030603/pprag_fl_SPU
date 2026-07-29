import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "checkpoint_a" / "01_checkpoint_a_integrity_audit.py"


class CheckpointAuditTest(unittest.TestCase):
    def test_refuses_partial_run_without_decision_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v17"
            root.mkdir()
            output = Path(tmp) / "out"
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--v17-root", str(root),
                "--output-dir", str(output),
            ], check=False)
            self.assertEqual(completed.returncode, 2)
            report = json.loads((output / "checkpoint_a_integrity.json").read_text())
            self.assertEqual(report["status"], "refused_partial_run")


if __name__ == "__main__":
    unittest.main()
