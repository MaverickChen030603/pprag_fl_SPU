from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_config import RAGTEST_DIR
from metrics import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream RAGTest evaluation for a trained retriever.")
    parser.add_argument("--model", required=True, help="Path to retriever model directory or checkpoint accepted by RAGTest.")
    parser.add_argument("--script", default="main_100_test.py", help="RAGTest script name, e.g. main_100_test.py or main_response.py.")
    parser.add_argument("--output-dir", default="SUP_v3/outputs/rag_eval")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    script_path = RAGTEST_DIR / args.script
    model_path = str(Path(args.model).expanduser().resolve())
    command = [args.python, str(script_path), "--model", model_path]
    write_json(output_dir / "rag_eval_command.json", {"command": command, "cwd": str(RAGTEST_DIR)})
    if args.dry_run:
        print(" ".join(command))
        return
    with (output_dir / "rag_eval_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "rag_eval_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=str(RAGTEST_DIR), check=True, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    main()
