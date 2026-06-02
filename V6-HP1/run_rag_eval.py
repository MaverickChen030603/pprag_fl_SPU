from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiment_config import RAGTEST_DIR
from metrics import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream RAGTest evaluation for a trained retriever.")
    parser.add_argument("--model", required=True, help="Path to retriever model directory or checkpoint accepted by RAGTest.")
    parser.add_argument("--script", default="main_100_test.py", help="RAGTest script name, e.g. main_100_test.py or main_response.py.")
    parser.add_argument("--output-dir", default="V6-HP1/outputs/rag_eval")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default="hotpot_qa")
    parser.add_argument("--hotpot-split", default="validation")
    parser.add_argument("--hotpot-max-examples", type=int, default=1000)
    parser.add_argument("--query-subset", default=None)
    parser.add_argument("--save-per-query", action="store_true")
    parser.add_argument("--per-query-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    script_path = RAGTEST_DIR / args.script
    model_path = str(Path(args.model).expanduser().resolve())
    command = [args.python, str(script_path), "--model", model_path]
    if args.query_subset:
        command.extend(["--query-subset", str(Path(args.query_subset).expanduser().resolve())])
    if args.save_per_query:
        per_query_output = str(Path(args.per_query_output or (Path(output_dir) / "per_query_results.jsonl")).expanduser().resolve())
        command.extend(["--save-per-query", "--per-query-output", per_query_output])
    env = os.environ.copy()
    env["RAGTEST_DATASET"] = args.dataset
    env["HOTPOT_SPLIT"] = args.hotpot_split
    env["HOTPOT_MAX_EXAMPLES"] = str(args.hotpot_max_examples)
    write_json(
        output_dir / "rag_eval_command.json",
        {"command": command, "cwd": str(RAGTEST_DIR), "env": {"RAGTEST_DATASET": args.dataset, "HOTPOT_SPLIT": args.hotpot_split, "HOTPOT_MAX_EXAMPLES": args.hotpot_max_examples}},
    )
    if args.dry_run:
        print(" ".join(command))
        return
    with (output_dir / "rag_eval_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "rag_eval_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=str(RAGTEST_DIR), env=env, check=True, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    main()
