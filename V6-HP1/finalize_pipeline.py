from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiment_config import OUTPUT_ROOT
from report_generator import write_full_pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the V6-HP1 upstream + downstream experiment pipeline.")
    parser.add_argument("--suite-name", default="all_v6_hp1")
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--upstream-root", default=str(OUTPUT_ROOT / "pprag_fl_v6_hp1" / "v6hp1_adhoc"))
    parser.add_argument("--downstream-root", default=str(OUTPUT_ROOT / "rag_eval_all_v6_hp1" / "v6hp1_adhoc"))
    parser.add_argument("--script", default="main_100_test.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default="hotpot_qa")
    parser.add_argument("--hotpot-split", default="validation")
    parser.add_argument("--hotpot-max-examples", type=int, default=1000)
    parser.add_argument("--query-subset", default=None)
    parser.add_argument("--save-per-query", action="store_true")
    parser.add_argument("--force-rag", action="store_true")
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    while args.wait_pid > 0 and pid_alive(args.wait_pid):
        time.sleep(max(args.poll_seconds, 1))

    run_command(
        [
            args.python,
            str(Path(__file__).resolve().parent / "summarize_results.py"),
            "--root",
            args.upstream_root,
            "--output",
            str(Path(args.upstream_root) / "summary"),
        ]
    )
    rag_command = [
        args.python,
        str(Path(__file__).resolve().parent / "run_all_rag_eval.py"),
        "--upstream-root",
        args.upstream_root,
        "--output-root",
        args.downstream_root,
        "--script",
        args.script,
        "--python",
        args.python,
        "--dataset",
        args.dataset,
        "--hotpot-split",
        args.hotpot_split,
        "--hotpot-max-examples",
        str(args.hotpot_max_examples),
    ]
    if args.query_subset:
        rag_command.extend(["--query-subset", str(Path(args.query_subset).expanduser().resolve())])
    if args.save_per_query:
        rag_command.append("--save-per-query")
    if args.force_rag:
        rag_command.append("--force")
    run_command(rag_command)

    report_path = write_full_pipeline_report(
        args.suite_name,
        Path(args.upstream_root),
        Path(args.downstream_root),
    )
    print(f"Full pipeline archive report written to {report_path}")


if __name__ == "__main__":
    main()
