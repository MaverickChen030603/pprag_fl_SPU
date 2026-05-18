from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiment_config import OUTPUT_ROOT
from report_generator import write_full_pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the V4 upstream + downstream experiment pipeline.")
    parser.add_argument("--suite-name", default="all")
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--upstream-root", default=str(OUTPUT_ROOT / "pprag_fl_v4" / "v4_adhoc"))
    parser.add_argument("--downstream-root", default=str(OUTPUT_ROOT / "rag_eval_all_v4" / "v4_adhoc"))
    parser.add_argument("--script", default="main_100_test.py")
    parser.add_argument("--python", default=sys.executable)
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
    ]
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
