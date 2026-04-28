from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiment_config import OUTPUT_ROOT
from metrics import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream RAG evaluation for every upstream SUP_v2 run.")
    parser.add_argument("--upstream-root", default=str(OUTPUT_ROOT / "pprag_fl_sup_v2"))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT / "rag_eval_all"))
    parser.add_argument("--script", default="main_100_test.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-pid", type=int, default=0, help="Wait for an upstream PID to finish before scanning runs.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--include-pattern", default="", help="Only evaluate run directories containing this substring.")
    parser.add_argument("--force", action="store_true", help="Re-run even if an evaluation log already exists.")
    return parser.parse_args()


def latest_hf_model(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("retriever_hf_*"))
    return candidates[-1] if candidates else None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def discover_runs(root: Path, include_pattern: str) -> list[Path]:
    runs = []
    if not root.exists():
        return runs
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if include_pattern and include_pattern not in run_dir.name:
            continue
        if latest_hf_model(run_dir) is not None:
            runs.append(run_dir)
    return runs


def should_skip_existing(output_dir: Path, force: bool) -> bool:
    if force:
        return False
    stdout_log = output_dir / "rag_eval_stdout.log"
    if not stdout_log.exists():
        return False
    stderr_log = output_dir / "rag_eval_stderr.log"
    if stderr_log.exists():
        try:
            stderr_text = stderr_log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            stderr_text = ""
        if "Traceback" in stderr_text:
            return False
    return True


def run_eval(args: argparse.Namespace, run_dir: Path) -> dict:
    model_dir = latest_hf_model(run_dir)
    if model_dir is None:
        return {"run_dir": str(run_dir), "status": "skipped", "reason": "no_hf_model"}
    output_dir = ensure_dir(Path(args.output_root) / run_dir.name)
    done_flag = output_dir / "rag_eval_stdout.log"
    command = [
        args.python,
        str(Path(__file__).resolve().parent / "run_rag_eval.py"),
        "--model",
        str(model_dir),
        "--script",
        args.script,
        "--output-dir",
        str(output_dir),
        "--python",
        args.python,
    ]
    if should_skip_existing(output_dir, args.force):
        return {
            "run_dir": str(run_dir),
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "status": "skipped",
            "reason": "already_exists",
        }
    subprocess.run(command, check=True)
    return {
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "status": "completed",
    }


def main() -> None:
    args = parse_args()
    upstream_root = Path(args.upstream_root)
    output_root = ensure_dir(args.output_root)
    if args.wait_pid > 0:
        while pid_alive(args.wait_pid):
            time.sleep(max(args.poll_seconds, 1))
    records = []
    for run_dir in discover_runs(upstream_root, args.include_pattern):
        try:
            records.append(run_eval(args, run_dir))
        except subprocess.CalledProcessError as exc:
            records.append(
                {
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "returncode": exc.returncode,
                }
            )
    write_json(Path(output_root) / "rag_eval_all_summary.json", records)
    print(f"Processed {len(records)} upstream runs into {output_root}")


if __name__ == "__main__":
    main()
