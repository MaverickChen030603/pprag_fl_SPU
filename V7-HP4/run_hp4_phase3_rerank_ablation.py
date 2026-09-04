from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RERANK = _load("V7-HP4/run_hp4_phase3_reader_aware_rerank.py", "hp4_reader_aware")


ORDER_PROFILES: dict[str, tuple[float, float, float]] = {
    "lexical": (0.65, 0.25, 0.10),
    "balanced": (0.50, 0.30, 0.20),
    "diverse": (0.40, 0.25, 0.35),
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_summary(summary: dict[str, dict[str, float]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for mode, metrics in summary.items():
        rows.append({
            "mode": mode,
            **metrics,
            "delta_support_recall_at_k": metrics["support_recall_at_k"] - baseline["support_recall_at_k"],
            "delta_sp_f1": metrics["sp_f1"] - baseline["sp_f1"],
            "delta_answer_f1": metrics["answer_f1"] - baseline["answer_f1"],
            "delta_joint_f1": metrics["joint_f1"] - baseline["joint_f1"],
            "reader_pass": metrics["answer_f1"] + 1e-12 >= baseline["answer_f1"]
            and metrics["joint_f1"] + 1e-12 >= baseline["joint_f1"],
        })
    rows.sort(
        key=lambda r: (
            bool(r["reader_pass"]),
            r["delta_answer_f1"],
            r["delta_joint_f1"],
            r["delta_support_recall_at_k"],
        ),
        reverse=True,
    )
    return rows


def write_report(path: Path, baseline: dict[str, float], ranked: list[dict[str, Any]]) -> None:
    lines = [
        "# V7-HP4 Phase 3 Reader-Aware Rerank Ablation",
        "",
        "- fixed routing: `A_tau0.7_gate4`",
        "- ablation: `background_pool in {8,16,24}` and ordering profiles lexical/balanced/diverse",
        "- strict_no_leak: no support labels, gold titles, or answer-presence features are used by policy or rerank",
        "",
        "## Baseline",
        "",
        (
            f"- A_tau1_no_gate: access@5={baseline['answer_access_at_k']:.4f}, "
            f"support_recall@5={baseline['support_recall_at_k']:.4f}, sp_f1={baseline['sp_f1']:.4f}, "
            f"answer_f1={baseline['answer_f1']:.4f}, joint_f1={baseline['joint_f1']:.4f}"
        ),
        "",
        "## Ranked Results",
        "",
        "| rank | mode | pass | access@5 | recall@5 | sp_f1 | answer_f1 | joint_f1 | d_answer | d_joint | d_recall |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['mode']} | {str(row['reader_pass']).lower()} | "
            f"{row['answer_access_at_k']:.4f} | {row['support_recall_at_k']:.4f} | {row['sp_f1']:.4f} | "
            f"{row['answer_f1']:.4f} | {row['joint_f1']:.4f} | "
            f"{row['delta_answer_f1']:+.4f} | {row['delta_joint_f1']:+.4f} | {row['delta_support_recall_at_k']:+.4f} |"
        )
    lines.extend(["", "## Decision", ""])
    passing = [row for row in ranked if row["reader_pass"]]
    if passing:
        best = passing[0]
        lines.append(f"Launch 1000 validation with `{best['mode']}` because answer_f1 and joint_f1 both meet or exceed baseline.")
    else:
        best = ranked[0]
        lines.append(
            f"No config recovered answer_f1 to baseline. Best near-miss is `{best['mode']}` "
            f"with d_answer={best['delta_answer_f1']:+.4f}, d_joint={best['delta_joint_f1']:+.4f}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_rerank_ablation")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--policy-a", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt")
    parser.add_argument("--prior-summary", default="V7-HP4/outputs/hp4_phase3_100_batch/phase3_100_summary.json")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--gate", type=int, default=4)
    parser.add_argument("--gate-floor", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))
    baseline = prior["A_phase2_baseline_tau1"]
    RERANK.log(f"materializing {args.sample_size} validation cases")
    examples = RERANK.READER.materialize_dev(Path(args.validation), args.sample_size)
    policy, policy_device = RERANK.load_policy(Path(args.policy_a), args.device)
    RERANK.log(f"loading reader {args.reader_model} on {args.device}")
    reader = RERANK.READER.Reader(args.reader_model, args.device, args.reader_batch_size)

    all_rows = []
    summary: dict[str, dict[str, float]] = {}
    for pool in [8, 16, 24]:
        for profile, order_weights in ORDER_PROFILES.items():
            mode = f"A_tau{args.tau:g}_gate{args.gate}_bgpool{pool}_{profile}"
            rows, mode_summary = RERANK.evaluate(
                examples,
                policy,
                policy_device,
                mode,
                args.tau,
                args.gate,
                args.gate_floor,
                args.top_k,
                args.alpha,
                pool,
                reader,
                order_weights=order_weights,
            )
            all_rows.extend(rows)
            summary.update(mode_summary)
            save_json(out / "rerank_ablation_partial_summary.json", flatten_summary(summary, baseline))

    ranked = flatten_summary(summary, baseline)
    payload = {
        "sample_size": len(examples),
        "baseline": baseline,
        "order_profiles": ORDER_PROFILES,
        "summary": summary,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "has_reader_pass": any(row["reader_pass"] for row in ranked),
    }
    save_json(out / "rerank_ablation_rows.json", all_rows)
    save_json(out / "rerank_ablation_summary.json", payload)
    report = Path(args.report_dir) / "v7_hp4_phase3_rerank_ablation_latest.md"
    write_report(report, baseline, ranked)
    print(json.dumps({**payload, "report_path": str(report), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
