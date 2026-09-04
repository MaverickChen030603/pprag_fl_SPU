#!/usr/bin/env python3
"""Plot frozen Answer-Joint-latency operating points without imputing missing splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "v7_sigirap_targeted_strengthening"


def points() -> dict[str, list[dict[str, float | str]]]:
    metrics = json.loads((SOURCE / "outputs/reranker/ce_reranker_metrics.json").read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, float | str]]] = {}
    for split in ("holdout3000", "revision3405"):
        split_rows = metrics["splits"][split]["metrics"]
        rows = [
            {
                "system": "Frozen Top-5",
                "answer_f1": split_rows["baseline"]["answer_f1"],
                "joint_f1": split_rows["baseline"]["joint_f1"],
                "latency_ms": 140.88,
            },
            {
                "system": "CrossEncoder-Top5",
                "answer_f1": split_rows["ce_score_order"]["answer_f1"],
                "joint_f1": split_rows["ce_score_order"]["joint_f1"],
                "latency_ms": 149.90,
            },
            {
                "system": "Full",
                "answer_f1": split_rows["full"]["answer_f1"],
                "joint_f1": split_rows["full"]["joint_f1"],
                "latency_ms": 213.48,
            },
        ]
        if split == "holdout3000":
            rows.append({"system": "RECOMP-660", "answer_f1": 0.6226, "joint_f1": 0.3259, "latency_ms": 169.64})
        else:
            rows.append({"system": "Lite", "answer_f1": 0.6149, "joint_f1": 0.3217, "latency_ms": 143.97})
        result[split] = rows
    return result


def write_data(data: dict[str, list[dict[str, float | str]]]) -> None:
    with (HERE / "outputs/answer_joint_latency_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "system", "answer_f1", "joint_f1", "latency_ms"])
        writer.writeheader()
        for split, rows in data.items():
            for row in rows:
                writer.writerow({"split": split, **row})


def plot(data: dict[str, list[dict[str, float | str]]]) -> None:
    palette = {
        "Frozen Top-5": "#555555",
        "CrossEncoder-Top5": "#0072B2",
        "Full": "#D55E00",
        "Lite": "#009E73",
        "RECOMP-660": "#CC79A7",
    }
    markers = {"Frozen Top-5": "o", "CrossEncoder-Top5": "s", "Full": "D", "Lite": "^", "RECOMP-660": "v"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    titles = {"holdout3000": "(a) Original holdout (3,000)", "revision3405": "(b) Revision holdout (3,405)"}
    offsets = {
        "Frozen Top-5": (5, -16),
        "CrossEncoder-Top5": (5, 8),
        "Full": (-80, 7),
        "Lite": (5, 8),
        "RECOMP-660": (-70, -18),
    }
    for axis, split in zip(axes, ("holdout3000", "revision3405")):
        for row in data[split]:
            system = str(row["system"])
            x, y = float(row["answer_f1"]), float(row["joint_f1"])
            axis.scatter(x, y, s=70, color=palette[system], marker=markers[system], edgecolor="black", linewidth=0.5, zorder=3)
            axis.annotate(
                f"{system}\n{float(row['latency_ms']):.2f} ms",
                (x, y),
                xytext=offsets[system],
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
            )
        axis.set_title(titles[split], fontsize=10)
        axis.set_xlabel("Answer F1")
        axis.set_ylabel("Joint F1")
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.margins(x=0.18, y=0.22)
    fig.suptitle("Frozen answer-evidence-latency operating points", fontsize=11)
    fig.text(
        0.5,
        0.005,
        "Non-dominance is assessed only among the evaluated systems, metrics, and measured post-retrieval latency.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(HERE / "answer_joint_latency_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(HERE / "outputs/answer_joint_latency_tradeoff.png", dpi=180, bbox_inches="tight")


def main() -> None:
    data = points()
    write_data(data)
    plot(data)
    print(json.dumps({"status": "complete", "points": {key: len(value) for key, value in data.items()}}, indent=2))


if __name__ == "__main__":
    main()
