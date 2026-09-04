#!/usr/bin/env python3
"""Render the diagnostic risk-coverage curve from the frozen CSV artifact."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "nested" / "risk_coverage_curve.csv"
TARGET = HERE / "outputs" / "nested" / "risk_coverage_figure.pdf"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    coverage = [float(row["realized_coverage"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].plot(coverage, [float(r["answer_f1_delta"]) for r in rows], marker="o", label="Answer F1")
    axes[0].plot(coverage, [float(r["product_delta"]) for r in rows], marker="s", label="Answer-title product")
    axes[0].set(xlabel="Realized intervention coverage", ylabel="Mean delta", title="Reader outcomes")
    axes[0].legend(frameon=False)

    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].plot(coverage, [float(r["title_support_recall_delta"]) for r in rows], marker="o", label="Title recall")
    axes[1].plot(coverage, [float(r["title_support_f1_delta"]) for r in rows], marker="s", label="Title F1")
    axes[1].plot(coverage, [float(r["selected_answer_drop_rate"]) for r in rows], marker="^", label="Answer-drop rate")
    axes[1].set(xlabel="Realized intervention coverage", ylabel="Rate / mean delta", title="Evidence gain and selected risk")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0.08, 0.92)

    fig.suptitle("Diagnostic risk-coverage sweep (not used for primary model selection)")
    fig.tight_layout()
    fig.savefig(TARGET, bbox_inches="tight")
    print(TARGET)


if __name__ == "__main__":
    main()
