#!/usr/bin/env python3
"""Plot the preregistered Phase-A opportunity-gap decomposition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = (
    ("routing_absence_rate", "Routing"),
    ("local_retrieval_absence_rate", "Local retrieval"),
    ("pool_absence_rate", "Pool"),
    ("single_action_absence_rate", "Single action"),
    ("cross_client_composition_absence_rate", "Composition"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["partition"] in {"topic_silo", "entity_community", "random_control"}
            and int(row["client_budget"]) == 3
        ]
    rows.sort(key=lambda row: (row["dataset"], row["reader"], row["partition"]))
    labels = [f"{row['dataset']}\n{row['reader']}\n{row['partition'].replace('_', ' ')}" for row in rows]
    x = np.arange(len(rows))
    width = 0.15
    fig, axis = plt.subplots(figsize=(max(12, len(rows) * 0.7), 5.8))
    for offset, (field, label) in enumerate(FIELDS):
        values = [float(row[field]) for row in rows]
        axis.bar(x + (offset - 2) * width, values, width=width, label=label)
    axis.set_ylabel("Query rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
