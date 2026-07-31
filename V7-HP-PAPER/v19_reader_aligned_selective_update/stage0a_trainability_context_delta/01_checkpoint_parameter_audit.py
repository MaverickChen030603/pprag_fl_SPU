#!/usr/bin/env python3
"""Audit smoke checkpoints before interpreting a retrieval tie."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

from audit_common import flatten, grouped_state, sha256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(args.smoke_root.glob("*/adapter.pt"))
    if not checkpoints:
        raise FileNotFoundError(args.smoke_root)
    payloads = {path.parent.name: torch.load(path, map_location="cpu") for path in checkpoints}
    if "frozen" not in payloads:
        raise ValueError("frozen checkpoint is required as initialization reference")
    reference = payloads["frozen"]["adapter_state"]
    inventory, deltas, pairwise = [], [], []
    for path in checkpoints:
        condition, payload = path.parent.name, payloads[path.parent.name]
        state = payload["adapter_state"]
        log = path.parent / "round_log.csv"
        optimizer_steps = "not_recorded" if not log.exists() else sum(1 for _ in log.open(encoding="utf-8")) - 1
        inventory.append({"condition": condition, "checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256(path),
            "adapter_parameter_sha256": sha256(path), "training_seed": payload.get("seed"), "optimizer_steps": optimizer_steps,
            "examples_seen": "not_recorded", "final_training_loss": "not_recorded", "save_timestamp": path.stat().st_mtime,
            "tensor_count": len(state), "inference_hash_verified": "pending_forward_audit"})
        ref_groups, current_groups = grouped_state(reference), grouped_state(state)
        for block in sorted(ref_groups):
            before, after = flatten(ref_groups[block]), flatten(current_groups[block]); diff = after - before
            deltas.append({"condition": condition, "block_id": block, "initial_norm": float(before.norm()), "trained_norm": float(after.norm()),
                "delta_l2_norm": float(diff.norm()), "relative_delta_norm": float(diff.norm() / before.norm().clamp_min(1e-12)),
                "nonzero_delta_ratio": float((diff != 0).float().mean()), "max_abs_delta": float(diff.abs().max()),
                "mean_abs_delta": float(diff.abs().mean()), "gradient_nonzero_steps": "not_recorded", "mean_gradient_norm": "not_recorded",
                "max_gradient_norm": "not_recorded", "optimizer_state_present": False})
    for left, right in itertools.combinations(sorted(payloads), 2):
        a, b = payloads[left]["adapter_state"], payloads[right]["adapter_state"]
        x, y = flatten(list(a.values())), flatten(list(b.values()))
        pairwise.append({"left": left, "right": right, "parameter_l2_distance": float((x-y).norm()),
            "parameter_cosine_similarity": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "exact_tensor_equality": bool(torch.equal(x, y)), "equal_parameter_count": int((x == y).sum()), "different_parameter_count": int((x != y).sum())})
    for name, data, fields in (("checkpoint_inventory.csv", inventory, inventory[0].keys()), ("block_parameter_delta.csv", deltas, deltas[0].keys()), ("checkpoint_pairwise_distance.csv", pairwise, pairwise[0].keys())):
        with (args.output_root / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(fields)); w.writeheader(); w.writerows(data)
    nonzero = {row["condition"]: any(x["delta_l2_norm"] > 0 for x in deltas if x["condition"] == row["condition"]) for row in inventory}
    report = ["# Checkpoint Parameter Audit", "", f"Reference: `frozen`; checkpoints: {', '.join(sorted(payloads))}.", "",
              "| Condition | Non-zero delta vs frozen | Checkpoint hash |", "|---|---:|---|"]
    report += [f"| {row['condition']} | {nonzero[row['condition']]} | `{row['checkpoint_sha256'][:12]}` |" for row in inventory]
    report += ["", "Gradient and exact optimizer-step telemetry were not recorded by the original smoke runner. This is an audit finding: later Stage 0A/positive-control runs must persist them. Existing non-zero parameter deltas are sufficient to proceed to the forward-path audit, not to a scientific effectiveness claim."]
    (args.output_root / "checkpoint_parameter_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "conditions": sorted(payloads), "nonzero_delta": nonzero}, indent=2))


if __name__ == "__main__": main()
