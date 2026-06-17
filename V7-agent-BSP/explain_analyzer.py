from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from metrics import write_csv, write_json


def _load_round_logs(run_dir: Path) -> List[Dict]:
    path = run_dir / "round_logs.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_run_explanations(run_dir: Path) -> Dict:
    logs = _load_round_logs(run_dir)
    block_counter: Counter[str] = Counter()
    client_counter: Dict[int, Counter[str]] = defaultdict(Counter)
    budget_values: list[int] = []
    for row in logs:
        for detail in row.get("selection_details", []):
            client_id = int(detail.get("client_id", -1))
            budget_values.append(int(detail.get("budget_topk", 0)))
            for block in detail.get("upload_blocks", []):
                if block == "__ALL__":
                    continue
                block_counter[block] += 1
                client_counter[client_id][block] += 1
    top_blocks = [{"block": block, "count": count} for block, count in block_counter.most_common()]
    client_rows = []
    for client_id, counter in sorted(client_counter.items()):
        for block, count in counter.most_common():
            client_rows.append({"client_id": client_id, "block": block, "count": count})
    avg_budget = sum(budget_values) / len(budget_values) if budget_values else 0.0
    return {
        "run_dir": str(run_dir),
        "top_blocks": top_blocks,
        "client_block_counts": client_rows,
        "avg_budget_topk": avg_budget,
    }


def write_explain_artifacts(run_dir: Path, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or run_dir
    analysis = analyze_run_explanations(run_dir)
    json_path = Path(output_dir) / "explain_summary.json"
    csv_path = Path(output_dir) / "explain_client_block_counts.csv"
    write_json(json_path, analysis)
    write_csv(csv_path, analysis.get("client_block_counts", []))
    return json_path
