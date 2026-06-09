#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path("V7-HP3")
REPORT_DIR = Path("实验分析报告/V7-HP3")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def agg(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        if str(r.get("status")) != "completed":
            continue
        groups[(r.get("method"), r.get("profile"))].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        item = {"method": key[0], "profile": key[1], "runs": len(vals)}
        for f in fields:
            item[f] = mean(float(v.get(f) or 0.0) for v in vals)
        out.append(item)
    return out


def table(rows, fields):
    if not rows:
        return "_暂无结果_\n"
    head = ["method", "profile", "runs"] + fields
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in rows:
        vals = []
        for h in head:
            v = r.get(h, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def best(rows, metric, pred):
    vals = [float(r.get(metric, 0.0) or 0.0) for r in rows if pred(r)]
    return max(vals) if vals else 0.0


def main():
    official = load_rows(ROOT / "outputs/hotpot_official_fullwiki_hard100/official_eval_all_summary.json")
    reader = load_rows(ROOT / "outputs/hotpot_reader_strong_hard100/reader_eval_all_summary.json")
    meta_path = ROOT / "data/hotpot_recoverable_hard100.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    off = agg(official, ["answer_access_at_k", "support_title_recall_at_k", "sp_f1", "joint_f1"])
    rea = agg(reader, ["answer_em", "answer_f1", "sp_f1", "joint_f1", "answer_access_at_k"])
    is_agent = lambda r: "agent" in str(r.get("profile", ""))
    is_base = lambda r: "baseline" in str(r.get("profile", "")) or str(r.get("method", "")).startswith("hypernet")
    off_gap = best(off, "joint_f1", is_agent) - best(off, "joint_f1", is_base)
    reader_gap = best(rea, "joint_f1", is_agent) - best(rea, "joint_f1", is_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = """# V7-HP3 Reset Hard-Reader 实验报告

生成时间: {now}

## 实验目的

HP3 是对 HP2 打平结果的三步重置：

- Reader 从 T5-small 提升到强 reader，默认 `google/flan-t5-large`，脚本也支持 Qwen/Llama causal reader。
- Reader-aware reward 改为阶梯式 high-contrast：top block +10，bottom block -5，中间 0。
- 从 HP2 per-query 成绩单反筛 Recoverable-Hard 100，用于 hard-case official 与 reader 分析。

## Hard100 筛选诊断

- 样本数: {actual_size}
- 严格 recoverable 数: {strict_count}
- 选入严格 recoverable 数: {selected_strict}
- fallback medium-hard 数: {fallback_count}
- easy removed: {easy_removed}
- impossible removed: {impossible_removed}

注意：严格 recoverable 若很少，说明 HP2 各方法 per-query 输出同质化；HP3 的主要检验点变成强 reader 与 step reward 能否在 hard100 上制造新的分离。

## Official Hard100

{official_table}

## Strong Reader Hard100

{reader_table}

## 初步判断

- Official agent-baseline best joint_f1 gap: {off_gap:+.4f}
- Reader agent-baseline best joint_f1 gap: {reader_gap:+.4f}

若 gap 为正且 reader-aware memory/tail 同时提高 support 与 answer F1，可作为 V7 agent 正信号；若仍打平，则说明 block-selection 对 retriever 表征影响被当前训练/评估链路抹平，需要进入更强训练分叉或真实 online reader reward。
""".format(
        now=datetime.now().isoformat(timespec="seconds"),
        actual_size=meta.get("actual_size"),
        strict_count=meta.get("strict_recoverable_count"),
        selected_strict=meta.get("selected_strict_count"),
        fallback_count=meta.get("fallback_medium_hard_count"),
        easy_removed=meta.get("easy_removed"),
        impossible_removed=meta.get("impossible_removed"),
        official_table=table(off, ["answer_access_at_k", "support_title_recall_at_k", "sp_f1", "joint_f1"]),
        reader_table=table(rea, ["answer_em", "answer_f1", "sp_f1", "joint_f1", "answer_access_at_k"]),
        off_gap=off_gap,
        reader_gap=reader_gap,
    )
    report = REPORT_DIR / f"v7_hp3_reset_hard_reader_{ts}.md"
    latest = REPORT_DIR / "v7_hp3_reset_hard_reader_latest.md"
    report.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    print(json.dumps({"report_path": str(report), "latest_path": str(latest), "official_runs": len(official), "reader_runs": len(reader), "official_gap": off_gap, "reader_gap": reader_gap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
