from __future__ import annotations

import re
import shutil
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from experiment_config import REPORT_ROOT, UpstreamConfig
from metrics import ensure_dir, write_json
from summarize_results import summarize_run


def _safe_float(value, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _report_dir() -> Path:
    return ensure_dir(REPORT_ROOT)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


RUN_METADATA_FILES = [
    "upstream_config.json",
    "run_metadata.json",
    "round_logs.json",
    "round_logs.jsonl",
    "round_logs.csv",
    "final_artifacts.json",
]

HF_METADATA_FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.json",
]

DOWNSTREAM_ARTIFACT_FILES = [
    "rag_eval_command.json",
    "rag_eval_stdout.log",
    "rag_eval_stderr.log",
]


def _copy_hf_metadata(run_dir: Path, dst_root: Path) -> None:
    hf_model_dir = _find_hf_model_dir(run_dir)
    if not hf_model_dir:
        return
    src_dir = Path(hf_model_dir)
    dst_dir = ensure_dir(dst_root / src_dir.name)
    for name in HF_METADATA_FILES:
        _copy_if_exists(src_dir / name, dst_dir / name)


def _copy_run_snapshot(run_dir: Path, dst_root: Path) -> None:
    ensure_dir(dst_root)
    for name in RUN_METADATA_FILES:
        _copy_if_exists(run_dir / name, dst_root / name)
    _copy_hf_metadata(run_dir, dst_root)


def _copy_downstream_snapshot(output_dir: Path, dst_root: Path) -> None:
    ensure_dir(dst_root)
    for name in DOWNSTREAM_ARTIFACT_FILES:
        _copy_if_exists(output_dir / name, dst_root / name)


def _find_hf_model_dir(run_dir: Path) -> str:
    dirs = sorted(run_dir.glob("retriever_hf_*"))
    return str(dirs[-1]) if dirs else ""


def build_single_run_report(config: UpstreamConfig, run_dir: Path) -> Dict:
    summary = summarize_run(run_dir)
    summary = summary or {}
    hf_model_dir = _find_hf_model_dir(run_dir)
    return {
        "report_type": "single_run",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": config.experiment_name,
        "run_name": f"{config.task_name}/{run_dir.name}",
        "suite_tag": config.suite_tag,
        "task_name": config.task_name,
        "strategy": config.selection_strategy,
        "topk_blocks": config.topk_blocks,
        "warmup_rounds": config.warmup_rounds,
        "estimate_encryption": config.estimate_encryption,
        "rounds": _safe_int(summary.get("rounds", config.num_rounds)),
        "avg_payload_ratio": _safe_float(summary.get("avg_payload_ratio", 0.0)),
        "overall_payload_ratio": _safe_float(summary.get("overall_payload_ratio", 0.0)),
        "communication_reduction_ratio": _safe_float(1.0 - float(summary.get("overall_payload_ratio", 0.0))),
        "total_uploaded_params": _safe_int(summary.get("total_uploaded_params", 0)),
        "total_full_params": _safe_int(summary.get("total_full_params", 0)),
        "total_encrypted_bytes_est": _safe_float(summary.get("total_encrypted_bytes_est", 0.0), digits=2),
        "avg_hn_loss": _safe_float(summary.get("avg_hn_loss", 0.0)),
        "run_dir": str(run_dir),
        "hf_model_dir": hf_model_dir,
    }


def render_single_run_markdown(report: Dict) -> str:
    return f"""# 实验分析报告：{report['run_name']}

## 1. 实验概况

- 生成时间：{report['created_at']}
- 实验名称：{report['experiment_name']}
- 运行目录：`{report['run_dir']}`
- 上传策略：`{report['strategy']}`
- Top-K 参数块：{report['topk_blocks']}
- Warmup 轮数：{report['warmup_rounds']}
- 是否估算加密通信：{report['estimate_encryption']}

## 2. 核心结果

- 实际记录轮数：{report['rounds']}
- 平均 payload ratio：{report['avg_payload_ratio']:.4f}
- 整体 payload ratio：{report['overall_payload_ratio']:.4f}
- 通信压缩比例：{report['communication_reduction_ratio']:.4f}
- 累计上传参数量：{report['total_uploaded_params']}
- 全量上传参数量：{report['total_full_params']}
- 估算加密通信字节数：{report['total_encrypted_bytes_est']}
- 平均超网络损失：{report['avg_hn_loss']:.4f}

## 3. 自动分析

本次实验完成了联邦训练主流程，并成功输出通信统计与模型产物。

从通信角度看，整体 payload ratio 为 {report['overall_payload_ratio']:.4f}，意味着相对于全量上传，通信保留比例约为 {report['overall_payload_ratio']:.4f}，通信压缩比例约为 {report['communication_reduction_ratio']:.4f}。如果该值明显低于 1，则说明选择性上传策略已经开始发挥作用；如果接近 1，则说明当前配置下上传压缩仍然有限，后续可优先从 `topk_blocks`、warmup 轮数和重要性学习效果继续优化。

从方法行为看，本次策略为 `{report['strategy']}`。如果采用的是 `hypernet`，则平均超网络损失为 {report['avg_hn_loss']:.4f}，可以作为后续观察重要性学习是否逐步稳定的参考信号。

## 4. 下游衔接

- HF 检索器目录：`{report['hf_model_dir'] or '未检测到'}`

如果该目录存在，则可以进一步接入下游 `RAGTest` 做检索与问答效果验证，检查“通信下降后性能是否保持稳定”。
"""


def write_single_run_report(config: UpstreamConfig, run_dir: Path) -> Path:
    report = build_single_run_report(config, run_dir)
    report_root = _report_dir()
    stem = f"{config.experiment_name}_{run_dir.name}_{_timestamp()}"
    bundle_dir = ensure_dir(report_root / stem)
    data_dir = ensure_dir(bundle_dir / "data")
    json_path = bundle_dir / "report.json"
    md_path = bundle_dir / "report.md"
    write_json(json_path, report)
    md_path.write_text(render_single_run_markdown(report), encoding="utf-8")
    _copy_run_snapshot(run_dir, data_dir)
    return md_path


def build_suite_report(suite_name: str, configs: Iterable[UpstreamConfig]) -> Dict:
    reports: List[Dict] = []
    for config in configs:
        run_dir = Path(config.output_dir)
        if run_dir.exists():
            reports.append(build_single_run_report(config, run_dir))
    reports = [item for item in reports if item]
    if not reports:
        return {
            "report_type": "suite",
            "suite_name": suite_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runs": [],
        }

    sorted_by_payload = sorted(reports, key=lambda x: x["overall_payload_ratio"])
    sorted_by_upload = sorted(reports, key=lambda x: x["total_uploaded_params"])
    best_comm = sorted_by_payload[0]
    lowest_upload = sorted_by_upload[0]
    baseline = next((item for item in reports if item["strategy"] == "full"), None)

    return {
        "report_type": "suite",
        "suite_name": suite_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": reports,
        "best_communication_ratio_run": best_comm["run_name"],
        "lowest_upload_params_run": lowest_upload["run_name"],
        "baseline_full_upload_run": baseline["run_name"] if baseline else "",
    }


def render_suite_markdown(report: Dict) -> str:
    lines = [
        f"# 实验套件分析报告：{report['suite_name']}",
        "",
        "## 1. 总体概况",
        "",
        f"- 生成时间：{report['created_at']}",
        f"- 套件名称：`{report['suite_name']}`",
        f"- 运行数量：{len(report.get('runs', []))}",
    ]
    if report.get("runs"):
        lines.extend(
            [
                f"- 通信压缩最佳运行：`{report.get('best_communication_ratio_run', '')}`",
                f"- 上传参数量最低运行：`{report.get('lowest_upload_params_run', '')}`",
                f"- 全量上传基线：`{report.get('baseline_full_upload_run', '未包含')}`",
                "",
                "## 2. 各运行结果",
                "",
            ]
        )
        for item in report["runs"]:
            lines.append(
                f"- `{item['run_name']}`: strategy={item['strategy']}, overall_payload={item['overall_payload_ratio']:.4f}, "
                f"reduction={item['communication_reduction_ratio']:.4f}, uploaded={item['total_uploaded_params']}"
            )
        lines.extend(
            [
                "",
                "## 3. 自动分析",
                "",
                "该套件报告用于比较不同选择性上传策略或不同超参数设置下的通信效率差异。",
                "优先关注 `overall_payload_ratio` 与 `communication_reduction_ratio`，前者越低表示上传占比越小，后者越高表示压缩越明显。",
                "如果 `hypernet` 在保持较低 payload ratio 的同时没有明显异常的训练日志，则说明超网络在参数重要性评估上具备价值；如果 `random` 或 `static_top` 表现接近，则说明当前重要性学习优势仍需进一步放大。",
            ]
        )
    else:
        lines.extend(["", "## 2. 自动分析", "", "当前未检测到可汇总的运行结果。"])
    return "\n".join(lines) + "\n"


def write_suite_report(suite_name: str, configs: Iterable[UpstreamConfig]) -> Path:
    report = build_suite_report(suite_name, configs)
    report_root = _report_dir()
    stem = f"suite_{suite_name}_{_timestamp()}"
    bundle_dir = ensure_dir(report_root / stem)
    data_dir = ensure_dir(bundle_dir / "data")
    json_path = bundle_dir / "report.json"
    md_path = bundle_dir / "report.md"
    write_json(json_path, report)
    md_path.write_text(render_suite_markdown(report), encoding="utf-8")
    for item in report.get("runs", []):
        run_dir = Path(item["run_dir"])
        target_dir = data_dir / run_dir.name
        _copy_run_snapshot(run_dir, target_dir)
    return md_path


DOWNSTREAM_METRICS = [
    "cos_1",
    "cos_3",
    "cos_5",
    "cos_10",
    "recall_1",
    "recall_3",
    "recall_5",
    "recall_10",
    "precision",
    "precision_3",
    "precision_5",
    "precision_10",
    "hit_2",
    "hit_4",
    "hit_8",
    "F1",
    "em",
    "mrr",
    "hit1",
    "hit10",
    "MAP",
    "NDCG",
    "DCG",
    "IDCG",
]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_last_float(text: str, metric: str) -> float | None:
    matches = re.findall(rf"^{re.escape(metric)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text, flags=re.MULTILINE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def summarize_downstream_run(output_dir: Path, upstream_run_dir: Path | None = None, relative_run_name: str = "") -> Dict:
    stdout_text = _read_text(output_dir / "rag_eval_stdout.log")
    stderr_text = _read_text(output_dir / "rag_eval_stderr.log")
    has_traceback = "Traceback" in stderr_text
    metrics = {}
    for key in DOWNSTREAM_METRICS:
        value = _extract_last_float(stdout_text, key)
        if value is not None:
            metrics[key] = round(value, 4)
    return {
        "run_name": relative_run_name or output_dir.name,
        "output_dir": str(output_dir),
        "upstream_run_dir": str(upstream_run_dir) if upstream_run_dir else "",
        "status": "failed" if has_traceback else ("completed" if stdout_text else "missing"),
        "has_traceback": has_traceback,
        "stdout_bytes": len(stdout_text.encode("utf-8")),
        "stderr_bytes": len(stderr_text.encode("utf-8")),
        "metrics": metrics,
    }


def build_grouped_pipeline_summary(merged_runs: List[Dict]) -> List[Dict]:
    groups: dict[tuple, list[Dict]] = {}
    for item in merged_runs:
        key = (
            item.get("suite_tag", ""),
            item.get("task_name", ""),
            item.get("strategy", ""),
            item.get("topk_blocks", 0),
            item.get("warmup_rounds", 0),
            bool(item.get("estimate_encryption", False)),
        )
        groups.setdefault(key, []).append(item)

    grouped: list[Dict] = []
    for key, items in groups.items():
        suite_tag, task_name, strategy, topk_blocks, warmup_rounds, estimate_encryption = key
        record = {
            "suite_tag": suite_tag,
            "task_name": task_name,
            "strategy": strategy,
            "topk_blocks": topk_blocks,
            "warmup_rounds": warmup_rounds,
            "estimate_encryption": estimate_encryption,
            "seed_count": len(items),
            "overall_payload_ratio_mean": _mean([float(item.get("overall_payload_ratio", 0.0)) for item in items]),
            "overall_payload_ratio_std": _std([float(item.get("overall_payload_ratio", 0.0)) for item in items]),
            "communication_reduction_mean": _mean([1.0 - float(item.get("overall_payload_ratio", 0.0)) for item in items]),
            "communication_reduction_std": _std([1.0 - float(item.get("overall_payload_ratio", 0.0)) for item in items]),
        }
        for metric in ("cos_3", "recall_3", "mrr", "NDCG", "F1", "em"):
            values = [
                float(item.get("downstream_metrics", {}).get(metric))
                for item in items
                if metric in item.get("downstream_metrics", {})
            ]
            if values:
                record[f"{metric}_mean"] = _mean(values)
                record[f"{metric}_std"] = _std(values)
        grouped.append(record)
    return grouped


def build_full_pipeline_report(
    suite_name: str,
    upstream_root: Path,
    downstream_root: Path,
) -> Dict:
    upstream_runs = []
    downstream_runs = []
    for metadata_path in sorted(upstream_root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        summary = summarize_run(run_dir)
        if summary:
            summary["relative_run_name"] = str(run_dir.relative_to(upstream_root))
            upstream_runs.append(summary)
    for stdout_path in sorted(downstream_root.rglob("rag_eval_stdout.log")):
        output_dir = stdout_path.parent
        relative_run_name = str(output_dir.relative_to(downstream_root))
        upstream_run_dir = upstream_root / relative_run_name
        downstream_runs.append(
            summarize_downstream_run(
                output_dir,
                upstream_run_dir if upstream_run_dir.exists() else None,
                relative_run_name=relative_run_name,
            )
        )

    downstream_map = {item["run_name"]: item for item in downstream_runs}
    merged_runs = []
    for upstream in upstream_runs:
        downstream = downstream_map.get(upstream.get("relative_run_name", ""), {})
        merged_runs.append(
            {
                **upstream,
                "downstream_status": downstream.get("status", "missing"),
                "downstream_metrics": downstream.get("metrics", {}),
                "downstream_output_dir": downstream.get("output_dir", ""),
            }
        )
    grouped_runs = build_grouped_pipeline_summary(merged_runs)

    return {
        "report_type": "full_pipeline",
        "suite_name": suite_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "upstream_root": str(upstream_root),
        "downstream_root": str(downstream_root),
        "upstream_runs": upstream_runs,
        "downstream_runs": downstream_runs,
        "merged_runs": merged_runs,
        "grouped_runs": grouped_runs,
        "completed_upstream": sum(1 for item in upstream_runs if item),
        "completed_downstream": sum(1 for item in downstream_runs if item.get("status") == "completed"),
    }


def render_full_pipeline_markdown(report: Dict) -> str:
    lines = [
        f"# 全流程实验归档报告：{report['suite_name']}",
        "",
        "## 1. 运行概况",
        "",
        f"- 生成时间：{report['created_at']}",
        f"- 上游结果目录：`{report['upstream_root']}`",
        f"- 下游结果目录：`{report['downstream_root']}`",
        f"- 已完成上游实验数：{report['completed_upstream']}",
        f"- 已完成下游实验数：{report['completed_downstream']}",
        "",
        "## 2. 上下游对照",
        "",
    ]
    for item in report.get("merged_runs", []):
        metrics = item.get("downstream_metrics", {})
        metric_text = ", ".join(f"{key}={value}" for key, value in metrics.items() if key in ("cos_1", "cos_3", "recall_3", "mrr", "NDCG"))
        lines.append(
            f"- `{Path(item['run_dir']).name}`: strategy={item['strategy']}, overall_payload={item['overall_payload_ratio']:.4f}, "
            f"reduction={1.0 - item['overall_payload_ratio']:.4f}, downstream={item['downstream_status']}"
            + (f", {metric_text}" if metric_text else "")
        )
    if report.get("grouped_runs"):
        lines.extend(["", "## 3. Seed 聚合统计", ""])
        for item in report["grouped_runs"]:
            metric_text = ", ".join(
                f"{metric}={item[f'{metric}_mean']:.4f}±{item[f'{metric}_std']:.4f}"
                for metric in ("cos_3", "recall_3", "mrr", "NDCG")
                if f"{metric}_mean" in item
            )
            lines.append(
                f"- `{item['suite_tag']}/{item['task_name']}/{item['strategy']}`: "
                f"payload={item['overall_payload_ratio_mean']:.4f}±{item['overall_payload_ratio_std']:.4f}, "
                f"reduction={item['communication_reduction_mean']:.4f}±{item['communication_reduction_std']:.4f}"
                + (f", {metric_text}" if metric_text else "")
            )
    lines.extend(
        [
            "",
            "## 4. 自动分析",
            "",
            "本报告同时归档上游联邦训练与下游 RAG 检索评测结果，用于判断“通信压缩是否换来了可接受的下游性能保持”。",
            "优先关注上游 `overall_payload_ratio` 与下游 `cos_1/cos_3/recall_3/mrr/NDCG` 的联合变化。",
            "如果某一策略在显著降低 payload ratio 的同时仍保持稳定的下游命中率与排序指标，则说明该策略更适合作为通信高效的联邦检索方案。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_full_pipeline_report(
    suite_name: str,
    upstream_root: Path,
    downstream_root: Path,
) -> Path:
    report = build_full_pipeline_report(suite_name, upstream_root, downstream_root)
    bundle_dir = ensure_dir(_report_dir() / f"full_pipeline_{suite_name}_{_timestamp()}")
    data_dir = ensure_dir(bundle_dir / "data")
    json_path = bundle_dir / "report.json"
    md_path = bundle_dir / "report.md"
    write_json(json_path, report)
    md_path.write_text(render_full_pipeline_markdown(report), encoding="utf-8")
    _copy_if_exists(upstream_root / "summary.json", data_dir / "upstream_summary.json")
    _copy_if_exists(upstream_root / "summary.csv", data_dir / "upstream_summary.csv")
    _copy_if_exists(upstream_root / "summary_grouped.json", data_dir / "upstream_summary_grouped.json")
    _copy_if_exists(upstream_root / "summary_grouped.csv", data_dir / "upstream_summary_grouped.csv")
    _copy_if_exists(downstream_root / "rag_eval_all_summary.json", data_dir / "downstream_summary.json")
    for item in report.get("merged_runs", []):
        run_name = Path(item["run_dir"]).name
        _copy_run_snapshot(Path(item["run_dir"]), data_dir / "upstream" / run_name)
        if item.get("downstream_output_dir"):
            _copy_downstream_snapshot(Path(item["downstream_output_dir"]), data_dir / "downstream" / run_name)
    return md_path
