from __future__ import annotations

import shutil
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
        "run_name": run_dir.name,
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
    for name in [
        "upstream_config.json",
        "run_metadata.json",
        "round_logs.json",
        "round_logs.jsonl",
        "round_logs.csv",
        "final_artifacts.json",
    ]:
        _copy_if_exists(run_dir / name, data_dir / name)
    hf_model_dir = _find_hf_model_dir(run_dir)
    if hf_model_dir:
        _copy_if_exists(Path(hf_model_dir), data_dir / Path(hf_model_dir).name)
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
        ensure_dir(target_dir)
        for name in [
            "upstream_config.json",
            "run_metadata.json",
            "round_logs.json",
            "round_logs.jsonl",
            "round_logs.csv",
            "final_artifacts.json",
        ]:
            _copy_if_exists(run_dir / name, target_dir / name)
    return md_path
