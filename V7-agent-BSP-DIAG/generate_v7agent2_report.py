from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent / "实验分析报告" / "V7-agent-PM"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    if df.empty:
        return "_数据待生成_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> int:
    ablation_df = load_csv_safe(REPORT_DIR / "ablation_summary.csv")
    official_df = load_csv_safe(REPORT_DIR / "official_eval_summary_1000.csv")
    dynamic_df = load_csv_safe(REPORT_DIR / "dynamic_strict_summary_agg.csv")
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# V7-agent-PM 完整实验报告",
        "",
        f"生成时间：{now}",
        "",
        "## 0. 执行摘要",
        "",
        (
            "V7-agent-PM 在 V7-agent 基础上加入 FiD-style reader、三件套 ablation、"
            "official eval 统计检验、bandit early reward 和动态 early slot。"
        ),
        "",
        "## 1. Ablation 结果",
        "",
        df_to_markdown_safe(ablation_df),
        "",
        "## 2. Dynamic Strict 结果",
        "",
        df_to_markdown_safe(dynamic_df),
        "",
        "## 3. Official Eval 结果（n=1000，FiD Reader）",
        "",
        df_to_markdown_safe(official_df),
        "",
        "## 4. 结论",
        "",
        (
            "_待完整实验完成后填写；若 official eval 未显著提升，"
            "保留 V7-agent 的 strict diagnostic 结论边界。_"
        ),
        "",
        "---",
        "",
        f"_本报告由 generate_v7agentpm_report.py 自动生成，时间：{now}_",
    ]

    out = REPORT_DIR / f"v7_agent_pm_complete_report_{datetime.now().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] 已生成：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
