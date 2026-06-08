#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_rows(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    return [r for r in data if str(r.get('status', '')).startswith('completed')]


def aggregate(rows, metrics):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get('method'), r.get('profile'))].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = {'method': key[0], 'profile': key[1], 'runs': len(rs)}
        for m in metrics:
            vals = [float(r.get(m) or 0.0) for r in rs]
            rec[m] = sum(vals) / len(vals) if vals else 0.0
        out.append(rec)
    return out


def best(records, metric, pred=lambda r: True):
    candidates = [r for r in records if pred(r)]
    return max(candidates, key=lambda r: r.get(metric, 0.0)) if candidates else None


def table(records, fields):
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for r in records:
        vals = []
        for f in fields:
            v = r.get(f, '')
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def main() -> int:
    official_root = Path('V7-HP2/outputs/hotpot_official_fullwiki_dev300')
    reader_root = Path('V7-HP2/outputs/hotpot_reader_fullwiki_t5small_dev300')
    official_rows = load_rows(official_root / 'official_eval_all_summary.json')
    reader_rows = load_rows(reader_root / 'reader_eval_all_summary.json')
    official = aggregate(official_rows, ['answer_access_at_k', 'support_title_recall_at_k', 'all_gold_sp_retrieved_at_k', 'sp_f1', 'joint_f1'])
    reader = aggregate(reader_rows, ['answer_em', 'answer_f1', 'joint_f1', 'answer_access_at_k', 'sp_f1'])
    report_dir = Path('实验分析报告/V7-HP2')
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = report_dir / f'v7_hp2_reader_alignment_{ts}.md'
    latest = report_dir / 'v7_hp2_reader_alignment_latest.md'
    lines = [
        '# V7-HP2 Reader-Aware Agent 实验报告',
        '',
        f'生成时间: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## 实验配置',
        '',
        '- 训练套件: `hp2_reader_aligned`',
        '- 方法: baseline hypernet/adaptive, no-reader memory/tail, reader-aware memory/tail',
        '- Reader-aware reward: retrieval proxy 与 reader feedback proxy 混合进入 agent utility memory EMA',
        '- 下游评估: HotpotQA validation/dev 分层随机 300 条',
        '- Official fullwiki: supporting-fact / joint 指标',
        '- Reader: `google-t5/t5-small` 生成答案 EM/F1',
        '',
        '## Official Fullwiki Dev300',
        '',
    ]
    if official:
        lines.append(table(official, ['method', 'profile', 'runs', 'answer_access_at_k', 'support_title_recall_at_k', 'all_gold_sp_retrieved_at_k', 'sp_f1', 'joint_f1']))
    else:
        lines.append('尚无 official fullwiki 完成结果。')
    lines += ['', '## Reader Dev300', '']
    if reader:
        lines.append(table(reader, ['method', 'profile', 'runs', 'answer_em', 'answer_f1', 'joint_f1', 'answer_access_at_k', 'sp_f1']))
    else:
        lines.append('尚无 reader 完成结果。')
    lines += ['', '## 初步判断', '']
    if official:
        base = best(official, 'joint_f1', lambda r: str(r.get('method')) in {'hypernet_v6', 'adaptive_v6'} or str(r.get('profile','')).startswith('hp2_baseline'))
        reader_agent = best(official, 'joint_f1', lambda r: 'reader' in str(r.get('method','')) or 'reader' in str(r.get('profile','')))
        if base and reader_agent:
            lines.append(f"Official fullwiki 最佳 baseline joint_f1={base['joint_f1']:.4f}; 最佳 reader-aware agent `{reader_agent['method']}` joint_f1={reader_agent['joint_f1']:.4f}; 差值={reader_agent['joint_f1']-base['joint_f1']:+.4f}。")
    if reader:
        base_r = best(reader, 'joint_f1', lambda r: str(r.get('method')) in {'hypernet_v6', 'adaptive_v6'} or str(r.get('profile','')).startswith('hp2_baseline'))
        agent_r = best(reader, 'joint_f1', lambda r: 'reader' in str(r.get('method','')) or 'reader' in str(r.get('profile','')))
        if base_r and agent_r:
            lines.append(f"Reader 最佳 baseline joint_f1={base_r['joint_f1']:.4f}; 最佳 reader-aware agent `{agent_r['method']}` joint_f1={agent_r['joint_f1']:.4f}; 差值={agent_r['joint_f1']-base_r['joint_f1']:+.4f}。")
    lines.append('结论需以 full pipeline 完成后为准；若 reader-aware agent 只提升 supporting-fact 而 reader EM/F1 不升，则仍不能宣称 QA 成功。')
    text = '\n'.join(lines) + '\n'
    path.write_text(text, encoding='utf-8')
    latest.write_text(text, encoding='utf-8')
    print(json.dumps({'report_path': str(path), 'latest_path': str(latest), 'official_runs': len(official_rows), 'reader_runs': len(reader_rows)}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
