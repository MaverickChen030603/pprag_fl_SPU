#!/usr/bin/env python3
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "V7-HP-PAPER" / "cross_dataset_validation"
ALIGN = BASE / "2wiki_selector_alignment"
REPAIR = BASE / "2wiki_selector_repair_bm25_anchor"
AUDIT = BASE / "2wiki_positive_action_detectability_audit"

BEST_SELECTOR = "bm25_anchor_answer_neutral_selector"

FEATURES = [
    "support_proxy_delta_vs_bm25",
    "evidence_proxy_delta_vs_bm25",
    "title_bridge_score",
    "answer_risk_score",
    "safe_answer_prob",
    "positive_action_prob",
    "bm25_top1_preserved",
    "bm25_top2_preserved",
    "bm25_top3_preserved",
    "prefix2_preserved",
    "prefix3_preserved",
    "num_added_docs",
    "num_removed_docs",
    "bm25_score_delta",
    "lexical_score_delta",
]


def ensure_dirs():
    for rel in [
        "outputs/collected",
        "outputs/feature_margin",
        "outputs/candidate_pool",
        "outputs/selector_recall",
        "outputs/safety_predictor",
        "outputs/case_studies",
        "outputs/tables",
        "reports",
    ]:
        (AUDIT / rel).mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out) + "\n"


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def mean(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def as_num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def auc_univariate(pos, neg):
    pos = [float(x) for x in pos if x is not None]
    neg = [float(x) for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = ties = total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total


def rankdata(vals):
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(as_num(x), as_num(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def positive_label(row):
    return (
        as_num(row.get("answer_f1_delta_vs_bm25")) is not None
        and as_num(row.get("joint_f1_delta_vs_bm25")) is not None
        and as_num(row.get("evidence_f1_delta_vs_bm25")) is not None
        and as_num(row.get("answer_f1_delta_vs_bm25")) >= 0
        and as_num(row.get("joint_f1_delta_vs_bm25")) > 0
        and as_num(row.get("evidence_f1_delta_vs_bm25")) >= 0
    )


def action_score(row):
    return (
        2.0 * (as_num(row.get("positive_action_prob")) or 0.0)
        + 0.5 * (as_num(row.get("safe_answer_prob")) or 0.0)
        + 0.5 * (as_num(row.get("evidence_proxy_delta_vs_bm25")) or 0.0)
        + 0.25 * (as_num(row.get("support_proxy_delta_vs_bm25")) or 0.0)
        - 0.5 * (as_num(row.get("answer_risk_score")) or 0.0)
    )


def load_all():
    align_summary = read_json(ALIGN / "outputs/selector_smoke_300/summary.json")
    align_failure = read_json(ALIGN / "outputs/selector_smoke_300/failure_summary.json")
    oracle_summary = read_json(REPAIR / "outputs/oracle_gap_300/oracle_gap_summary.json")
    oracle_rows = read_jsonl(REPAIR / "outputs/oracle_gap_300/oracle_gap_rows.jsonl")
    action_rows = read_jsonl(REPAIR / "outputs/action_table_300/bm25_anchor_action_table_300.jsonl")
    safety_summary = read_json(REPAIR / "outputs/safety_predictor/safety_predictor_summary.json")
    pred_rows = read_jsonl(REPAIR / "outputs/safety_predictor/crossfit_predictions.jsonl")
    per_example = read_jsonl(REPAIR / "outputs/selector_smoke_300/per_example_delta.jsonl")
    repair_summary = read_json(REPAIR / "outputs/selector_smoke_300/summary.json")
    repair_failure = read_json(REPAIR / "outputs/diagnostics/failure_summary.json")

    selected = {
        r["query_id"]: r for r in per_example
        if r.get("method") == BEST_SELECTOR
    }
    selected_keys = {
        (qid, row.get("candidate_name"), row.get("candidate_family"))
        for qid, row in selected.items()
    }
    oracle_by_q = {r["query_id"]: r for r in oracle_rows}
    action_by_key = {(r.get("query_id"), r.get("candidate_name"), r.get("candidate_family")): r for r in action_rows}

    merged = []
    for r in pred_rows:
        row = dict(action_by_key.get((r.get("query_id"), r.get("candidate_name"), r.get("candidate_family")), {}))
        row.update(r)
        row["positive_vs_bm25"] = bool(positive_label(row))
        row["selected_by_best_no_leak_selector"] = (row.get("query_id"), row.get("candidate_name"), row.get("candidate_family")) in selected_keys
        row["selector_score_proxy"] = action_score(row)
        orow = oracle_by_q.get(row.get("query_id"), {})
        row["query_has_positive_vs_bm25"] = bool(orow.get("positive_vs_bm25"))
        row["best_candidate_family_for_query"] = orow.get("best_candidate_family")
        row["best_candidate_name_for_query"] = orow.get("best_candidate_name")
        merged.append(row)

    return {
        "align_summary": align_summary,
        "align_failure": align_failure,
        "oracle_summary": oracle_summary,
        "oracle_rows": oracle_rows,
        "action_rows": action_rows,
        "safety_summary": safety_summary,
        "pred_rows": pred_rows,
        "per_example": per_example,
        "repair_summary": repair_summary,
        "repair_failure": repair_failure,
        "selected": selected,
        "merged": merged,
    }


def collect_existing_results():
    ensure_dirs()
    data = load_all()
    best = data["repair_summary"]["methods"][BEST_SELECTOR]
    safety = data["safety_summary"]
    oracle = data["oracle_summary"]
    failure = data["repair_failure"].get("failure_distribution", {})
    summary = {
        "status": "complete",
        "num_queries": oracle.get("num_queries"),
        "num_actions": len(data["merged"]),
        "num_positive_vs_bm25_queries": oracle.get("num_queries_with_positive_vs_bm25"),
        "positive_vs_bm25_rate": oracle.get("positive_vs_bm25_rate"),
        "candidate_pool_no_positive_vs_bm25": failure.get("candidate_pool_no_positive_vs_bm25"),
        "oracle_best_answer_delta_vs_bm25": oracle.get("oracle_best_answer_delta_vs_bm25"),
        "oracle_best_evidence_delta_vs_bm25": oracle.get("oracle_best_evidence_delta_vs_bm25"),
        "oracle_best_joint_delta_vs_bm25": oracle.get("oracle_best_joint_delta_vs_bm25"),
        "best_no_leak_selector": BEST_SELECTOR,
        "best_no_leak_selector_delta_vs_bm25": {
            "answer_f1_delta_vs_bm25": best.get("answer_f1_delta_vs_bm25"),
            "evidence_f1_delta_vs_bm25": best.get("evidence_f1_delta_vs_bm25"),
            "evidence_recall_delta_vs_bm25": best.get("evidence_recall_delta_vs_bm25"),
            "joint_f1_delta_vs_bm25": best.get("joint_f1_delta_vs_bm25"),
        },
        "positive_vs_bm25_recall": best.get("positive_vs_bm25_recall"),
        "selected_effective_action_rate": best.get("selected_effective_action_rate"),
        "safety_predictor_auc": safety.get("answer_safe_auc"),
        "paper_positive_auc": safety.get("paper_positive_auc"),
    }
    write_json(AUDIT / "outputs/collected/2wiki_collected_summary.json", summary)
    write_jsonl(AUDIT / "outputs/collected/2wiki_action_outcome_table.jsonl", data["merged"])
    return summary


def group_stats(rows, feature):
    vals = [as_num(r.get(feature)) for r in rows]
    vals = [v for v in vals if v is not None]
    return {"mean": mean(vals), "std": std(vals), "n": len(vals)}


def analyze_feature_margin():
    ensure_dirs()
    data = load_all()
    rows = data["merged"]
    pos = [r for r in rows if r["positive_vs_bm25"]]
    non = [r for r in rows if not r["positive_vs_bm25"]]
    selected = [r for r in rows if r["selected_by_best_no_leak_selector"]]
    rejected_pos = [r for r in pos if not r["selected_by_best_no_leak_selector"]]
    wrong_selected = [r for r in selected if not r["positive_vs_bm25"] and r.get("candidate_name") != "bm25_fallback"]
    feature_rows = []
    for feat in FEATURES:
        pvals = [as_num(r.get(feat)) for r in pos]
        nvals = [as_num(r.get(feat)) for r in non]
        pvals = [v for v in pvals if v is not None]
        nvals = [v for v in nvals if v is not None]
        pm, nm = mean(pvals), mean(nvals)
        ps, ns = std(pvals), std(nvals)
        pooled = math.sqrt(((ps or 0) ** 2 + (ns or 0) ** 2) / 2.0) if pvals and nvals else 0.0
        eff = (pm - nm) / pooled if pooled else 0.0
        feature_rows.append({
            "feature": feat,
            "positive_mean": pm,
            "non_positive_mean": nm,
            "mean_difference": None if pm is None or nm is None else pm - nm,
            "standardized_effect_size": eff,
            "auc_univariate": auc_univariate(pvals, nvals),
            "rank_correlation_with_joint_delta": spearman([r.get(feat) for r in rows], [r.get("joint_f1_delta_vs_bm25") for r in rows]),
            "rank_correlation_with_evidence_delta": spearman([r.get(feat) for r in rows], [r.get("evidence_f1_delta_vs_bm25") for r in rows]),
            "rank_correlation_with_answer_delta": spearman([r.get(feat) for r in rows], [r.get("answer_f1_delta_vs_bm25") for r in rows]),
        })

    fam = []
    for family, rs in sorted(defaultdict(list, ((k, [r for r in rows if r.get("candidate_family") == k]) for k in set(r.get("candidate_family") for r in rows))).items()):
        fam.append({
            "candidate_family": family,
            "n": len(rs),
            "positive_rate": sum(r["positive_vs_bm25"] for r in rs) / len(rs) if rs else 0,
            "answer_drop_rate": sum((as_num(r.get("answer_f1_delta_vs_bm25")) or 0) < 0 for r in rs) / len(rs) if rs else 0,
        })

    weak = sum((abs(r["standardized_effect_size"]) < 0.2 and (r["auc_univariate"] is None or 0.4 < r["auc_univariate"] < 0.6)) for r in feature_rows)
    summary = {
        "status": "complete",
        "num_actions": len(rows),
        "num_positive_actions": len(pos),
        "num_non_positive_actions": len(non),
        "num_selected_actions": len(selected),
        "num_rejected_positive_actions": len(rejected_pos),
        "num_wrong_selected_actions": len(wrong_selected),
        "feature_margin": feature_rows,
        "candidate_family_margin": fam,
        "positive_vs_rejected_positive": {
            feat: {"selected_positive": group_stats([r for r in pos if r["selected_by_best_no_leak_selector"]], feat),
                   "rejected_positive": group_stats(rejected_pos, feat)}
            for feat in FEATURES
        },
        "selected_wrong_vs_rejected_positive": {
            feat: {"wrong_selected": group_stats(wrong_selected, feat),
                   "rejected_positive": group_stats(rejected_pos, feat)}
            for feat in FEATURES
        },
        "interpretation": "positive actions are weakly distinguishable with current features" if weak >= len(FEATURES) * 0.5 else "some current features contain signal, but recall remains poor",
    }
    write_json(AUDIT / "outputs/feature_margin/feature_margin_summary.json", summary)

    table = md_table(
        ["feature", "pos_mean", "non_pos_mean", "diff", "effect_size", "auc", "rho_joint", "rho_evidence", "rho_answer"],
        [[r["feature"], fmt(r["positive_mean"]), fmt(r["non_positive_mean"]), fmt(r["mean_difference"]),
          fmt(r["standardized_effect_size"]), fmt(r["auc_univariate"]), fmt(r["rank_correlation_with_joint_delta"]),
          fmt(r["rank_correlation_with_evidence_delta"]), fmt(r["rank_correlation_with_answer_delta"])]
         for r in feature_rows]
    )
    (AUDIT / "outputs/tables/positive_action_feature_margin_table.md").write_text(table)
    cmp_rows = []
    for feat in FEATURES:
        sp = summary["positive_vs_rejected_positive"][feat]["selected_positive"]["mean"]
        rp = summary["positive_vs_rejected_positive"][feat]["rejected_positive"]["mean"]
        ws = summary["selected_wrong_vs_rejected_positive"][feat]["wrong_selected"]["mean"]
        cmp_rows.append([feat, fmt(sp), fmt(rp), fmt(ws)])
    (AUDIT / "outputs/tables/rejected_positive_comparison_table.md").write_text(
        md_table(["feature", "selected_positive_mean", "rejected_positive_mean", "wrong_selected_mean"], cmp_rows)
    )
    return summary


def analyze_candidate_pool():
    ensure_dirs()
    data = load_all()
    rows = data["merged"]
    byq = defaultdict(list)
    for r in rows:
        byq[r["query_id"]].append(r)
    pos_by_q = {q: [r for r in rs if r["positive_vs_bm25"]] for q, rs in byq.items()}
    dist = Counter(len(v) for v in pos_by_q.values())
    fam = []
    for family in sorted(set(r.get("candidate_family") for r in rows)):
        rs = [r for r in rows if r.get("candidate_family") == family]
        fam.append({
            "candidate_family": family,
            "n": len(rs),
            "positive_rate": sum(r["positive_vs_bm25"] for r in rs) / len(rs),
            "answer_drop_rate": sum((as_num(r.get("answer_f1_delta_vs_bm25")) or 0) < 0 for r in rs) / len(rs),
            "joint_positive_rate": sum((as_num(r.get("joint_f1_delta_vs_bm25")) or 0) > 0 for r in rs) / len(rs),
            "evidence_positive_rate": sum((as_num(r.get("evidence_f1_delta_vs_bm25")) or 0) > 0 for r in rs) / len(rs),
        })
    no_pos = [q for q, ps in pos_by_q.items() if not ps]
    bm25_strong = 0
    evidence_without_joint = 0
    answer_without_evidence = 0
    for q, rs in byq.items():
        if not pos_by_q[q]:
            bm25 = next((r for r in rs if r.get("candidate_name") in ("bm25_fallback", "bm25_no_change_control")), rs[0])
            if (as_num(bm25.get("answer_f1")) or 0) >= 0.75 and (as_num(bm25.get("evidence_f1")) or 0) >= 0.75:
                bm25_strong += 1
        if any((as_num(r.get("evidence_f1_delta_vs_bm25")) or 0) > 0 and (as_num(r.get("joint_f1_delta_vs_bm25")) or 0) <= 0 for r in rs):
            evidence_without_joint += 1
        if any((as_num(r.get("answer_f1_delta_vs_bm25")) or 0) > 0 and (as_num(r.get("evidence_f1_delta_vs_bm25")) or 0) <= 0 for r in rs):
            answer_without_evidence += 1
    oracle_positive_q = set(r["query_id"] for r in data["oracle_rows"] if r.get("positive_vs_bm25"))
    summary = {
        "status": "complete",
        "num_queries": len(byq),
        "queries_with_positive_vs_bm25": len(oracle_positive_q),
        "queries_without_positive_vs_bm25": len(byq) - len(oracle_positive_q),
        "positive_vs_bm25_rate": len(oracle_positive_q) / len(byq),
        "strict_action_label_queries_with_positive": sum(1 for ps in pos_by_q.values() if ps),
        "strict_action_label_queries_without_positive": len(no_pos),
        "strict_action_label_positive_rate": sum(1 for ps in pos_by_q.values() if ps) / len(byq),
        "positive_actions_per_query_distribution": dict(sorted(dist.items())),
        "candidate_family_positive_rate": fam,
        "BM25_already_strong_cases": bm25_strong,
        "no_added_doc_better_than_BM25_tail_cases": len(byq) - len(oracle_positive_q),
        "evidence_gain_without_joint_gain_cases": evidence_without_joint,
        "answer_gain_without_evidence_gain_cases": answer_without_evidence,
        "interpretation": "The dominant bottleneck on 2Wiki is not severe answer-anchor disruption after BM25 anchoring, but the lack of positive actions beyond a strong BM25 baseline in the current candidate pool.",
    }
    write_json(AUDIT / "outputs/candidate_pool/candidate_pool_vs_bm25_summary.json", summary)
    (AUDIT / "outputs/tables/candidate_pool_vs_bm25_table.md").write_text(md_table(
        ["metric", "value"],
        [[k, fmt(v)] for k, v in summary.items() if k not in ("candidate_family_positive_rate", "positive_actions_per_query_distribution")]
    ))
    (AUDIT / "outputs/tables/candidate_family_positive_rate_table.md").write_text(md_table(
        ["family", "n", "positive_rate", "answer_drop_rate", "joint_positive_rate", "evidence_positive_rate"],
        [[r["candidate_family"], r["n"], fmt(r["positive_rate"]), fmt(r["answer_drop_rate"]), fmt(r["joint_positive_rate"]), fmt(r["evidence_positive_rate"])] for r in fam]
    ))
    return summary


def analyze_selector_recall():
    ensure_dirs()
    data = load_all()
    rows = data["merged"]
    byq = defaultdict(list)
    for r in rows:
        byq[r["query_id"]].append(r)
    oracle_positive_qs = [r["query_id"] for r in data["oracle_rows"] if r.get("positive_vs_bm25")]
    positive_qs = [q for q, rs in byq.items() if any(r["positive_vs_bm25"] for r in rs)]
    positive_selected = []
    positive_missed = []
    wrong_selected = []
    fallback_positive = []
    answer_drop_selected = []
    rank_dist = Counter()
    group_rows = defaultdict(list)
    for q in positive_qs:
        rs = byq[q]
        sorted_rs = sorted(rs, key=action_score, reverse=True)
        best_pos = max([r for r in rs if r["positive_vs_bm25"]], key=action_score)
        best_rank = sorted_rs.index(best_pos) + 1
        rank_dist[best_rank] += 1
        sel = next((r for r in rs if r["selected_by_best_no_leak_selector"]), None)
        if sel and sel["positive_vs_bm25"]:
            positive_selected.append(q)
            group_rows["positive_selected"].append(sel)
        else:
            positive_missed.append(q)
            group_rows["positive_available_but_not_selected"].append(best_pos)
            if sel:
                if sel.get("candidate_name") == "bm25_fallback":
                    fallback_positive.append(q)
                    group_rows["fallback_when_positive_available"].append(sel)
                else:
                    wrong_selected.append(q)
                    group_rows["wrong_action_selected"].append(sel)
                if (as_num(sel.get("answer_f1_delta_vs_bm25")) or 0) < 0:
                    answer_drop_selected.append(q)
                    group_rows["answer_drop_selected"].append(sel)
    def avg_features(rs):
        return {f: mean([as_num(r.get(f)) for r in rs]) for f in FEATURES + ["selector_score_proxy"]}
    summary = {
        "status": "complete",
        "oracle_positive_query_count": len(oracle_positive_qs),
        "strict_action_label_positive_query_count": len(positive_qs),
        "positive_query_count": len(positive_qs),
        "positive_selected_count": len(positive_selected),
        "positive_recall": len(positive_selected) / len(positive_qs) if positive_qs else 0,
        "positive_available_but_not_selected_count": len(positive_missed),
        "oracle_positive_but_no_strict_positive_action_in_anchor_table": len(set(oracle_positive_qs) - set(positive_qs)),
        "fallback_on_positive_query_count": len(fallback_positive),
        "wrong_action_on_positive_query_count": len(wrong_selected),
        "answer_drop_selected_count": len(answer_drop_selected),
        "best_positive_action_predicted_rank_distribution": dict(sorted(rank_dist.items())),
        "group_feature_means": {k: avg_features(v) for k, v in group_rows.items()},
        "threshold_blocked_positives": sum(1 for q in positive_qs for r in byq[q] if r["positive_vs_bm25"] and (as_num(r.get("positive_action_prob")) or 0) < 0.1),
        "safety_blocked_positives": sum(1 for q in positive_qs for r in byq[q] if r["positive_vs_bm25"] and (as_num(r.get("safe_answer_prob")) or 0) < 0.5),
        "support_proxy_misranked_positives": sum(1 for q in positive_qs if max(byq[q], key=lambda r: as_num(r.get("support_proxy_delta_vs_bm25")) or -999).get("positive_vs_bm25") is False),
        "interpretation": "feature/ranker cannot identify positives reliably" if len(positive_selected) / len(positive_qs) < 0.1 else "some positive actions are recoverable, but recall is still low",
    }
    write_json(AUDIT / "outputs/selector_recall/selector_recall_failure_summary.json", summary)
    (AUDIT / "outputs/tables/selector_recall_failure_table.md").write_text(md_table(
        ["metric", "value"],
        [[k, fmt(v)] for k, v in summary.items() if k not in ("group_feature_means", "best_positive_action_predicted_rank_distribution")]
    ))
    (AUDIT / "outputs/tables/best_positive_rank_distribution.md").write_text(md_table(
        ["rank", "count"], [[k, v] for k, v in summary["best_positive_action_predicted_rank_distribution"].items()]
    ))
    return summary


def analyze_safety():
    ensure_dirs()
    data = load_all()
    rows = data["merged"]
    safe = [r for r in rows if r.get("answer_safe") == 1]
    unsafe = [r for r in rows if r.get("answer_safe") == 0]
    positive = [r for r in rows if r["positive_vs_bm25"]]
    non_positive = [r for r in rows if not r["positive_vs_bm25"]]
    answer_drop = [r for r in rows if (as_num(r.get("answer_f1_delta_vs_bm25")) or 0) < 0]
    false_safe = [r for r in rows if (as_num(r.get("safe_answer_prob")) or 0) >= 0.5 and r.get("answer_safe") == 0]
    false_negative = [r for r in rows if (as_num(r.get("safe_answer_prob")) or 0) < 0.5 and r.get("answer_safe") == 1]
    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        br = [r for r in rows if lo <= (as_num(r.get("safe_answer_prob")) or 0) < hi or (i == 9 and (as_num(r.get("safe_answer_prob")) or 0) == 1.0)]
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(br), "answer_safe_rate": mean([r.get("answer_safe") for r in br])})
    summary = {
        "status": "complete",
        "answer_safe_auc": data["safety_summary"].get("answer_safe_auc"),
        "paper_positive_auc": data["safety_summary"].get("paper_positive_auc"),
        "answer_safe_class_balance": {"safe": len(safe), "unsafe": len(unsafe), "safe_rate": len(safe) / len(rows)},
        "paper_positive_class_balance": {"positive": len(positive), "non_positive": len(non_positive), "positive_rate": len(positive) / len(rows)},
        "false_safe_cases": len(false_safe),
        "false_negative_cases": len(false_negative),
        "false_safe_rate": len(false_safe) / len(rows),
        "false_negative_rate": len(false_negative) / len(rows),
        "calibration_curve": bins,
        "safe_answer_prob_distribution": {
            "positive_mean": mean([r.get("safe_answer_prob") for r in positive]),
            "non_positive_mean": mean([r.get("safe_answer_prob") for r in non_positive]),
            "answer_drop_mean": mean([r.get("safe_answer_prob") for r in answer_drop]),
        },
        "interpretation": "do not claim 2Wiki has a reliable safety predictor; AUC remains near 0.55",
    }
    write_json(AUDIT / "outputs/safety_predictor/safety_predictor_weakness_summary.json", summary)
    (AUDIT / "outputs/tables/safety_predictor_error_table.md").write_text(md_table(
        ["metric", "value"],
        [["answer_safe_auc", fmt(summary["answer_safe_auc"])], ["paper_positive_auc", fmt(summary["paper_positive_auc"])],
         ["false_safe_cases", summary["false_safe_cases"]], ["false_negative_cases", summary["false_negative_cases"]],
         ["positive_safe_prob_mean", fmt(summary["safe_answer_prob_distribution"]["positive_mean"])],
         ["non_positive_safe_prob_mean", fmt(summary["safe_answer_prob_distribution"]["non_positive_mean"])],
         ["answer_drop_safe_prob_mean", fmt(summary["safe_answer_prob_distribution"]["answer_drop_mean"])]]
    ))
    return summary


def export_cases():
    ensure_dirs()
    data = load_all()
    rows = data["merged"]
    byq = defaultdict(list)
    for r in rows:
        byq[r["query_id"]].append(r)
    def case_from(row, diagnosis):
        rs = byq[row["query_id"]]
        best_pos = next(iter(sorted([r for r in rs if r["positive_vs_bm25"]], key=lambda r: as_num(r.get("joint_f1_delta_vs_bm25")) or -999, reverse=True)), None)
        sel = next((r for r in rs if r["selected_by_best_no_leak_selector"]), None)
        src = sel or row
        return {
            "query_id": row.get("query_id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "bm25_titles": row.get("bm25_titles"),
            "selected_titles": src.get("candidate_titles") or src.get("selected_titles"),
            "best_positive_titles": best_pos.get("candidate_titles") if best_pos else None,
            "added_titles": src.get("added_titles"),
            "removed_titles": src.get("removed_titles"),
            "answer_f1_delta_vs_bm25": src.get("answer_f1_delta_vs_bm25"),
            "evidence_f1_delta_vs_bm25": src.get("evidence_f1_delta_vs_bm25"),
            "joint_f1_delta_vs_bm25": src.get("joint_f1_delta_vs_bm25"),
            "safe_answer_prob": src.get("safe_answer_prob"),
            "positive_action_prob": src.get("positive_action_prob"),
            "support_proxy_delta_vs_bm25": src.get("support_proxy_delta_vs_bm25"),
            "title_bridge_score": src.get("title_bridge_score"),
            "answer_risk_score": src.get("answer_risk_score"),
            "diagnosis": diagnosis,
        }
    selected_pos = [r for r in rows if r["positive_vs_bm25"] and r["selected_by_best_no_leak_selector"]]
    missed_pos = [r for r in rows if r["positive_vs_bm25"] and not r["selected_by_best_no_leak_selector"]]
    no_pos_q = [q for q, rs in byq.items() if not any(r["positive_vs_bm25"] for r in rs)]
    bm25_strong = []
    for q in no_pos_q:
        bm = next((r for r in byq[q] if r.get("candidate_name") == "bm25_fallback"), byq[q][0])
        bm25_strong.append(bm)
    answer_drop = [r for r in rows if r["selected_by_best_no_leak_selector"] and (as_num(r.get("answer_f1_delta_vs_bm25")) or 0) < 0]
    cases = {
        "positive_selected": [case_from(r, "positive action selected by best no-leak selector") for r in selected_pos[:5]],
        "positive_missed": [case_from(r, "positive action available but selector did not choose it") for r in missed_pos[:5]],
        "bm25_already_strong": [case_from(r, "no candidate action improves over the strong BM25 baseline") for r in bm25_strong[:5]],
        "answer_drop": [case_from(r, "selected action reduced answer F1 versus BM25") for r in answer_drop[:3]],
    }
    write_json(AUDIT / "outputs/case_studies/case_studies.json", cases)
    for key, title in [
        ("positive_selected", "Positive action selected"),
        ("positive_missed", "Positive action available but not selected"),
        ("bm25_already_strong", "BM25 already strong / no positive action"),
        ("answer_drop", "Answer drop selected"),
    ]:
        lines = [f"# {title}", ""]
        for c in cases[key]:
            lines += [
                f"## {c['query_id']}",
                f"- Question: {c.get('question')}",
                f"- Answer: {c.get('answer')}",
                f"- BM25 titles: {c.get('bm25_titles')}",
                f"- Selected titles: {c.get('selected_titles')}",
                f"- Best positive titles: {c.get('best_positive_titles')}",
                f"- Added: {c.get('added_titles')}",
                f"- Removed: {c.get('removed_titles')}",
                f"- Deltas: answer={fmt(c.get('answer_f1_delta_vs_bm25'))}, evidence={fmt(c.get('evidence_f1_delta_vs_bm25'))}, joint={fmt(c.get('joint_f1_delta_vs_bm25'))}",
                f"- Prob/features: safe={fmt(c.get('safe_answer_prob'))}, positive={fmt(c.get('positive_action_prob'))}, support_proxy={fmt(c.get('support_proxy_delta_vs_bm25'))}, bridge={fmt(c.get('title_bridge_score'))}, risk={fmt(c.get('answer_risk_score'))}",
                f"- Diagnosis: {c.get('diagnosis')}",
                "",
            ]
        filename = {
            "positive_selected": "positive_selected_cases.md",
            "positive_missed": "positive_missed_cases.md",
            "bm25_already_strong": "bm25_already_strong_cases.md",
            "answer_drop": "answer_drop_cases.md",
        }[key]
        (AUDIT / "outputs/case_studies" / filename).write_text("\n".join(lines))
    return cases


def export_tables():
    ensure_dirs()
    data = load_all()
    methods = data["repair_summary"]["methods"]
    rows = []
    role = {
        "bm25_or_lexical_routing": "strong baseline",
        "previous_2wiki_v23_crossfit_selector": "failed selector transfer",
        BEST_SELECTOR: "best no-leak repair, diagnostic only",
        "oracle_diagnostic_only": "oracle upper bound, not inference-time",
    }
    for m in ["bm25_or_lexical_routing", "previous_2wiki_v23_crossfit_selector", BEST_SELECTOR, "oracle_diagnostic_only"]:
        s = methods[m]
        rows.append([m, fmt(s["answer_f1"]), fmt(s["evidence_f1"]), fmt(s["joint_f1"]),
                     fmt(s["answer_f1_delta_vs_bm25"]), fmt(s["evidence_f1_delta_vs_bm25"]),
                     fmt(s["joint_f1_delta_vs_bm25"]), role[m]])
    (AUDIT / "outputs/tables/2wiki_main_status_table.md").write_text(md_table(
        ["method", "answer_f1", "evidence_f1", "joint_f1", "answer_delta_vs_BM25", "evidence_delta_vs_BM25", "joint_delta_vs_BM25", "paper_role"], rows
    ))
    o = data["oracle_summary"]
    (AUDIT / "outputs/tables/2wiki_oracle_gap_table.md").write_text(md_table(
        ["positive_vs_BM25_queries", "positive_rate", "oracle_best_answer_delta", "oracle_best_evidence_delta", "oracle_best_joint_delta", "selector_positive_recall"],
        [[o["num_queries_with_positive_vs_bm25"], fmt(o["positive_vs_bm25_rate"]), fmt(o["oracle_best_answer_delta_vs_bm25"]), fmt(o["oracle_best_evidence_delta_vs_bm25"]), fmt(o["oracle_best_joint_delta_vs_bm25"]), fmt(o["selector_recall_of_positive_vs_bm25"])]]
    ))
    rep_rows = []
    for m in ["bm25_anchor_support_first", "bm25_anchor_safety_first", "bm25_anchor_positive_selector", BEST_SELECTOR]:
        s = methods[m]
        rep_rows.append([m, fmt(s["answer_f1_delta_vs_bm25"]), fmt(s["evidence_f1_delta_vs_bm25"]), fmt(s["joint_f1_delta_vs_bm25"]), fmt(s["selected_effective_action_rate"]), fmt(s["positive_vs_bm25_recall"])])
    (AUDIT / "outputs/tables/2wiki_selector_repair_table.md").write_text(md_table(
        ["method", "answer_delta", "evidence_delta", "joint_delta", "effective_rate", "positive_recall"], rep_rows
    ))
    fdist = data["repair_failure"]["failure_distribution"]
    interp = {
        "candidate_pool_no_positive_vs_bm25": "candidate generation rarely beats strong BM25",
        "positive_vs_bm25_available_but_not_selected": "selector/ranker misses available positive actions",
        "answer_drop_selected": "answer-anchor disruption after repair is rare but not zero",
    }
    (AUDIT / "outputs/tables/2wiki_failure_summary_table.md").write_text(md_table(
        ["failure_type", "count", "interpretation"],
        [[k, v, interp.get(k, "")] for k, v in fdist.items()]
    ))
    fm_path = AUDIT / "outputs/tables/positive_action_feature_margin_table.md"
    if fm_path.exists():
        (AUDIT / "outputs/tables/2wiki_feature_margin_table.md").write_text(fm_path.read_text())


def write_reports():
    ensure_dirs()
    # Ensure all summaries exist.
    collect = read_json(AUDIT / "outputs/collected/2wiki_collected_summary.json")
    feature = read_json(AUDIT / "outputs/feature_margin/feature_margin_summary.json")
    pool = read_json(AUDIT / "outputs/candidate_pool/candidate_pool_vs_bm25_summary.json")
    recall = read_json(AUDIT / "outputs/selector_recall/selector_recall_failure_summary.json")
    safety = read_json(AUDIT / "outputs/safety_predictor/safety_predictor_weakness_summary.json")
    feature_top = sorted(feature["feature_margin"], key=lambda r: abs(r.get("standardized_effect_size") or 0), reverse=True)[:6]
    report = f"""# 2Wiki Positive Action Detectability Audit

## 1. Executive Summary

This audit does not run 2Wiki-1000 reader validation and does not modify the frozen HotpotQA v2.3 result. It diagnoses why the current no-leak selector fails to reliably identify actions that outperform a strong BM25 baseline on 2Wiki dev-300.

Main conclusion: positive actions beyond BM25 exist, but they are sparse and weakly captured by the current feature/ranker stack. The best BM25-anchor no-leak selector nearly matches BM25 but does not provide enough reliable margin for full 1000-sample validation.

## 2. Existing 2Wiki Results

- Positive-vs-BM25 queries: {collect['num_positive_vs_bm25_queries']} / {collect['num_queries']} ({collect['positive_vs_bm25_rate']:.4f})
- Best no-leak selector: `{collect['best_no_leak_selector']}`
- Best no-leak joint delta vs BM25: {collect['best_no_leak_selector_delta_vs_bm25']['joint_f1_delta_vs_bm25']:.4f}
- Selected effective action rate: {collect['selected_effective_action_rate']:.4f}
- Positive-vs-BM25 recall: {collect['positive_vs_bm25_recall']:.4f}

## 3. Oracle Opportunity beyond BM25

Oracle diagnostics show a non-trivial opportunity: oracle answer delta {collect['oracle_best_answer_delta_vs_bm25']:.4f}, evidence delta {collect['oracle_best_evidence_delta_vs_bm25']:.4f}, and joint delta {collect['oracle_best_joint_delta_vs_bm25']:.4f}. This remains diagnostic only and is not an inference-time method.

## 4. Candidate Pool Limitation

The candidate pool is the dominant bottleneck: oracle diagnostics mark {pool['queries_without_positive_vs_bm25']} / {pool['num_queries']} queries as having no positive action beyond BM25. Under the stricter action-level label available in the BM25-anchor action table, only {pool['strict_action_label_queries_with_positive']} / {pool['num_queries']} queries expose a positive action to the current selector features. This supports the interpretation that future work should improve candidate generation beyond BM25 rather than simply tuning selector thresholds.

## 5. Positive Action Feature Margin

Feature detectability summary: {feature['interpretation']}.

Top absolute univariate effects:

{md_table(['feature','effect_size','auc','rho_joint'], [[r['feature'], fmt(r['standardized_effect_size']), fmt(r['auc_univariate']), fmt(r['rank_correlation_with_joint_delta'])] for r in feature_top])}

## 6. Selector Recall Failure

Among strict action-labeled positive-vs-BM25 queries, selector positive recall is {recall['positive_recall']:.4f} ({recall['positive_selected_count']} / {recall['strict_action_label_positive_query_count']}). At the broader oracle-query level, {recall['oracle_positive_but_no_strict_positive_action_in_anchor_table']} / {recall['oracle_positive_query_count']} oracle-positive queries do not expose a strict positive action inside the BM25-anchor table used by the no-leak selector. Missed positives therefore reflect both candidate/action mismatch and ranker weakness.

Best-positive predicted rank distribution:

{md_table(['rank','count'], [[k,v] for k,v in recall['best_positive_action_predicted_rank_distribution'].items()])}

## 7. Safety Predictor Weakness

The safety predictor is weak cross-dataset: answer-safe AUC {safety['answer_safe_auc']:.4f}, paper-positive AUC {safety['paper_positive_auc']:.4f}. Its probabilities should not be used as evidence that answer-neutral calibration transfers reliably to 2Wiki.

## 8. Case Studies

Case-study files were exported under `outputs/case_studies/`, covering selected positives, missed positives, BM25-strong cases, and answer-drop selections.

## 9. Paper Recommendation

Freeze HotpotQA v2.3 as the main result. Use 2Wiki as external diagnostic / limitation and appendix evidence, not as main selector-level generalization success. The paper-safe conclusion is that 2Wiki validates the adapter, reader-backed smoke pipeline, and lexical-routing sanity check, while exposing a cross-dataset selector detectability limitation.
"""
    (AUDIT / "reports/2wiki_positive_action_detectability_report.md").write_text(report)

    limitation = """# 2Wiki Limitation Section Draft

We further tested the pipeline on 2WikiMultiHopQA as an external sanity check. A strong lexical/BM25 baseline substantially improved reader-backed evidence and joint metrics over the raw context order, indicating that the dataset adapter and reader evaluation pipeline transfer correctly. However, when evaluated against this strong BM25 baseline, the HotpotQA-trained selector and the 2Wiki cross-fitted selector did not establish reliable selector-level generalization. A BM25-anchor repair reduced negative transfer and nearly matched BM25, but the gain was too small to justify a full 1000-sample validation. Oracle diagnostics show that positive actions beyond BM25 exist, but the current no-leak features and safety predictor do not identify them reliably. We therefore report 2Wiki as a diagnostic limitation rather than as a main generalization claim.
"""
    (AUDIT / "reports/2wiki_paper_limitation_section_draft.md").write_text(limitation)

    appendix = f"""# Appendix: Cross-Dataset Diagnostic on 2WikiMultiHopQA

## BM25 lexical smoke result

The 2Wiki adapter and reader-backed evaluation pipeline were validated on dev-300. BM25 / lexical routing is a strong baseline and substantially improves over raw context order.

## Selector alignment failure

Direct selector transfer and the original 2Wiki crossfit selector underperform the strong BM25 baseline. This should not be written as selector-level generalization success.

## BM25-anchor repair result

The BM25-anchor repair preserves BM25 top-1/top-2/top-3 anchors and reduces negative transfer. The best no-leak repair (`{collect['best_no_leak_selector']}`) obtains joint delta {collect['best_no_leak_selector_delta_vs_bm25']['joint_f1_delta_vs_bm25']:.4f} versus BM25, which is too small to justify 1000-sample expansion.

## Oracle gap

Oracle positive actions exist for {collect['num_positive_vs_bm25_queries']} / {collect['num_queries']} queries, with oracle joint delta {collect['oracle_best_joint_delta_vs_bm25']:.4f}. This is diagnostic only.

## Failure analysis

The dominant failure mode is candidate-pool limitation: {pool['queries_without_positive_vs_bm25']} / {pool['num_queries']} queries have no oracle positive action beyond BM25, and only {pool['strict_action_label_queries_with_positive']} / {pool['num_queries']} expose strict positive actions in the BM25-anchor table. Selector recall over strict available positives is {recall['positive_recall']:.4f}.

## Feature detectability

{feature['interpretation']}. Safety calibration is weak, with answer-safe AUC {safety['answer_safe_auc']:.4f} and paper-positive AUC {safety['paper_positive_auc']:.4f}.

## Claim boundary

2Wiki is reported as pipeline validation and limitation analysis. It is not used as a main method success claim, and oracle rows are not inference-time evidence.
"""
    (AUDIT / "reports/2wiki_appendix_diagnostic.md").write_text(appendix)


def run_all():
    ensure_dirs()
    collect_existing_results()
    analyze_feature_margin()
    analyze_candidate_pool()
    analyze_selector_recall()
    analyze_safety()
    export_cases()
    export_tables()
    write_reports()


if __name__ == "__main__":
    run_all()
