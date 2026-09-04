#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "selector_v2_2"
V23 = ROOT / "selector_v2_3"
FEATURES = [
    "safe_answer_prob",
    "support_proxy_delta",
    "support_proxy_delta_vs_replaced_doc",
    "support_proxy_delta_vs_baseline_tail_mean",
    "answer_risk_score",
    "displacement_score",
    "hybrid_score_delta",
    "agent_weight_delta",
    "title_bridge_score",
    "prefix1_preserved",
    "prefix2_preserved",
    "prefix3_preserved",
    "num_added_docs",
    "num_removed_docs",
    "removed_baseline_top1",
    "removed_baseline_top2",
]
FAMILIES = {
    "insert1_plus_bridge": {"insert1", "bridge"},
    "all_effective_conservative": {"insert1", "bridge", "top4_bg1"},
    "all_effective": None,
}


def ensure_dirs():
    for rel in [
        "outputs/labels",
        "outputs/model_cv",
        "outputs/calibration",
        "outputs/final_1000",
        "outputs/ablation",
        "outputs/diagnostics",
        "reports",
    ]:
        (V23 / rel).mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def iter_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows():
    p = V22 / "outputs/action_table/effective_action_table.jsonl"
    return list(iter_jsonl(p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def fnum(row, key, default=0.0):
    val = row.get(key, default)
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        if val is None or math.isnan(float(val)):
            return default
        return float(val)
    except Exception:
        return default


def get_features(row, drop_support=False, drop_safety=False):
    vals = []
    for k in FEATURES:
        if drop_support and k in {"support_proxy_delta", "support_proxy_delta_vs_replaced_doc", "support_proxy_delta_vs_baseline_tail_mean", "agent_weight_delta"}:
            continue
        if drop_safety and k == "safe_answer_prob":
            continue
        vals.append(fnum(row, k))
    fam = row.get("candidate_family", "")
    name = row.get("candidate_name", "")
    for v in ["insert1", "bridge", "top4_bg1", "insert2"]:
        vals.append(1.0 if fam == v else 0.0)
    for v in ["keep_top3", "keep_top2", "slot4", "slot3", "balanced", "strict"]:
        vals.append(1.0 if v in name else 0.0)
    return vals


def add_labels(row):
    out = dict(row)
    answer = fnum(row, "answer_f1_delta")
    joint = fnum(row, "joint_f1_delta")
    support = fnum(row, "support_recall_delta")
    sp = fnum(row, "sp_f1_delta")
    out["answer_safe"] = int(answer >= 0)
    out["joint_positive"] = int(joint > 0)
    out["answer_safe_joint_positive"] = int(answer >= 0 and joint > 0)
    out["support_positive"] = int(support > 0 or sp > 0)
    out["paper_positive"] = int(answer >= 0 and joint > 0 and (support > 0 or sp >= 0))
    out["answer_drop"] = int(answer < 0)
    out["large_answer_drop"] = int(answer < -0.05)
    out["joint_only_positive"] = int(joint > 0 and answer < 0)
    out["support_gain_no_reader_gain"] = int(support > 0 and joint <= 0)
    target = joint + 0.8 * answer + 0.3 * support + 0.2 * sp
    out["listwise_target_score"] = target
    return out


def build_labels():
    ensure_dirs()
    rows = [add_labels(r) for r in load_rows()]
    write_jsonl(V23 / "outputs/labels/action_labels.jsonl", rows)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)
    pos_counts = [sum(x["paper_positive"] for x in xs) for xs in by_q.values()]
    dist = Counter(pos_counts)
    n = len(rows)
    summary = {
        "num_actions": n,
        "num_queries": len(by_q),
        "answer_safe_rate": sum(r["answer_safe"] for r in rows) / n,
        "joint_positive_rate": sum(r["joint_positive"] for r in rows) / n,
        "answer_safe_joint_positive_rate": sum(r["answer_safe_joint_positive"] for r in rows) / n,
        "paper_positive_rate": sum(r["paper_positive"] for r in rows) / n,
        "answer_drop_rate": sum(r["answer_drop"] for r in rows) / n,
        "large_answer_drop_rate": sum(r["large_answer_drop"] for r in rows) / n,
        "positive_actions_per_query_distribution": dict(sorted((str(k), v) for k, v in dist.items())),
        "queries_with_no_positive_action": sum(1 for c in pos_counts if c == 0),
    }
    write_json(V23 / "outputs/labels/label_summary.json", summary)
    return rows, summary


def load_labeled_rows():
    p = V23 / "outputs/labels/action_labels.jsonl"
    if not p.exists():
        return build_labels()[0]
    return list(iter_jsonl(p))


def split_queries(rows, k=5):
    queries = sorted({r["query_id"] for r in rows})
    def key(q):
        return int(hashlib.md5(q.encode()).hexdigest(), 16)
    queries.sort(key=key)
    folds = []
    for i in range(k):
        test = set(queries[i::k])
        train = set(queries) - test
        folds.append((train, test))
    return folds


class LinearModel:
    def __init__(self, model_type, weights, mean, scale, bias=0.0, drop_support=False, drop_safety=False):
        self.model_type = model_type
        self.weights = weights
        self.mean = mean
        self.scale = scale
        self.bias = bias
        self.drop_support = drop_support
        self.drop_safety = drop_safety

    def raw(self, row):
        x = get_features(row, self.drop_support, self.drop_safety)
        s = self.bias
        for i, w in enumerate(self.weights):
            z = (x[i] - self.mean[i]) / self.scale[i] if self.scale[i] else 0.0
            s += w * z
        return s

    def prob(self, row):
        if self.model_type in {"constrained_regression_answer", "constrained_regression_joint", "constrained_regression_support", "listwise_ranker"}:
            return self.raw(row)
        return sigmoid(self.raw(row))


def fit_classifier(rows, target, model_type, drop_support=False, drop_safety=False):
    xs = [get_features(r, drop_support, drop_safety) for r in rows]
    ys = [fnum(r, target) for r in rows]
    m = len(xs[0]) if xs else 1
    mean = [statistics.fmean(x[i] for x in xs) if xs else 0.0 for i in range(m)]
    scale = []
    for i in range(m):
        vals = [x[i] for x in xs]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
        scale.append(sd or 1.0)
    pos = [x for x, y in zip(xs, ys) if y >= 0.5]
    neg = [x for x, y in zip(xs, ys) if y < 0.5]
    weights = []
    for i in range(m):
        p = statistics.fmean((x[i] - mean[i]) / scale[i] for x in pos) if pos else 0.0
        n = statistics.fmean((x[i] - mean[i]) / scale[i] for x in neg) if neg else 0.0
        weights.append(p - n)
    prior = (sum(ys) + 1) / (len(ys) + 2)
    bias = math.log(prior / (1 - prior))
    return LinearModel(model_type, weights, mean, scale, bias, drop_support, drop_safety)


def fit_regressor(rows, target, model_type, drop_support=False, drop_safety=False):
    xs = [get_features(r, drop_support, drop_safety) for r in rows]
    ys = [fnum(r, target) for r in rows]
    m = len(xs[0]) if xs else 1
    mean = [statistics.fmean(x[i] for x in xs) if xs else 0.0 for i in range(m)]
    scale = []
    weights = []
    ymean = statistics.fmean(ys) if ys else 0.0
    ysd = statistics.pstdev(ys) if len(ys) > 1 else 1.0
    ysd = ysd or 1.0
    for i in range(m):
        vals = [x[i] for x in xs]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
        sd = sd or 1.0
        scale.append(sd)
        corr = statistics.fmean(((x[i] - mean[i]) / sd) * ((y - ymean) / ysd) for x, y in zip(xs, ys)) if xs else 0.0
        weights.append(corr * ysd / max(1.0, math.sqrt(m)))
    return LinearModel(model_type, weights, mean, scale, ymean, drop_support, drop_safety)


def train_models(train_rows, drop_support=False, drop_safety=False):
    return {
        "answer_safe": fit_classifier(train_rows, "answer_safe", "answer_safe", drop_support, drop_safety),
        "answer_safe_joint_positive": fit_classifier(train_rows, "answer_safe_joint_positive", "answer_safe_joint_positive", drop_support, drop_safety),
        "paper_positive": fit_classifier(train_rows, "paper_positive", "paper_positive", drop_support, drop_safety),
        "answer_drop": fit_classifier(train_rows, "answer_drop", "answer_drop", drop_support, drop_safety),
        "listwise": fit_regressor(train_rows, "listwise_target_score", "listwise_ranker", drop_support, drop_safety),
        "pred_answer_delta": fit_regressor(train_rows, "answer_f1_delta", "constrained_regression_answer", drop_support, drop_safety),
        "pred_joint_delta": fit_regressor(train_rows, "joint_f1_delta", "constrained_regression_joint", drop_support, drop_safety),
        "pred_support_delta": fit_regressor(train_rows, "support_recall_delta", "constrained_regression_support", drop_support, drop_safety),
    }


def score_row(row, models, model_type, answer_margin=0.0, answer_drop_lambda=0.4):
    p_safe = models["answer_safe"].prob(row)
    p_joint = models["answer_safe_joint_positive"].prob(row)
    p_paper = models["paper_positive"].prob(row)
    p_drop = models["answer_drop"].prob(row)
    pred_answer = models["pred_answer_delta"].prob(row)
    pred_joint = models["pred_joint_delta"].prob(row)
    pred_support = models["pred_support_delta"].prob(row)
    listwise = models["listwise"].prob(row)
    support_first = fnum(row, "support_proxy_delta") + 0.3 * fnum(row, "title_bridge_score") + 0.2 * fnum(row, "agent_weight_delta")
    if model_type == "two_stage":
        score = p_safe * p_joint
        allowed = p_safe >= 0.0
    elif model_type == "paper_positive_classifier":
        score = p_paper
        allowed = True
    elif model_type == "answer_drop_rejector_support_ranker":
        score = support_first - answer_drop_lambda * p_drop
        allowed = True
    elif model_type == "pairwise_ranker":
        score = listwise
        allowed = True
    elif model_type == "constrained_regression":
        score = pred_joint + 0.3 * pred_support
        allowed = pred_answer >= answer_margin
    else:
        score = support_first
        allowed = True
    out = dict(row)
    out.update({
        "pred_answer_safe_prob": p_safe,
        "pred_positive_prob": p_paper if model_type == "paper_positive_classifier" else p_joint,
        "pred_answer_drop_prob": p_drop,
        "pred_answer_delta": pred_answer,
        "pred_joint_delta": pred_joint,
        "pred_support_delta": pred_support,
        "utility_score": score,
        "model_allowed": allowed,
    })
    return out


def baseline_from_row(row):
    return {
        "answer_access_at_k": fnum(row, "baseline_answer_access_at_k"),
        "support_recall_at_k": fnum(row, "baseline_support_recall"),
        "sp_f1": fnum(row, "baseline_sp_f1"),
        "answer_em": fnum(row, "baseline_answer_em"),
        "answer_f1": fnum(row, "baseline_answer_f1"),
        "joint_f1": fnum(row, "baseline_joint_f1"),
    }


def candidate_from_row(row):
    return {
        "answer_access_at_k": fnum(row, "candidate_answer_access_at_k"),
        "support_recall_at_k": fnum(row, "candidate_support_recall"),
        "sp_f1": fnum(row, "candidate_sp_f1"),
        "answer_em": fnum(row, "candidate_answer_em"),
        "answer_f1": fnum(row, "candidate_answer_f1"),
        "joint_f1": fnum(row, "candidate_joint_f1"),
    }


def summarize_selection(selected, by_q):
    metrics = ["answer_access_at_k", "support_recall_at_k", "sp_f1", "answer_em", "answer_f1", "joint_f1"]
    totals = Counter()
    base_totals = Counter()
    per = []
    selected_count = 0
    eff_count = 0
    dist = Counter()
    answer_drop = 0
    joint_pos = 0
    paper_pos = 0
    q_with_pos = 0
    q_with_answer_safe_pos = 0
    selected_pos_queries = 0
    selected_answer_safe_pos_queries = 0
    for q, rows in by_q.items():
        base = baseline_from_row(rows[0])
        row = selected.get(q)
        if row is None:
            cur = base
            dist["baseline_fallback"] += 1
            fallback = True
        else:
            cur = candidate_from_row(row)
            selected_count += 1
            eff_count += int(bool(row.get("effective_context_changed", True)))
            dist[row.get("candidate_name", "candidate")] += 1
            answer_drop += int(row.get("answer_drop", 0))
            joint_pos += int(row.get("joint_positive", 0))
            paper_pos += int(row.get("paper_positive", 0))
            fallback = False
        has_pos = any(r.get("paper_positive", 0) for r in rows)
        has_answer_safe_pos = any(r.get("answer_safe_joint_positive", 0) for r in rows)
        q_with_pos += int(has_pos)
        q_with_answer_safe_pos += int(has_answer_safe_pos)
        if row is not None and row.get("paper_positive", 0):
            selected_pos_queries += 1
        if row is not None and row.get("answer_safe_joint_positive", 0):
            selected_answer_safe_pos_queries += 1
        rec = {
            "query_id": q,
            "question": rows[0].get("question", ""),
            "fallback": fallback,
            "candidate_name": row.get("candidate_name") if row else "baseline_fallback",
            "candidate_family": row.get("candidate_family") if row else "baseline",
            "selected": row is not None,
        }
        for m in metrics:
            totals[m] += cur[m]
            base_totals[m] += base[m]
            rec[m] = cur[m]
            rec["baseline_" + m] = base[m]
            rec[m + "_delta"] = cur[m] - base[m]
        if row:
            for k in ["baseline_titles", "candidate_titles", "added_titles", "removed_titles", "safe_answer_prob", "pred_answer_safe_prob", "pred_positive_prob", "pred_answer_drop_prob", "support_proxy_delta", "answer_risk_score", "utility_score", "paper_positive", "answer_drop", "joint_positive"]:
                rec[k] = row.get(k)
        per.append(rec)
    n = len(by_q)
    summary = {m: totals[m] / n for m in metrics}
    summary["baseline"] = {m: base_totals[m] / n for m in metrics}
    summary.update({
        "n": n,
        "answer_f1_delta": summary["answer_f1"] - summary["baseline"]["answer_f1"],
        "joint_f1_delta": summary["joint_f1"] - summary["baseline"]["joint_f1"],
        "support_recall_delta": summary["support_recall_at_k"] - summary["baseline"]["support_recall_at_k"],
        "sp_f1_delta": summary["sp_f1"] - summary["baseline"]["sp_f1"],
        "fallback_rate": 1 - selected_count / n,
        "selected_count": selected_count,
        "effective_selected_count": eff_count,
        "selected_effective_action_rate": (eff_count / selected_count) if selected_count else 1.0,
        "selected_candidate_distribution": dict(dist),
        "positive_candidate_recall": (selected_pos_queries / q_with_pos) if q_with_pos else 0.0,
        "answer_safe_positive_candidate_recall": (selected_answer_safe_pos_queries / q_with_answer_safe_pos) if q_with_answer_safe_pos else 0.0,
        "selected_answer_drop_count": answer_drop,
        "selected_answer_drop_rate": (answer_drop / selected_count) if selected_count else 0.0,
        "selected_joint_positive_count": joint_pos,
        "selected_joint_positive_rate": (joint_pos / selected_count) if selected_count else 0.0,
        "selected_paper_positive_count": paper_pos,
    })
    summary["gate_pass"] = (
        summary["answer_f1_delta"] >= 0
        and summary["joint_f1_delta"] > 0
        and summary["support_recall_delta"] > 0
        and summary["sp_f1_delta"] >= 0
        and abs(summary["selected_effective_action_rate"] - 1.0) < 1e-9
        and summary["fallback_rate"] <= 0.80
        and summary["positive_candidate_recall"] > 0.1839
    )
    summary["paper_main_recommended"] = (
        summary["answer_f1_delta"] >= 0
        and summary["joint_f1_delta"] >= 0.0081
        and summary["support_recall_delta"] >= 0.0075
        and summary["sp_f1_delta"] >= 0.0103
        and summary["positive_candidate_recall"] >= 0.30
    )
    return summary, per


def select_actions(rows, models, config):
    by_q = defaultdict(list)
    fam_allowed = FAMILIES.get(config.get("candidate_family", "all_effective"))
    for r in rows:
        if fam_allowed is not None and r.get("candidate_family") not in fam_allowed:
            continue
        sr = score_row(r, models, config["model_type"], config.get("answer_margin", 0.0), config.get("answer_drop_lambda", 0.4))
        if not sr["model_allowed"]:
            continue
        if sr["pred_answer_safe_prob"] < config.get("answer_safe_threshold", 0.0):
            continue
        if sr["pred_positive_prob"] < config.get("positive_threshold", -1.0):
            continue
        if sr["pred_answer_drop_prob"] > config.get("answer_drop_threshold", 1.0):
            continue
        by_q[sr["query_id"]].append(sr)
    candidates = []
    for q, xs in by_q.items():
        xs.sort(key=lambda x: (x["utility_score"], x.get("safe_answer_prob", 0.0), x.get("support_proxy_delta", 0.0)), reverse=True)
        candidates.append(xs[0])
    candidates.sort(key=lambda x: (x["utility_score"], x["pred_answer_safe_prob"], -x["pred_answer_drop_prob"]), reverse=True)
    budget = int(round(config["selected_fraction"] * len({r["query_id"] for r in rows})))
    return {r["query_id"]: r for r in candidates[:budget]}


def config_grid(model_types=None, compact=False):
    model_types = model_types or ["two_stage", "paper_positive_classifier", "answer_drop_rejector_support_ranker", "pairwise_ranker", "constrained_regression"]
    fracs = [0.30, 0.40, 0.50, 0.60] if not compact else [0.40, 0.50]
    safe_th = [0.50, 0.55, 0.60, 0.65, 0.70] if not compact else [0.50]
    pos_th = [0.10, 0.15, 0.20, 0.25, 0.30] if not compact else [0.10]
    drop_th = [0.20, 0.30, 0.40, 0.50] if not compact else [0.30]
    margins = [-0.005, 0.0, 0.002, 0.005] if not compact else [0.0]
    families = ["insert1_plus_bridge", "all_effective_conservative", "all_effective"] if not compact else ["insert1_plus_bridge", "all_effective_conservative"]
    for mt in model_types:
        for frac in fracs:
            for fam in families:
                if mt == "constrained_regression":
                    for margin in margins:
                        yield {"model_type": mt, "selected_fraction": frac, "answer_margin": margin, "candidate_family": fam, "answer_safe_threshold": 0.0, "positive_threshold": -1.0, "answer_drop_threshold": 1.0}
                elif mt == "answer_drop_rejector_support_ranker":
                    for dt in drop_th:
                        yield {"model_type": mt, "selected_fraction": frac, "answer_drop_threshold": dt, "candidate_family": fam, "answer_safe_threshold": 0.0, "positive_threshold": -1.0, "answer_margin": 0.0}
                else:
                    for st in safe_th:
                        for pt in pos_th:
                            yield {"model_type": mt, "selected_fraction": frac, "answer_safe_threshold": st, "positive_threshold": pt, "answer_drop_threshold": 1.0, "candidate_family": fam, "answer_margin": 0.0}


def choose_config(train_rows, models, model_types=None, compact=False):
    by_q = group_by_query(train_rows)
    best = None
    best_tuple = None
    records = []
    for cfg in config_grid(model_types, compact):
        selected = select_actions(train_rows, models, cfg)
        summary, _ = summarize_selection(selected, by_q)
        cfg_sum = dict(cfg)
        cfg_sum.update({k: summary[k] for k in ["answer_f1_delta", "joint_f1_delta", "support_recall_delta", "sp_f1_delta", "fallback_rate", "selected_effective_action_rate", "positive_candidate_recall", "selected_answer_drop_rate", "selected_joint_positive_rate", "gate_pass"]})
        records.append(cfg_sum)
        hard = (
            summary["answer_f1_delta"] >= 0,
            summary["support_recall_delta"] > 0,
            summary["sp_f1_delta"] >= 0,
            0.40 <= summary["fallback_rate"] <= 0.70,
            summary["selected_effective_action_rate"] == 1.0,
            summary["selected_answer_drop_rate"] <= 0.05,
            summary["positive_candidate_recall"] >= 0.30,
        )
        score = (
            int(all(hard)),
            int(summary["answer_f1_delta"] >= 0),
            min(summary["positive_candidate_recall"], 0.45),
            summary["joint_f1_delta"],
            summary["support_recall_delta"],
            -abs(summary["fallback_rate"] - 0.5),
            -summary["selected_answer_drop_rate"],
        )
        if best_tuple is None or score > best_tuple:
            best_tuple = score
            best = cfg_sum
    records.sort(key=lambda r: (r["answer_f1_delta"] >= 0, r["positive_candidate_recall"], r["joint_f1_delta"]), reverse=True)
    return best, records[:20]


def group_by_query(rows):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)
    return by_q


def run_crossfit(model_types=None, compact=False, ablation_name=None, drop_support=False, drop_safety=False):
    rows = load_labeled_rows()
    folds = split_queries(rows, 5)
    all_selected = {}
    fold_configs = []
    model_summaries = []
    for fold_id, (train_q, test_q) in enumerate(folds):
        train_rows = [r for r in rows if r["query_id"] in train_q]
        test_rows = [r for r in rows if r["query_id"] in test_q]
        models = train_models(train_rows, drop_support=drop_support, drop_safety=drop_safety)
        best, top = choose_config(train_rows, models, model_types, compact)
        cfg = {k: best[k] for k in ["model_type", "selected_fraction", "answer_safe_threshold", "positive_threshold", "answer_drop_threshold", "candidate_family", "answer_margin"]}
        selected = select_actions(test_rows, models, cfg)
        by_test = group_by_query(test_rows)
        heldout_summary, _ = summarize_selection(selected, by_test)
        all_selected.update(selected)
        fold_configs.append({"fold_id": fold_id, "n_train": len(train_q), "n_test": len(test_q), "config": cfg, "train_best": best, "heldout_summary": heldout_summary, "top_train_configs": top[:5]})
        model_summaries.append({"fold_id": fold_id, "config": cfg, "heldout": heldout_summary})
    by_all = group_by_query(rows)
    summary, per = summarize_selection(all_selected, by_all)
    summary["fold_config_distribution"] = dict(Counter(json.dumps(f["config"], sort_keys=True) for f in fold_configs))
    if ablation_name:
        summary["ablation_name"] = ablation_name
    return summary, per, fold_configs, model_summaries


def bootstrap(per, samples=2000, seed=13):
    rng = random.Random(seed)
    metrics = {
        "answer_f1": "answer_f1_delta",
        "joint_f1": "joint_f1_delta",
        "support_recall@5": "support_recall_at_k_delta",
        "sp_f1": "sp_f1_delta",
    }
    out = {"n": len(per), "num_bootstrap_samples": samples, "metrics": {}}
    n = len(per)
    for name, key in metrics.items():
        vals = [fnum(r, key) for r in per]
        mean = statistics.fmean(vals)
        boots = []
        for _ in range(samples):
            boots.append(statistics.fmean(vals[rng.randrange(n)] for __ in range(n)))
        boots.sort()
        lo = boots[int(0.025 * samples)]
        hi = boots[int(0.975 * samples)]
        if mean >= 0:
            p = sum(1 for b in boots if b <= 0) / samples
        else:
            p = sum(1 for b in boots if b >= 0) / samples
        out["metrics"][name] = {"mean_delta": mean, "ci95": [lo, hi], "p_value": p}
    return out


def write_final_outputs():
    ensure_dirs()
    summary, per, folds, model_summaries = run_crossfit(compact=True)
    write_json(V23 / "outputs/final_1000/final_1000_crossfit_summary.json", summary)
    write_jsonl(V23 / "outputs/final_1000/per_example_delta.jsonl", per)
    write_json(V23 / "outputs/final_1000/fold_configs.json", folds)
    write_json(V23 / "outputs/final_1000/significance_report.json", bootstrap(per))
    write_json(V23 / "outputs/model_cv/model_cv_summary.json", {"folds": model_summaries})
    write_json(V23 / "outputs/calibration/calibration_summary.json", {"fold_configs": folds, "final_summary": summary})
    return summary


def run_ablations():
    ensure_dirs()
    results = {}
    ablations = {
        "ablation_two_stage": {"model_types": ["two_stage"], "compact": True},
        "ablation_paper_positive_classifier": {"model_types": ["paper_positive_classifier"], "compact": True},
        "ablation_answer_drop_rejector_support_ranker": {"model_types": ["answer_drop_rejector_support_ranker"], "compact": True},
        "ablation_constrained_regression": {"model_types": ["constrained_regression"], "compact": True},
        "ablation_no_answer_constraint": {"model_types": ["pairwise_ranker"], "compact": True},
        "ablation_no_support_features": {"model_types": ["paper_positive_classifier", "two_stage"], "drop_support": True, "compact": True},
        "ablation_no_safety_predictor": {"model_types": ["paper_positive_classifier", "two_stage"], "drop_safety": True, "compact": True},
        "ablation_all_effective": {"model_types": ["paper_positive_classifier", "two_stage"], "compact": True},
        "ablation_insert1_plus_bridge": {"model_types": ["paper_positive_classifier", "two_stage"], "compact": True},
    }
    for name, opts in ablations.items():
        summary, _, _, _ = run_crossfit(
            model_types=opts.get("model_types"),
            compact=opts.get("compact", False),
            ablation_name=name,
            drop_support=opts.get("drop_support", False),
            drop_safety=opts.get("drop_safety", False),
        )
        if name == "ablation_insert1_plus_bridge":
            summary["note"] = "Grid includes insert1_plus_bridge and reports selected best under this family-sensitive run."
        results[name] = summary
    v22 = read_json(V22 / "outputs/final_1000/final_1000_crossfit_summary.json")
    results["ablation_v2_2_support_first"] = v22
    write_json(V23 / "outputs/ablation/ablation_summary.json", results)
    return results


def diagnose():
    rows = load_labeled_rows()
    final_p = V23 / "outputs/final_1000/per_example_delta.jsonl"
    per = list(iter_jsonl(final_p))
    by_q = group_by_query(rows)
    cases = []
    labels = Counter()
    for rec in per:
        q = rec["query_id"]
        rows_q = by_q[q]
        selected = rec.get("selected", False)
        positive_exists = any(r.get("paper_positive") for r in rows_q)
        answer_safe_positive_exists = any(r.get("answer_safe_joint_positive") for r in rows_q)
        label = None
        if not positive_exists:
            label = "candidate_pool_no_positive_action"
        elif not answer_safe_positive_exists:
            label = "candidate_pool_no_answer_safe_positive"
        elif not selected:
            label = "positive_action_available_but_not_selected"
        elif rec.get("answer_drop"):
            label = "answer_drop_selected"
        elif rec.get("joint_f1_delta", 0) <= 0 and rec.get("support_recall_at_k_delta", 0) > 0:
            label = "support_positive_but_joint_negative"
        elif rec.get("joint_f1_delta", 0) > 0 and rec.get("answer_f1_delta", 0) < 0:
            label = "joint_positive_but_answer_negative"
        elif selected and not rec.get("paper_positive"):
            label = "wrong_action_selected"
        elif selected and rec.get("paper_positive"):
            label = "selected_positive"
        else:
            label = "baseline_already_optimal"
        labels[label] += 1
        case = {k: rec.get(k) for k in [
            "query_id", "question", "baseline_titles", "candidate_titles", "candidate_name", "added_titles", "removed_titles",
            "answer_f1_delta", "joint_f1_delta", "support_recall_at_k_delta", "sp_f1_delta", "safe_answer_prob",
            "pred_answer_safe_prob", "pred_positive_prob", "pred_answer_drop_prob", "support_proxy_delta", "answer_risk_score",
            "utility_score", "selected", "fallback",
        ]}
        case["selected_titles"] = rec.get("candidate_titles")
        case["positive_action_exists"] = positive_exists
        case["answer_safe_positive_action_exists"] = answer_safe_positive_exists
        case["failure_label"] = label
        cases.append(case)
    write_jsonl(V23 / "outputs/diagnostics/failure_cases.jsonl", cases)
    final = read_json(V23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    summary = {
        "n_cases": len(cases),
        "label_counts": dict(labels),
        "positive_candidate_recall": final.get("positive_candidate_recall"),
        "answer_safe_positive_candidate_recall": final.get("answer_safe_positive_candidate_recall"),
        "selected_answer_drop_count": final.get("selected_answer_drop_count"),
        "selected_answer_drop_rate": final.get("selected_answer_drop_rate"),
        "positive_action_available_but_not_selected_count": labels.get("positive_action_available_but_not_selected", 0),
        "wrong_action_selected_count": labels.get("wrong_action_selected", 0),
        "model_false_positive_count": labels.get("wrong_action_selected", 0) + labels.get("answer_drop_selected", 0),
        "model_false_negative_count": labels.get("positive_action_available_but_not_selected", 0),
        "candidate_pool_no_positive_action_count": labels.get("candidate_pool_no_positive_action", 0),
        "candidate_pool_no_answer_safe_positive_count": labels.get("candidate_pool_no_answer_safe_positive", 0),
    }
    write_json(V23 / "outputs/diagnostics/failure_summary.json", summary)
    return summary


def oracle_gap_recall():
    rows = load_labeled_rows()
    final = read_json(V23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    by_q = group_by_query(rows)
    q_pos = sum(1 for xs in by_q.values() if any(r.get("paper_positive") for r in xs))
    q_safe = sum(1 for xs in by_q.values() if any(r.get("answer_safe_joint_positive") for r in xs))
    out = {
        "queries_with_actions": len(by_q),
        "queries_with_paper_positive": q_pos,
        "queries_with_answer_safe_positive": q_safe,
        "positive_candidate_query_rate": q_pos / len(by_q),
        "answer_safe_positive_query_rate": q_safe / len(by_q),
        "selector_positive_candidate_recall": final.get("positive_candidate_recall"),
        "selector_answer_safe_positive_candidate_recall": final.get("answer_safe_positive_candidate_recall"),
        "v2_2_selector_recall_reference": 0.1839,
        "recall_improved_over_v2_2": final.get("positive_candidate_recall", 0) > 0.1839,
    }
    write_json(V23 / "outputs/diagnostics/oracle_gap_recall_analysis.json", out)
    return out


def make_report():
    final = read_json(V23 / "outputs/final_1000/final_1000_crossfit_summary.json")
    sig = read_json(V23 / "outputs/final_1000/significance_report.json")
    labels = read_json(V23 / "outputs/labels/label_summary.json")
    failure = read_json(V23 / "outputs/diagnostics/failure_summary.json")
    ablation = read_json(V23 / "outputs/ablation/ablation_summary.json") if (V23 / "outputs/ablation/ablation_summary.json").exists() else {}
    lines = []
    lines.append("# V7-HP-PAPER selector_v2.3 answer-neutral positive selector 报告\n")
    lines.append("## 1. 实验目的\n")
    lines.append("v2.3 旨在解决 v2.2 的核心瓶颈：support-side 指标已有正信号，但 answer_f1 微弱为负，且 positive candidate recall 只有 0.1839。v2.3 不重跑 reader，复用 v2.2 effective action table，在 query-level cross-fitting 下训练 answer-neutral positive-action selector。\n")
    lines.append("## 2. 标签分布\n")
    lines.append("```json\n" + json.dumps(labels, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("## 3. Final 1000 结果\n")
    rows = [
        ("answer_f1_delta", final["answer_f1_delta"]),
        ("joint_f1_delta", final["joint_f1_delta"]),
        ("support_recall_delta", final["support_recall_delta"]),
        ("sp_f1_delta", final["sp_f1_delta"]),
        ("fallback_rate", final["fallback_rate"]),
        ("positive_candidate_recall", final["positive_candidate_recall"]),
        ("selected_answer_drop_rate", final["selected_answer_drop_rate"]),
        ("gate_pass", final["gate_pass"]),
        ("paper_main_recommended", final["paper_main_recommended"]),
    ]
    lines.append("\n".join(f"- `{k}` = `{v}`" for k, v in rows) + "\n")
    lines.append("## 4. Significance\n")
    lines.append("```json\n" + json.dumps(sig, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("## 5. Ablation 摘要\n")
    for name, s in ablation.items():
        lines.append(f"- `{name}`: answer `{s.get('answer_f1_delta')}`, joint `{s.get('joint_f1_delta')}`, support `{s.get('support_recall_delta')}`, sp `{s.get('sp_f1_delta')}`, recall `{s.get('positive_candidate_recall')}`")
    lines.append("\n## 6. Failure Diagnosis\n")
    lines.append("```json\n" + json.dumps(failure, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("## 7. 论文判断\n")
    if final["gate_pass"]:
        lines.append("v2.3 通过 strict no-leak gate，可作为论文主结果候选。核心叙事是 federated routing 暴露有用 support candidates，answer-neutral selector 将其转化为 downstream gains。\n")
    else:
        lines.append("v2.3 未通过 strict gate，不能宣称最终成功。当前可写成 diagnostic contribution：routing/candidate pool 暴露了潜力，但 strict no-leak selector 仍难以稳定识别 answer-safe joint-positive action。\n")
    report = "\n".join(lines)
    (V23 / "reports").mkdir(parents=True, exist_ok=True)
    (V23 / "reports/v7_hp_paper_selector_v2_3_report.md").write_text(report)
    if not final["gate_pass"]:
        memo = [
            "# V7-HP-PAPER selector_v2.3 Paper Decision Memo\n",
            "## 当前最强正式 no-leak 结果\n",
            json.dumps(final, ensure_ascii=False, indent=2),
            "\n## 为什么不能声称 final success\n",
            "strict gate 未全部满足，因此不能作为最终成功主张。\n",
            "\n## 可写论文贡献\n",
            "1. policy-action-to-reader gap diagnostic；\n2. strict no-leak selector 的规模化评估；\n3. oracle/candidate potential 与 selector recall 瓶颈定位；\n4. low-cost follow-up 可做 prompt/order sensitivity 与 candidate quality breakdown。\n",
            "\n## 建议\n",
            "停止盲目大规模 reader validation，优先做 reader_prompt_order_sensitivity_diagnostic、candidate_pool_quality_breakdown、positive_candidate_feature_importance 与 failure_case_study_export。\n",
        ]
        (V23 / "reports/paper_decision_memo.md").write_text("\n".join(memo))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["labels", "train", "calibrate", "final", "ablation", "diagnose", "oracle", "report", "all"])
    args = ap.parse_args()
    ensure_dirs()
    if args.command == "labels":
        build_labels()
    elif args.command == "train":
        rows = load_labeled_rows()
        folds = split_queries(rows)
        model_summaries = []
        for i, (train_q, test_q) in enumerate(folds):
            train_rows = [r for r in rows if r["query_id"] in train_q]
            test_rows = [r for r in rows if r["query_id"] in test_q]
            models = train_models(train_rows)
            model_summaries.append({"fold_id": i, "n_train_queries": len(train_q), "n_test_queries": len(test_q), "models": list(models)})
        write_json(V23 / "outputs/model_cv/model_cv_summary.json", {"folds": model_summaries})
    elif args.command == "calibrate":
        summary, per, folds, model_summaries = run_crossfit(compact=True)
        write_json(V23 / "outputs/calibration/calibration_summary.json", {"fold_configs": folds, "preview_summary": summary})
    elif args.command == "final":
        write_final_outputs()
    elif args.command == "ablation":
        run_ablations()
    elif args.command == "diagnose":
        diagnose()
    elif args.command == "oracle":
        oracle_gap_recall()
    elif args.command == "report":
        make_report()
    elif args.command == "all":
        build_labels()
        write_final_outputs()
        run_ablations()
        diagnose()
        oracle_gap_recall()
        make_report()


if __name__ == "__main__":
    main()
