import math


class EvaluationResult_TRT:
    def __init__(self, metrics=None):
        self.results = {
            "n": 0,
            "F1": 0.0,
            "em": 0.0,
            "mrr": 0.0,
            "hit1": 0.0,
            "hit10": 0.0,
            "MAP": 0.0,
            "NDCG": 0.0,
            "DCG": 0.0,
            "IDCG": 0.0,
        }
        self.metrics_results = {}

        if metrics is None:
            metrics = []
        metrics.extend(["n", "F1", "em", "mrr", "hit1", "hit10", "MAP", "NDCG", "DCG", "IDCG"])
        self.metrics = list(dict.fromkeys(metrics))

    def add(self, evaluate_result):
        for key in self.results.keys():
            if key in self.metrics:
                self.results[key] += evaluate_result.results[key]
        self.results["n"] += 1

    def print_results(self):
        for key, value in self.results.items():
            if key not in self.metrics:
                continue
            if key == "n":
                print(f"{key}: {value}")
            else:
                denominator = self.results["n"] or 1
                print(f"{key}: {value / denominator}")

    def print_results_to_path(self, path, config, sample_arr):
        print("save data to " + path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.writelines("\n")
            handle.writelines("=========================== begin ==================\n")
            handle.writelines("database: " + config.dataset + "\n")
            handle.writelines("path: " + sample_arr + "\n")
            for key, value in self.results.items():
                if key not in self.metrics:
                    continue
                if key == "n":
                    handle.writelines(f"{key}: {value}\n")
                else:
                    denominator = self.results["n"] or 1
                    handle.writelines(f"{key}: {value / denominator}\n")


def evaluating_TRT(retrieval_ids, golden_context_ids):
    eval_result = EvaluationResult_TRT()
    eval_result.results["F1"] = f1(retrieval_ids, golden_context_ids)
    eval_result.results["em"] = exact_match(retrieval_ids, golden_context_ids)
    eval_result.results["mrr"] = mrr(retrieval_ids, golden_context_ids)
    eval_result.results["hit1"] = hit(retrieval_ids, golden_context_ids[:1])
    eval_result.results["hit10"] = hit(retrieval_ids, golden_context_ids[:10])
    eval_result.results["MAP"] = mean_average_precision(retrieval_ids, golden_context_ids)
    eval_result.results["DCG"] = dcg(retrieval_ids, golden_context_ids)
    eval_result.results["IDCG"] = idcg(retrieval_ids, golden_context_ids)
    eval_result.results["NDCG"] = ndcg(retrieval_ids, golden_context_ids)
    return eval_result


def mrr(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    for i, item in enumerate(retrieved_ids):
        if item in expected_ids:
            score = 1.0 / (i + 1)
            print(f"mrr: {score}")
            return score
    return 0.0


def hit(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    return 1.0 if any(item in expected_ids for item in retrieved_ids) else 0.0


def f1(retrieved_ids, expected_ids):
    retrieved_ids = retrieved_ids[: len(expected_ids)]
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    true_positive = len(retrieved_set & expected_set)
    false_positive = len(retrieved_set - expected_set)
    false_negative = len(expected_set - retrieved_set)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def exact_match(retrieved_ids, expected_ids):
    return 1.0 if sorted(expected_ids) == sorted(retrieved_ids[: len(expected_ids)]) else 0.0


def mean_average_precision(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    if not retrieved_ids or not expected_ids:
        return 0.0
    score = 0.0
    for rank, item in enumerate(expected_ids, start=1):
        if item in retrieved_ids:
            score += rank / (retrieved_ids.index(item) + 1)
    return score / len(expected_ids)


def dcg(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    score = 0.0
    for i, item in enumerate(retrieved_ids):
        if item in expected_ids:
            score += 1.0 / math.log2(i + 2)
    return score


def idcg(retrieved_ids, expected_ids):
    ideal_hits = [item for item in retrieved_ids if item in expected_ids]
    remaining = [item for item in expected_ids if item not in ideal_hits]
    ideal_ranking = ideal_hits + remaining
    return dcg(ideal_ranking, expected_ids)


def ndcg(retrieved_ids, expected_ids):
    ideal = idcg(retrieved_ids, expected_ids)
    if ideal == 0:
        return 0.0
    return dcg(retrieved_ids, expected_ids) / ideal
