"""Outcome-free matched-pair contrastive learning for geopolitical exposure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .directional_centroid import l2_normalize


CHARACTERISTICS = ["beta_60m_paper", "log_market_equity", "log_book_to_market",
                   "at_gr1_paper", "gp_at_paper", "momentum_12_1"]
DIMENSIONS = (2, 3, 4, 5, 10)


def build_matched_pairs(labels: pd.DataFrame, characteristics: pd.DataFrame,
                        industry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = labels[labels["class_label"].ne(0)].merge(
        characteristics[["permco", *CHARACTERISTICS]], left_on="firm_id", right_on="permco",
        how="inner", validate="one_to_one"
    ).merge(industry[["permco", "sic2"]], on="permco", how="inner", validate="one_to_one")
    values = sample[CHARACTERISTICS].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    means, scales = values.mean(), values.std(ddof=0).replace(0, 1)
    sample[CHARACTERISTICS] = (values - means) / scales
    positive = sample[sample["class_label"].eq(1)].sort_values("firm_id").reset_index(drop=True)
    negative = sample[sample["class_label"].eq(-1)].sort_values("firm_id").reset_index(drop=True)
    char_distance = np.linalg.norm(
        positive[CHARACTERISTICS].to_numpy(float)[:, None, :] -
        negative[CHARACTERISTICS].to_numpy(float)[None, :, :], axis=2
    ) / np.sqrt(len(CHARACTERISTICS))
    pos_sic = positive["sic2"].to_numpy(int)[:, None]
    neg_sic = negative["sic2"].to_numpy(int)[None, :]
    same_sic2 = pos_sic == neg_sic
    same_division = (pos_sic // 10) == (neg_sic // 10)
    # Lexicographically prioritize exact SIC2, then broad division, then characteristics.
    cost = char_distance + np.where(same_sic2, 0, np.where(same_division, 4, 6))
    pos_index, neg_index = linear_sum_assignment(cost)
    rows = []
    for pair_id, (p, n) in enumerate(zip(pos_index, neg_index), 1):
        row = {
            "pair_id": pair_id,
            "positive_firm_id": int(positive.loc[p, "firm_id"]),
            "negative_firm_id": int(negative.loc[n, "firm_id"]),
            "positive_sic2": int(positive.loc[p, "sic2"]),
            "negative_sic2": int(negative.loc[n, "sic2"]),
            "same_sic2": bool(same_sic2[p, n]),
            "same_sic_division": bool(same_division[p, n]),
            "characteristic_distance": float(char_distance[p, n]),
        }
        for column in CHARACTERISTICS:
            row[f"positive_{column}"] = float(positive.loc[p, column])
            row[f"negative_{column}"] = float(negative.loc[n, column])
        rows.append(row)
    pairs = pd.DataFrame(rows)
    balance_rows = []
    for column in CHARACTERISTICS:
        before = (positive[column].mean() - negative[column].mean())
        after = (pairs[f"positive_{column}"] - pairs[f"negative_{column}"]).mean()
        balance_rows.append({"characteristic": column, "standardized_difference_before": float(before),
                             "standardized_difference_after": float(after),
                             "absolute_before": float(abs(before)), "absolute_after": float(abs(after))})
    return pairs, pd.DataFrame(balance_rows)


def pair_differences(frame: pd.DataFrame, pairs: pd.DataFrame, dimension: int) -> np.ndarray:
    columns = [f"embedding_{i:03d}" for i in range(1, dimension + 1)]
    x = l2_normalize(frame[columns].to_numpy(float))
    lookup = pd.Series(np.arange(len(frame)), index=frame["permco"].astype(int))
    p = lookup.loc[pairs["positive_firm_id"].astype(int)].to_numpy()
    n = lookup.loc[pairs["negative_firm_id"].astype(int)].to_numpy()
    return x[p] - x[n]


def fit_ridge_direction(differences: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    d = np.ascontiguousarray(differences, dtype=np.float64)
    # Explicit contractions avoid a spurious Accelerate/BLAS matmul overflow
    # observed for these small 10-column arrays on some macOS NumPy builds.
    gram = np.einsum("ni,nj->ij", d, d, optimize=False) + penalty * np.eye(d.shape[1])
    target = d.sum(axis=0)
    direction = np.linalg.solve(gram, target)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else np.zeros(d.shape[1])


def loo_pair_scores(differences: np.ndarray, penalty: float = 1.0) -> pd.DataFrame:
    d = np.asarray(differences, dtype=float)
    rows = []
    for i in range(len(d)):
        train = np.arange(len(d)) != i
        direction = fit_ridge_direction(d[train], penalty)
        score = float(d[i] @ direction)
        centroid = d[train].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        rows.append({"pair_index": i, "contrastive_score": score,
                     "centroid_difference_score": float(d[i] @ centroid),
                     "pair_correct": score > 0})
    return pd.DataFrame(rows)


def pair_metrics(scores: pd.DataFrame) -> dict:
    correct = int(scores["pair_correct"].sum())
    total = len(scores)
    return {
        "pairs": total, "correct_pairs": correct, "pair_accuracy": correct / total,
        "exact_binomial_p": float(binomtest(correct, total, 0.5, alternative="greater").pvalue),
        "mean_contrastive_score": float(scores["contrastive_score"].mean()),
        "median_contrastive_score": float(scores["contrastive_score"].median()),
    }


def evaluate_representation(frame: pd.DataFrame, pairs: pd.DataFrame, representation: str,
                            penalty: float = 1.0, dimensions: tuple[int, ...] = DIMENSIONS
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    for dimension in dimensions:
        differences = pair_differences(frame, pairs, dimension)
        scored = loo_pair_scores(differences, penalty)
        scored["pair_id"] = pairs["pair_id"].to_numpy()
        scored["representation"] = representation
        scored["dimension"] = dimension
        scored["penalty"] = penalty
        metrics.append({"representation": representation, "dimension": dimension,
                        "penalty": penalty, **pair_metrics(scored)})
        predictions.append(scored)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def sign_flip_permutation(frame: pd.DataFrame, pairs: pd.DataFrame, representation: str,
                          dimension: int = 10, penalty: float = 1.0,
                          repetitions: int = 5000, seed: int = 20220224
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    differences = pair_differences(frame, pairs, dimension)
    observed_scores = loo_pair_scores(differences, penalty)
    observed = pair_metrics(observed_scores)
    rng = np.random.default_rng(seed)
    null_rows = []
    for iteration in range(repetitions):
        signs = rng.choice([-1.0, 1.0], size=len(differences))
        permuted = differences * signs[:, None]
        scores = loo_pair_scores(permuted, penalty)
        null_rows.append({"iteration": iteration + 1,
                          "pair_accuracy": float(scores["pair_correct"].mean()),
                          "mean_contrastive_score": float(scores["contrastive_score"].mean())})
    null = pd.DataFrame(null_rows)
    summary = pd.DataFrame([{
        "representation": representation, "dimension": dimension, "penalty": penalty,
        "repetitions": repetitions, **{f"observed_{k}": v for k, v in observed.items()},
        "accuracy_permutation_p": float((1 + (null["pair_accuracy"] >= observed["pair_accuracy"]).sum()) / (repetitions + 1)),
        "mean_score_permutation_p": float((1 + (null["mean_contrastive_score"] >= observed["mean_contrastive_score"]).sum()) / (repetitions + 1)),
    }])
    null["representation"] = representation
    return summary, null


def difference_alignment(frame: pd.DataFrame, pairs: pd.DataFrame, dimension: int = 10) -> pd.DataFrame:
    d = l2_normalize(pair_differences(frame, pairs, dimension))
    cosine = d @ d.T
    upper = cosine[np.triu_indices(len(d), 1)]
    return pd.DataFrame([{"pairs": len(d), "pairwise_comparisons": len(upper),
                          "mean_difference_cosine": float(upper.mean()),
                          "median_difference_cosine": float(np.median(upper)),
                          "positive_alignment_rate": float((upper > 0).mean())}])


def write_design(path: Path, pairs: pd.DataFrame) -> dict:
    pair_payload = pairs.sort_values("pair_id").to_dict("records")
    design = {
        "stage": "outcome_free_matched_pairwise_contrastive", "embedding_date": "2021Q4",
        "primary_representation": "all_s12_rs_binary", "primary_dimension": 10,
        "primary_penalty": 1.0, "dimensions": list(DIMENSIONS),
        "matching": "one-to-one Hungarian; exact SIC2, then SIC division, then six characteristics",
        "validation": "leave one complete matched pair out", "permutations": 5000,
        "permutation": "independent within-pair sign flips", "event_returns_loaded": False,
        "pair_sha256": hashlib.sha256(json.dumps(pair_payload, sort_keys=True).encode()).hexdigest(),
    }
    design["sha256"] = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(design, indent=2))
    return design
