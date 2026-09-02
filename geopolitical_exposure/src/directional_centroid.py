"""Outcome-free tests of beneficiary-versus-adverse directions in asset embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


DIMENSIONS = (2, 3, 4, 5, 10)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms <= 0):
        raise ValueError("Features contain non-finite or zero-norm rows.")
    return values / norms


def build_firm_labels(seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = seeds.copy()
    frame["firm_id"] = pd.to_numeric(frame["firm_id"], errors="raise").astype(int)
    frame["signed_score"] = pd.to_numeric(frame["signed_score"], errors="raise")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="raise")
    rows = []
    for firm_id, group in frame.groupby("firm_id", sort=True):
        positive = bool((group["signed_score"] > 0).any())
        negative = bool((group["signed_score"] < 0).any())
        state = "mixed" if positive and negative else ("positive_only" if positive else "negative_only")
        weight = group["confidence"].to_numpy(float)
        signed = group["signed_score"].to_numpy(float)
        rows.append({
            "firm_id": int(firm_id), "label_state": state,
            "class_label": 1 if state == "positive_only" else (-1 if state == "negative_only" else 0),
            "signed_text_score": float(np.average(signed, weights=weight)),
            "channels": " | ".join(sorted(group["channel"].unique())),
            "channel_count": int(group["channel"].nunique()),
            "maximum_confidence": float(group["confidence"].max()),
        })
    labels = pd.DataFrame(rows)
    channel_audit = (frame.assign(sign=np.where(frame["signed_score"] > 0, "positive", "negative"))
                     .groupby(["channel", "sign"], as_index=False)
                     .agg(firm_channel_labels=("firm_id", "size"), firms=("firm_id", "nunique")))
    return labels, channel_audit


def load_representations(root: Path) -> dict[str, pd.DataFrame]:
    base = root / "active_passive_extension/benchmarks"
    paths = {
        "all_s12_rs_binary": base / "all_s12_control/2021q4/rs_binary/asset_embeddings_10d.parquet",
        "all_s12_rs_l_min": base / "all_s12_control/2021q4/rs_l_min/asset_embeddings_10d.parquet",
    }
    for group in ["stable_stock_picker", "tactical_active", "benchmark_rebalancer", "passive_like"]:
        paths[f"{group}_rs_binary"] = base / f"behavioral_embeddings/2021q4/{group}/rs_binary/asset_embeddings_10d.parquet"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing embedding files: {missing}")
    return {name: pd.read_parquet(path).sort_values("permco").reset_index(drop=True) for name, path in paths.items()}


def loo_centroid_scores(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    x = l2_normalize(features)
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Labels must contain -1 and +1.")
    rows = []
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        positive = x[train & (y == 1)].mean(axis=0)
        negative = x[train & (y == -1)].mean(axis=0)
        direction = positive - negative
        norm = np.linalg.norm(direction)
        if norm <= 0:
            raise ValueError("Centroid direction has zero norm.")
        direction /= norm
        projection = float(x[i] @ direction)
        midpoint = float(0.5 * ((positive @ direction) + (negative @ direction)))
        cosine_difference = float(x[i] @ (positive - negative))
        rows.append({"centroid_projection": projection, "centroid_midpoint": midpoint,
                     "centroid_margin": projection - midpoint,
                     "class_balanced_cosine": cosine_difference})
    return pd.DataFrame(rows)


def score_metrics(y: np.ndarray, signed_text: np.ndarray, scores: np.ndarray) -> dict:
    y01 = (np.asarray(y) == 1).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= 0).astype(int)
    rho = spearmanr(np.asarray(signed_text, dtype=float), scores).statistic
    return {
        "auc": float(roc_auc_score(y01, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y01, predicted)),
        "spearman_signed_text": float(rho),
        "mean_positive_score": float(scores[y01 == 1].mean()),
        "mean_negative_score": float(scores[y01 == 0].mean()),
        "mean_separation": float(scores[y01 == 1].mean() - scores[y01 == 0].mean()),
    }


def evaluate_representation(frame: pd.DataFrame, labels: pd.DataFrame, representation: str,
                            dimensions: tuple[int, ...] = DIMENSIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = labels[labels["class_label"].ne(0)].merge(
        frame, left_on="firm_id", right_on="permco", how="inner", validate="one_to_one"
    ).sort_values("firm_id").reset_index(drop=True)
    if len(sample) != int(labels["class_label"].ne(0).sum()):
        raise ValueError(f"Incomplete labeled-firm coverage for {representation}.")
    metric_rows, prediction_frames = [], []
    for dimension in dimensions:
        columns = [f"embedding_{i:03d}" for i in range(1, dimension + 1)]
        scored = loo_centroid_scores(sample[columns].to_numpy(float), sample["class_label"].to_numpy(int))
        predictions = pd.concat([sample[["firm_id", "label_state", "class_label", "signed_text_score"]], scored], axis=1)
        predictions["representation"] = representation
        predictions["dimension"] = dimension
        for method, column in [("centroid_direction", "centroid_margin"),
                               ("class_balanced_cosine", "class_balanced_cosine")]:
            metrics = score_metrics(predictions["class_label"], predictions["signed_text_score"], predictions[column])
            metric_rows.append({"representation": representation, "dimension": dimension,
                                "method": method, "firms": len(predictions), **metrics})
        prediction_frames.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def permutation_test(frame: pd.DataFrame, labels: pd.DataFrame, representation: str, dimension: int = 10,
                     repetitions: int = 5000, seed: int = 20220224) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = labels[labels["class_label"].ne(0)].merge(
        frame, left_on="firm_id", right_on="permco", how="inner", validate="one_to_one"
    ).sort_values("firm_id").reset_index(drop=True)
    columns = [f"embedding_{i:03d}" for i in range(1, dimension + 1)]
    x = sample[columns].to_numpy(float)
    y = sample["class_label"].to_numpy(int)
    observed_scores = loo_centroid_scores(x, y)["centroid_margin"].to_numpy()
    observed = score_metrics(y, sample["signed_text_score"], observed_scores)
    rng = np.random.default_rng(seed)
    null_rows = []
    for iteration in range(repetitions):
        permuted = rng.permutation(y)
        scores = loo_centroid_scores(x, permuted)["centroid_margin"].to_numpy()
        metrics = score_metrics(permuted, permuted, scores)
        null_rows.append({"iteration": iteration + 1, "auc": metrics["auc"],
                          "balanced_accuracy": metrics["balanced_accuracy"],
                          "mean_separation": metrics["mean_separation"]})
    null = pd.DataFrame(null_rows)
    summary = pd.DataFrame([{
        "representation": representation, "dimension": dimension, "repetitions": repetitions,
        **{f"observed_{key}": observed[key] for key in ["auc", "balanced_accuracy", "mean_separation"]},
        "auc_permutation_p": float((1 + (null["auc"] >= observed["auc"]).sum()) / (repetitions + 1)),
        "balanced_accuracy_permutation_p": float((1 + (null["balanced_accuracy"] >= observed["balanced_accuracy"]).sum()) / (repetitions + 1)),
        "mean_separation_permutation_p": float((1 + (null["mean_separation"] >= observed["mean_separation"]).sum()) / (repetitions + 1)),
    }])
    null["representation"] = representation
    return summary, null


def industry_loo_scores(labels: pd.DataFrame, industry: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sample = labels[labels["class_label"].ne(0)].merge(
        industry[["permco", "sic2"]], left_on="firm_id", right_on="permco", how="left", validate="one_to_one"
    ).sort_values("firm_id").reset_index(drop=True)
    y = sample["class_label"].to_numpy(int)
    scores = []
    for i, row in sample.iterrows():
        train = sample.index != i
        same = train & sample["sic2"].eq(row["sic2"])
        use = same if same.sum() >= 2 else train
        scores.append(float((sample.loc[use, "class_label"].eq(1).mean()) - 0.5))
    sample["industry_score"] = scores
    metrics = score_metrics(y, sample["signed_text_score"], sample["industry_score"])
    return sample[["firm_id", "sic2", "industry_score"]], metrics


def characteristic_frame(root: Path, reference: pd.DataFrame) -> pd.DataFrame:
    columns = ["beta_60m_paper", "log_market_equity", "log_book_to_market",
               "at_gr1_paper", "gp_at_paper", "momentum_12_1"]
    raw = pd.read_parquet(root / "remaining_benchmarks/return_comovement/characteristics/2021q4.parquet")
    frame = reference[["permco", "firm_name"]].merge(raw, on="permco", how="left", validate="one_to_one")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    scale = values.std(ddof=0).replace(0, 1)
    frame[columns] = (values - values.mean()) / scale
    return frame[["permco", "firm_name", *columns]]


def residualize_industry(frame: pd.DataFrame, industry: pd.DataFrame, dimension: int) -> pd.DataFrame:
    columns = [f"embedding_{i:03d}" for i in range(1, dimension + 1)]
    merged = frame.merge(industry[["permco", "sic2"]], on="permco", how="left", validate="one_to_one")
    global_means = merged[columns].mean()
    industry_means = merged.groupby("sic2")[columns].transform("mean")
    residual = merged[columns] - industry_means.fillna(global_means)
    industry_counts = merged.groupby("sic2")["permco"].transform("size")
    singleton = merged["sic2"].isna() | industry_counts.lt(2)
    residual.loc[singleton, columns] = merged.loc[singleton, columns] - global_means
    zero_norm = np.linalg.norm(residual.to_numpy(float), axis=1) <= 1e-12
    residual.loc[zero_norm, columns] = merged.loc[zero_norm, columns] - global_means
    residual.columns = columns
    return pd.concat([merged[["permco", "firm_name"]], residual], axis=1)


def mixed_firm_scores(frame: pd.DataFrame, labels: pd.DataFrame, dimension: int = 10) -> pd.DataFrame:
    columns = [f"embedding_{i:03d}" for i in range(1, dimension + 1)]
    merged = labels.merge(frame, left_on="firm_id", right_on="permco", how="inner", validate="one_to_one")
    x = l2_normalize(merged[columns].to_numpy(float))
    training = merged["class_label"].ne(0).to_numpy()
    positive = x[training & merged["class_label"].eq(1).to_numpy()].mean(axis=0)
    negative = x[training & merged["class_label"].eq(-1).to_numpy()].mean(axis=0)
    direction = positive - negative
    direction /= np.linalg.norm(direction)
    midpoint = 0.5 * ((positive @ direction) + (negative @ direction))
    merged["full_direction_margin"] = x @ direction - midpoint
    return merged.loc[merged["label_state"].eq("mixed"),
                      ["firm_id", "firm_name", "channels", "signed_text_score", "full_direction_margin"]]


def write_design(path: Path, seed_path: Path, repetitions: int = 5000) -> dict:
    design = {
        "stage": "outcome_free_directional_centroid", "embedding_date": "2021Q4",
        "primary_representation": "all_s12_rs_binary", "primary_dimension": 10,
        "dimensions": list(DIMENSIONS), "mixed_firms_in_training": False,
        "centroid_weighting": "equal firm weight", "validation": "leave one labeled firm out",
        "permutation_repetitions": repetitions, "permutation_seed": 20220224,
        "event_returns_loaded": False,
        "seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    design["sha256"] = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(design, indent=2))
    return design
