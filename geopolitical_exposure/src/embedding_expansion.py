"""Outcome-free propagation of frozen text exposure through asset embeddings."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms == 0):
        raise ValueError("Embedding matrix contains non-finite or zero-norm rows.")
    return values / norms


def propagate_channel(
    firms: pd.DataFrame,
    normalized_features: np.ndarray,
    seed_rows: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """Propagate signed confidence-adjusted seed scores by cosine similarity."""
    index_by_firm = pd.Series(np.arange(len(firms)), index=firms["permco"].astype(int))
    missing = sorted(set(seed_rows["firm_id"].astype(int)).difference(index_by_firm.index))
    if missing:
        raise ValueError(f"Seeds missing from {feature_name}: {missing}")
    seed_indices = index_by_firm.loc[seed_rows["firm_id"].astype(int)].to_numpy()
    seed_weights = (seed_rows["signed_score"].astype(float) * seed_rows["confidence"].astype(float)).to_numpy()
    denominator = np.abs(seed_weights).sum()
    if denominator <= 0:
        raise ValueError("Channel seed weights have zero absolute mass.")
    similarities = normalized_features @ normalized_features[seed_indices].T
    propagated = similarities @ seed_weights / denominator
    maximum_similarity = similarities.max(axis=1)
    output = firms[["permco", "firm_name"]].copy()
    output["representation"] = feature_name
    output["channel"] = str(seed_rows["channel"].iloc[0])
    output["propagated_score"] = propagated
    output["maximum_seed_similarity"] = maximum_similarity
    output["channel_seed_count"] = len(seed_rows)
    output["is_seed"] = output["permco"].isin(seed_rows["firm_id"].astype(int))
    nonseed = ~output["is_seed"]
    output["nonseed_percentile"] = np.nan
    output.loc[nonseed, "nonseed_percentile"] = output.loc[nonseed, "propagated_score"].rank(pct=True)
    return output


def propagate_all_channels(
    firms: pd.DataFrame,
    feature_columns: Iterable[str],
    seeds: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    features = l2_normalize(firms[list(feature_columns)].to_numpy())
    frames = [
        propagate_channel(firms, features, group, feature_name)
        for _, group in seeds.groupby("channel", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)


def standardized_characteristics(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    """Median-impute and standardize pre-event characteristics without outcomes."""
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    scale = values.std(ddof=0).replace(0, 1)
    standardized = (values - values.mean()) / scale
    return l2_normalize(standardized.to_numpy())
