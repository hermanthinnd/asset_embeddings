"""Outcome-free construction and validation of expanded text-exposure seeds."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_directional_seeds(final_exposures):
    frame = final_exposures.copy()
    frame["firm_id"] = pd.to_numeric(frame["firm_id"], errors="raise").astype(int)
    frame["signed_score"] = pd.to_numeric(frame["signed_score"], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    ambiguous = frame[frame["direction"].eq("ambiguous") | frame["signed_score"].isna()].copy()
    directional = frame.drop(ambiguous.index).copy()
    directional["weighted_score"] = directional["signed_score"] * directional["confidence"]
    seeds = (directional.groupby(["firm_id", "channel"], as_index=False)
             .agg(weighted_score_sum=("weighted_score", "sum"),
                  confidence_sum=("confidence", "sum"),
                  confidence=("confidence", "max"),
                  source_documents=("document_id", "nunique"),
                  source_exposures=("document_id", "size")))
    seeds["signed_score"] = seeds["weighted_score_sum"] / seeds["confidence_sum"]
    seeds = seeds.drop(columns=["weighted_score_sum", "confidence_sum"])
    if seeds.duplicated(["firm_id", "channel"]).any():
        raise ValueError("Firm-channel aggregation did not produce unique seeds.")
    return seeds, ambiguous


def seed_hash(seeds):
    columns = ["firm_id", "channel", "signed_score", "confidence",
               "source_documents", "source_exposures"]
    payload = seeds[columns].sort_values(["firm_id", "channel"]).to_dict("records")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def leave_one_seed_out(firms, normalized_features, seeds, representation):
    index_by_firm = pd.Series(np.arange(len(firms)), index=firms["permco"].astype(int))
    rows = []
    for channel, group in seeds.groupby("channel", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < 3:
            continue
        missing = set(group.firm_id.astype(int)).difference(index_by_firm.index)
        if missing:
            raise ValueError(f"Seeds missing from {representation}: {sorted(missing)}")
        indices = index_by_firm.loc[group.firm_id.astype(int)].to_numpy()
        weights = (group.signed_score * group.confidence).to_numpy(float)
        seed_features = normalized_features[indices]
        similarities = seed_features @ seed_features.T
        for position, row in group.iterrows():
            keep = np.arange(len(group)) != position
            denominator = np.abs(weights[keep]).sum()
            prediction = np.nan if denominator <= 0 else similarities[position, keep] @ weights[keep] / denominator
            rows.append({
                "representation": representation, "channel": channel,
                "firm_id": int(row.firm_id), "actual_signed_score": float(row.signed_score),
                "predicted_signed_score": prediction, "training_seeds": int(keep.sum()),
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["absolute_error"] = (result.actual_signed_score - result.predicted_signed_score).abs()
        result["sign_correct"] = np.sign(result.actual_signed_score) == np.sign(result.predicted_signed_score)
    return result


def summarize_leave_one_out(predictions):
    rows = []
    for (representation, channel), group in predictions.groupby(["representation", "channel"]):
        valid = group.dropna(subset=["actual_signed_score", "predicted_signed_score"])
        rows.append({
            "representation": representation, "channel": channel, "seed_predictions": len(valid),
            "pearson_correlation": valid.actual_signed_score.corr(valid.predicted_signed_score),
            "spearman_correlation": valid.actual_signed_score.corr(valid.predicted_signed_score, method="spearman"),
            "mean_absolute_error": valid.absolute_error.mean(),
            "sign_accuracy": valid.sign_correct.mean(),
        })
    return pd.DataFrame(rows)
