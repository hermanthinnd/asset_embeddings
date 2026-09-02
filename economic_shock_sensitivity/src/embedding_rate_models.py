from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_quarter_frame(root: Path, target: pd.DataFrame, quarter: str, model: str, dimension: int,
                       embedding_lag: int, add_characteristics: bool = False,
                       characteristics_only: bool = False) -> tuple[pd.DataFrame, list[str], str]:
    period = pd.Period(quarter.upper(), freq="Q")
    input_label = str(period - embedding_lag).lower()
    base = target[target.anchor_quarter.eq(quarter)].copy()
    characteristic_columns = [
        "beta_60m_paper", "log_market_equity", "log_book_to_market",
        "at_gr1_paper", "gp_at_paper", "momentum_12_1",
    ]
    features = []
    if not characteristics_only:
        path = root / "full_replication_80q" / "embeddings" / input_label / model / "asset_embeddings_normalized_labeled.parquet"
        embedding = pd.read_parquet(path)
        features = [f"embedding_{number:03d}" for number in range(1, dimension + 1)]
        base = base.merge(embedding[["permco", *features]], on="permco", how="inner", validate="one_to_one")
    if add_characteristics or characteristics_only:
        path = root / "remaining_benchmarks" / "return_comovement" / "characteristics" / f"{input_label}.parquet"
        characteristics = pd.read_parquet(path)
        base = base.merge(characteristics[["permco", *characteristic_columns]], on="permco", how="inner", validate="one_to_one")
        features += characteristic_columns
    return base, features, input_label


def clipped_target(frame: pd.DataFrame, fit_splits=("train",), lower=0.01, upper=0.99):
    fit = frame.firm_split.isin(fit_splits) & frame.beta_raw.notna()
    bounds = frame.loc[fit, "beta_raw"].quantile([lower, upper])
    return frame.beta_raw.clip(bounds.iloc[0], bounds.iloc[1]), float(bounds.iloc[0]), float(bounds.iloc[1])


def fit_predict(frame: pd.DataFrame, features: list[str], alpha: float,
                fit_splits=("train",), predict_split="validation") -> tuple[pd.DataFrame, dict]:
    data = frame.copy()
    data["target"], lower, upper = clipped_target(data, fit_splits)
    fit = data.firm_split.isin(fit_splits) & data.target.notna()
    predict = data.firm_split.eq(predict_split) & data.target.notna()
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipeline.fit(data.loc[fit, features], data.loc[fit, "target"])
    output = data.loc[predict, ["anchor_quarter", "permco", "firm_split", "target", "beta_raw"]].copy()
    output["prediction"] = pipeline.predict(data.loc[predict, features])
    training_mean = float(data.loc[fit, "target"].mean())
    output["training_mean"] = training_mean
    residual_ss = float(np.square(output.target - output.prediction).sum())
    variance_ss = float(np.square(output.target - training_mean).sum())
    conventional_ss = float(np.square(output.target - output.target.mean()).sum())
    summary = {
        "fit_firms": int(fit.sum()), "prediction_firms": int(predict.sum()),
        "winsor_lower_train": lower, "winsor_upper_train": upper,
        "variance_r2": 1 - residual_ss / variance_ss if variance_ss > 0 else np.nan,
        "conventional_r2": 1 - residual_ss / conventional_ss if conventional_ss > 0 else np.nan,
        "correlation": output.target.corr(output.prediction),
        "residual_ss": residual_ss, "variance_ss": variance_ss,
    }
    return output, summary


def pooled_variance_r2(summaries: pd.DataFrame) -> float:
    return 1 - summaries.residual_ss.sum() / summaries.variance_ss.sum()


def load_behavioral_frame(root: Path, target: pd.DataFrame, quarter: str, style: str, model: str,
                          embedding_lag: int, add_characteristics: bool = True):
    period = pd.Period(quarter.upper(), freq="Q")
    input_label = str(period - embedding_lag).lower()
    base = target[target.anchor_quarter.eq(quarter)].copy()
    path = root / "active_passive_extension" / "benchmarks" / "behavioral_embeddings" / input_label / style / model / "asset_embeddings_10d.parquet"
    embedding = pd.read_parquet(path)
    embedding_columns = [f"embedding_{number:03d}" for number in range(1, 11)]
    base = base.merge(embedding[["permco", *embedding_columns]], on="permco", how="inner", validate="one_to_one")
    characteristic_columns = [
        "beta_60m_paper", "log_market_equity", "log_book_to_market",
        "at_gr1_paper", "gp_at_paper", "momentum_12_1",
    ]
    if add_characteristics:
        char_path = root / "remaining_benchmarks" / "return_comovement" / "characteristics" / f"{input_label}.parquet"
        characteristics = pd.read_parquet(char_path)
        base = base.merge(characteristics[["permco", *characteristic_columns]], on="permco", how="inner", validate="one_to_one")
    return base, embedding_columns + (characteristic_columns if add_characteristics else []), input_label
