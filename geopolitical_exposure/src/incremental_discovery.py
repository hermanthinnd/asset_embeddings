from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind


CAR_COLUMNS = ["car_0_1", "car_0_5", "car_1_20"]


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    scale = values.std(ddof=0)
    return (values - values.mean()) / scale if pd.notna(scale) and scale > 0 else values * 0


def build_panel(root: Path, representation: str = "all_s12") -> tuple[pd.DataFrame, pd.DataFrame]:
    expansion = root / "embedding_expansion_v2"
    text = root / "text_expansion_v2"
    corpus = pd.read_parquet(text / "processed/expanded_300_firms_frozen.parquet").rename(
        columns={"permco": "firm_id"}
    )
    corpus["firm_id"] = pd.to_numeric(corpus["firm_id"], errors="coerce").astype("Int64")

    responses = pd.read_parquet(text / "results/expanded_response_audit.parquet")
    responses["firm_id"] = pd.to_numeric(responses["firm_id"], errors="coerce").astype("Int64")
    contract_columns = ["contract_valid", "identity_valid", "source_hash_valid", "logical_empty_valid"]
    valid_firms = set(responses.loc[responses[contract_columns].all(axis=1), "firm_id"].dropna())

    seeds = pd.read_parquet(expansion / "directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds["firm_id"], errors="coerce").astype("Int64")
    channels = sorted(seeds["channel"].unique())
    firms = corpus[["firm_id", "comnam", "ticker", "sic2", "log_market_equity"]].drop_duplicates("firm_id")
    grid = firms.assign(_key=1).merge(pd.DataFrame({"channel": channels, "_key": 1}), on="_key").drop(columns="_key")
    direct = grid.merge(
        seeds[["firm_id", "channel", "signed_score", "confidence", "source_documents"]],
        on=["firm_id", "channel"], how="left", validate="one_to_one",
    )
    direct["text_reviewed"] = direct["firm_id"].isin(valid_firms)
    direct["text_state"] = np.select(
        [direct["signed_score"].notna(), direct["text_reviewed"]],
        ["direct_exposure", "reviewed_no_channel_exposure"], default="text_unavailable",
    )
    direct.loc[direct["text_reviewed"] & direct["signed_score"].isna(), "signed_score"] = 0.0
    direct = direct.rename(columns={"signed_score": "direct_text_score"})

    propagated = pd.read_parquet(expansion / "embedding_propagated_scores.parquet")
    propagated = propagated.loc[propagated["representation"].eq(representation)].copy()
    propagated["permco"] = pd.to_numeric(propagated["permco"], errors="coerce").astype("Int64")
    loo = pd.read_parquet(expansion / "leave_one_seed_out_predictions.parquet")
    loo = loo.loc[loo["representation"].eq(representation)].copy()
    loo["firm_id"] = pd.to_numeric(loo["firm_id"], errors="coerce").astype("Int64")
    propagated = propagated.merge(
        loo[["firm_id", "channel", "predicted_signed_score"]],
        left_on=["permco", "channel"], right_on=["firm_id", "channel"], how="left",
    )
    propagated["embedding_score"] = propagated["propagated_score"]
    replace = propagated["is_seed"] & propagated["predicted_signed_score"].notna()
    propagated.loc[replace, "embedding_score"] = propagated.loc[replace, "predicted_signed_score"]

    characteristics = pd.read_parquet(expansion / "characteristic_propagated_scores.parquet").rename(
        columns={"propagated_score": "characteristic_score"}
    )
    characteristics["permco"] = pd.to_numeric(characteristics["permco"], errors="coerce").astype("Int64")
    cars = pd.read_parquet(root / "event_data/market_model_cars.parquet")
    cars["permco"] = pd.to_numeric(cars["permco"], errors="coerce").astype("Int64")

    panel = direct.merge(
        propagated[["permco", "channel", "embedding_score", "nonseed_percentile", "is_seed"]],
        left_on=["firm_id", "channel"], right_on=["permco", "channel"], how="left", validate="one_to_one",
    )
    panel = panel.merge(
        characteristics[["permco", "channel", "characteristic_score"]],
        on=["permco", "channel"], how="left", validate="one_to_one",
    ).merge(cars, on="permco", how="left", validate="many_to_one")
    panel["embedding_only_candidate"] = panel["text_state"].eq("reviewed_no_channel_exposure") & panel["embedding_score"].notna()
    return panel, seeds


def _fit(frame: pd.DataFrame, outcome: str, predictors: list[str], label: str) -> dict:
    needed = [outcome, "log_market_equity", "sic2"] + predictors
    sample = frame.dropna(subset=needed).copy()
    design = pd.DataFrame(index=sample.index)
    for column in predictors:
        design[column] = zscore(sample[column])
    design["log_market_equity"] = zscore(sample["log_market_equity"])
    industry = sample["sic2"].astype(str)
    counts = industry.value_counts()
    industry = industry.where(industry.map(counts).ge(5), "OTHER")
    industries = pd.get_dummies(industry, prefix="sic2", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([design, industries], axis=1).astype(float), has_constant="add")
    model = sm.OLS(pd.to_numeric(sample[outcome], errors="coerce").astype(float), design).fit(cov_type="HC3")
    row = {"channel": sample["channel"].iloc[0], "outcome": outcome, "specification": label,
           "firms": len(sample), "adjusted_r2": model.rsquared_adj, "aic": model.aic}
    for predictor in predictors:
        row[f"{predictor}_coefficient"] = model.params.get(predictor, np.nan)
        row[f"{predictor}_pvalue"] = model.pvalues.get(predictor, np.nan)
    return row


def run_model_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reviewed = panel.loc[panel["text_reviewed"]]
    specifications = [
        (["direct_text_score"], "text_only"),
        (["embedding_score"], "embedding_only"),
        (["characteristic_score"], "characteristics_only"),
        (["direct_text_score", "embedding_score"], "text_plus_embedding"),
    ]
    for _, group in reviewed.groupby("channel"):
        for outcome in CAR_COLUMNS:
            for predictors, label in specifications:
                rows.append(_fit(group, outcome, predictors, label))
    return pd.DataFrame(rows)


def evaluate_embedding_only(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = panel.loc[panel["embedding_only_candidate"]]
    for channel, group in candidates.groupby("channel"):
        for outcome in CAR_COLUMNS:
            sample = group.dropna(subset=["embedding_score", outcome]).copy()
            if len(sample) < 20 or sample["embedding_score"].nunique() < 2:
                continue
            lower, upper = sample["embedding_score"].quantile([0.2, 0.8])
            bottom = sample.loc[sample["embedding_score"] <= lower, outcome].astype(float)
            top = sample.loc[sample["embedding_score"] >= upper, outcome].astype(float)
            rho, rho_p = spearmanr(sample["embedding_score"], sample[outcome])
            _, spread_p = ttest_ind(top, bottom, equal_var=False, nan_policy="omit")
            rows.append({"channel": channel, "outcome": outcome, "firms": len(sample),
                         "spearman_rho": rho, "spearman_pvalue": rho_p,
                         "top_minus_bottom_car": top.mean() - bottom.mean(), "spread_pvalue": spread_p,
                         "directional_agreement": float((np.sign(sample["embedding_score"]) * sample[outcome] > 0).mean())})
    return pd.DataFrame(rows)


def rank_candidates(panel: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    rows = []
    candidates = panel.loc[panel["embedding_only_candidate"]]
    for _, group in candidates.groupby("channel"):
        group = group.copy()
        group["embedding_z"] = zscore(group["embedding_score"])
        group["absolute_embedding_z"] = group["embedding_z"].abs()
        rows.append(group.nlargest(count, "absolute_embedding_z"))
    result = pd.concat(rows, ignore_index=True)
    columns = ["channel", "firm_id", "comnam", "ticker", "sic2", "embedding_score", "embedding_z",
               "characteristic_score"] + CAR_COLUMNS
    result = result[columns]
    result["return_direction_agrees_0_5"] = np.sign(result["embedding_score"]) * result["car_0_5"] > 0
    return result
