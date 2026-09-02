from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


def quarter_holdings(root: Path, period: pd.Period) -> pd.DataFrame:
    path = root / "full_replication_80q" / "processed" / str(period).lower() / "combined_final_holdings.parquet"
    frame = pd.read_parquet(path, columns=["source", "investor_id", "investor_key", "permco", "portfolio_weight"])
    frame = frame.loc[frame.source.eq("s12"), ["investor_id", "investor_key", "permco", "portfolio_weight"]].copy()
    frame["portfolio_weight"] = pd.to_numeric(frame.portfolio_weight, errors="coerce")
    return frame.dropna(subset=["portfolio_weight"])


def construct_response(root: Path, period: pd.Period, returns: pd.DataFrame,
                       activeness: pd.DataFrame) -> pd.DataFrame:
    previous = period - 1
    pre = quarter_holdings(root, previous).rename(columns={"portfolio_weight": "weight_pre"})
    post = quarter_holdings(root, period).rename(columns={"portfolio_weight": "weight_post"})
    path = root / "economic_shock_sensitivity/data/processed/rolling_rate_targets" / str(period).lower() / "stmt_ar00.parquet"
    beta = pd.read_parquet(path, columns=["permco", "beta_winsorized", "eligible"])
    beta = beta.loc[beta.eligible, ["permco", "beta_winsorized"]]
    ret = returns.loc[(returns.year.eq(period.year)) & (returns.quarter.eq(period.quarter)), ["permco", "quarter_return"]]
    universe = beta.merge(ret, on="permco", how="inner", validate="one_to_one")
    positions = pre.merge(post, on=["investor_id", "investor_key", "permco"], how="outer")
    positions = positions.merge(universe, on="permco", how="inner", validate="many_to_one")
    positions[["weight_pre", "weight_post"]] = positions[["weight_pre", "weight_post"]].fillna(0.0)
    positions["pre_sum"] = positions.groupby("investor_key").weight_pre.transform("sum")
    positions["post_sum"] = positions.groupby("investor_key").weight_post.transform("sum")
    positions = positions.loc[(positions.pre_sum > 0) & (positions.post_sum > 0)].copy()
    positions["weight_pre_analysis"] = positions.weight_pre / positions.pre_sum
    positions["weight_post_analysis"] = positions.weight_post / positions.post_sum
    positions["drift_value"] = positions.weight_pre_analysis * (1 + positions.quarter_return.clip(lower=-1))
    positions["drift_sum"] = positions.groupby("investor_key").drift_value.transform("sum")
    positions = positions.loc[positions.drift_sum > 0].copy()
    positions["passive_weight"] = positions.drift_value / positions.drift_sum
    positions["active_trade_weight"] = positions.weight_post_analysis - positions.passive_weight
    positions["active_rate_tilt_contribution"] = positions.active_trade_weight * positions.beta_winsorized
    response = positions.groupby(["investor_id", "investor_key"], as_index=False).agg(
        active_rate_tilt_change=("active_rate_tilt_contribution", "sum"),
        return_adjusted_turnover_recomputed=("active_trade_weight", lambda x: 0.5 * x.abs().sum()),
        beta_firms=("permco", "nunique"), pre_coverage=("pre_sum", "first"),
        post_coverage=("post_sum", "first"), weight_balance=("active_trade_weight", "sum"),
    )
    style = activeness.loc[(activeness.year.eq(previous.year)) & (activeness.quarter.eq(previous.quarter)),
                           ["investor_id", "behavioral_group", "active_share", "return_adjusted_turnover"]].drop_duplicates("investor_id")
    response = response.merge(style, on="investor_id", how="left", validate="one_to_one")
    response["quarter"] = str(period).lower()
    return response


def hac_quarterly_regression(panel: pd.DataFrame, outcome: str, shock: str, maxlags: int = 4) -> dict:
    frame = panel[[outcome, shock]].dropna()
    fit = sm.OLS(frame[outcome].astype(float), sm.add_constant(frame[shock].astype(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"observations": len(frame), "slope": fit.params[shock], "standard_error": fit.bse[shock],
            "t_statistic": fit.tvalues[shock], "p_value": fit.pvalues[shock], "r_squared": fit.rsquared}
