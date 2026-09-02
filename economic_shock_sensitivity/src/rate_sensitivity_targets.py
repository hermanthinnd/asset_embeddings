from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_grouped_betas(frame: pd.DataFrame, shock: str, outcome: str, minimum_events: int = 20) -> pd.DataFrame:
    data = frame[["permco", shock, outcome]].dropna().copy()
    data = data.rename(columns={shock: "x", outcome: "y"})
    data["x2"] = data["x"] ** 2
    data["xy"] = data["x"] * data["y"]
    data["y2"] = data["y"] ** 2
    grouped = data.groupby("permco", as_index=False).agg(
        events=("y", "size"), sx=("x", "sum"), sy=("y", "sum"),
        sxx=("x2", "sum"), sxy=("xy", "sum"), syy=("y2", "sum"),
    )
    grouped["centered_xx"] = grouped.sxx - grouped.sx.pow(2) / grouped.events
    grouped["centered_xy"] = grouped.sxy - grouped.sx * grouped.sy / grouped.events
    grouped["centered_yy"] = grouped.syy - grouped.sy.pow(2) / grouped.events
    grouped["beta_raw"] = grouped.centered_xy / grouped.centered_xx
    grouped["intercept"] = (grouped.sy - grouped.beta_raw * grouped.sx) / grouped.events
    grouped["sse"] = (grouped.centered_yy - grouped.beta_raw * grouped.centered_xy).clip(lower=0)
    grouped["residual_variance"] = grouped.sse / (grouped.events - 2)
    grouped["beta_standard_error"] = np.sqrt(grouped.residual_variance / grouped.centered_xx)
    grouped["regression_r2"] = grouped.centered_xy.pow(2) / (grouped.centered_xx * grouped.centered_yy)
    eligible = grouped.events.ge(minimum_events) & grouped.centered_xx.gt(0) & grouped.beta_raw.notna() & grouped.beta_standard_error.notna()
    grouped["eligible"] = eligible
    return grouped


def add_cross_sectional_shrinkage(frame: pd.DataFrame, lower=0.01, upper=0.99) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    valid = result.eligible & result.beta_raw.notna() & result.beta_standard_error.notna()
    cross = result.loc[valid, "beta_raw"]
    lower_bound, upper_bound = cross.quantile([lower, upper]) if len(cross) else (np.nan, np.nan)
    result["beta_winsorized"] = result.beta_raw.clip(lower=lower_bound, upper=upper_bound)
    prior_mean = result.loc[valid, "beta_winsorized"].mean()
    observed_variance = result.loc[valid, "beta_winsorized"].var(ddof=1)
    mean_noise_variance = result.loc[valid, "beta_standard_error"].pow(2).mean()
    prior_variance = max(float(observed_variance - mean_noise_variance), 0.0) if np.isfinite(observed_variance) else 0.0
    noise = result.beta_standard_error.pow(2)
    result["reliability"] = prior_variance / (prior_variance + noise) if prior_variance > 0 else 0.0
    result["beta_shrunk"] = prior_mean + result.reliability * (result.beta_winsorized - prior_mean)
    result.loc[~valid, ["beta_winsorized", "reliability", "beta_shrunk"]] = np.nan
    audit = {
        "eligible_firms": int(valid.sum()), "winsor_lower": lower_bound, "winsor_upper": upper_bound,
        "prior_mean": prior_mean, "observed_variance": observed_variance,
        "mean_noise_variance": mean_noise_variance, "estimated_prior_variance": prior_variance,
        "median_reliability": result.loc[valid, "reliability"].median(),
    }
    return result, audit
