from __future__ import annotations

import numpy as np
import pandas as pd


def demean_by_event(frame: pd.DataFrame, columns) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[f"{column}_dm"] = result[column] - result.groupby("event_date")[column].transform("mean")
    return result


def cluster_variance(x, residual, groups) -> float:
    work = pd.DataFrame({"x": x, "u": residual, "group": groups}).dropna()
    n = len(work)
    scores = (work.x * work.u).groupby(work.group).sum()
    g = len(scores)
    bread = float(np.square(work.x).sum())
    if g < 2 or bread <= 0:
        return np.nan
    correction = (g / (g - 1)) * ((n - 1) / max(n - 1, 1))
    return correction * float(np.square(scores).sum()) / bread**2


def interaction_regression(frame: pd.DataFrame, outcome: str, prediction: str) -> dict:
    data = frame[["event_date", "permco", outcome, prediction]].dropna().copy()
    data = demean_by_event(data, [outcome, prediction])
    y = data[f"{outcome}_dm"].to_numpy(float)
    x = data[f"{prediction}_dm"].to_numpy(float)
    xx = float(x @ x)
    slope = float(x @ y / xx) if xx > 0 else np.nan
    residual = y - slope * x
    v_event = cluster_variance(x, residual, data.event_date)
    v_firm = cluster_variance(x, residual, data.permco)
    v_observation = float(np.square(x * residual).sum() / xx**2) if xx > 0 else np.nan
    variance_two_way = v_event + v_firm - v_observation
    standard_error = np.sqrt(max(variance_two_way, 0)) if np.isfinite(variance_two_way) else np.nan
    t_statistic = slope / standard_error if standard_error and standard_error > 0 else np.nan
    variance_r2 = 1 - float(np.square(residual).sum()) / float(np.square(y).sum()) if np.square(y).sum() > 0 else np.nan
    return {
        "observations": len(data), "events": data.event_date.nunique(), "firms": data.permco.nunique(),
        "interaction_slope": slope, "two_way_clustered_se": standard_error,
        "t_statistic": t_statistic, "within_event_variance_r2": variance_r2,
    }


def event_correlations(frame: pd.DataFrame, outcome: str, prediction: str) -> pd.DataFrame:
    rows = []
    for event_date, group in frame.groupby("event_date", sort=True):
        valid = group[[outcome, prediction, "shock_value"]].dropna()
        rows.append({
            "event_date": event_date, "firms": len(valid), "shock_value": valid.shock_value.iloc[0] if len(valid) else np.nan,
            "pearson": valid[outcome].corr(valid[prediction]),
            "spearman": valid[outcome].corr(valid[prediction], method="spearman"),
        })
    return pd.DataFrame(rows)


def shuffled_slopes(frame: pd.DataFrame, outcome: str, shock: str, beta: str, repetitions=200, seed=20260902):
    rng = np.random.default_rng(seed)
    base = frame[["anchor_quarter", "event_date", "permco", outcome, shock, beta]].dropna().reset_index(drop=True)
    observed = interaction_regression(base.assign(prediction=base[shock] * base[beta]), outcome, "prediction")["interaction_slope"]
    distributions = []
    quarter_groups = {quarter: group.index.to_numpy() for quarter, group in base.groupby("anchor_quarter")}
    original = base[beta].to_numpy(copy=True)
    for repetition in range(repetitions):
        shuffled = original.copy()
        for indices in quarter_groups.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        candidate = base.copy()
        candidate["prediction"] = candidate[shock].to_numpy() * shuffled
        slope = interaction_regression(candidate, outcome, "prediction")["interaction_slope"]
        distributions.append({"repetition": repetition, "placebo_slope": slope})
    distribution = pd.DataFrame(distributions)
    p_upper = (1 + int((distribution.placebo_slope >= observed).sum())) / (1 + repetitions)
    return observed, p_upper, distribution
