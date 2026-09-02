"""Frozen event-study helpers for the geopolitical-exposure proof of concept."""

from __future__ import annotations

import numpy as np
import pandas as pd


def value_weight_firm_returns(security_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate security returns to PERMCO using lagged market equity."""
    frame = security_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["ret", "prc", "shrout"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["market_equity"] = frame["prc"].abs() * frame["shrout"]
    frame = frame.sort_values(["permno", "date"])
    frame["lag_market_equity"] = frame.groupby("permno")["market_equity"].shift()

    def aggregate(group: pd.DataFrame) -> pd.Series:
        valid = group["ret"].notna()
        weighted = valid & group["lag_market_equity"].gt(0)
        if weighted.any():
            weights = group.loc[weighted, "lag_market_equity"]
            value = np.average(group.loc[weighted, "ret"], weights=weights)
        elif valid.any():
            value = group.loc[valid, "ret"].mean()
        else:
            value = np.nan
        return pd.Series({"firm_return": value, "securities": int(valid.sum())})

    return (
        frame.groupby(["permco", "date"], as_index=False, sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index(drop=True)
    )


def market_model_cars(
    firm_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    event_date: str,
    estimation_window=(-250, -30),
    outcome_windows=((0, 1), (0, 5), (1, 20)),
    minimum_estimation_days=120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate firm market models and return abnormal returns and CARs."""
    market = market_returns[["date", "market_return"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    calendar = market.dropna().sort_values("date").drop_duplicates("date")
    event_date = pd.Timestamp(event_date)
    if event_date not in set(calendar["date"]):
        raise ValueError(f"Event date {event_date.date()} is not in the market calendar.")
    event_position = int(calendar.index[calendar["date"].eq(event_date)][0])
    calendar = calendar.reset_index(drop=True)
    event_position = int(calendar.index[calendar["date"].eq(event_date)][0])
    calendar["event_time"] = calendar.index - event_position

    panel = firm_returns.merge(calendar, on="date", how="inner", validate="many_to_one")
    rows = []
    abnormal_frames = []
    lo, hi = estimation_window
    for permco, group in panel.groupby("permco", sort=True):
        estimation = group[group["event_time"].between(lo, hi)].dropna(
            subset=["firm_return", "market_return"]
        )
        if len(estimation) < minimum_estimation_days:
            continue
        x = np.column_stack([np.ones(len(estimation)), estimation["market_return"]])
        y = estimation["firm_return"].to_numpy()
        alpha, beta = np.linalg.lstsq(x, y, rcond=None)[0]
        evaluated = group.copy()
        evaluated["expected_return"] = alpha + beta * evaluated["market_return"]
        evaluated["abnormal_return"] = evaluated["firm_return"] - evaluated["expected_return"]
        abnormal_frames.append(evaluated.assign(alpha=alpha, beta=beta))
        record = {
            "permco": permco,
            "alpha": alpha,
            "beta": beta,
            "estimation_observations": len(estimation),
        }
        for start, end in outcome_windows:
            values = evaluated.loc[
                evaluated["event_time"].between(start, end), "abnormal_return"
            ]
            label = f"car_{start}_{end}"
            record[label] = values.sum(min_count=end - start + 1)
            record[f"{label}_observations"] = int(values.notna().sum())
        rows.append(record)
    abnormal = pd.concat(abnormal_frames, ignore_index=True) if abnormal_frames else pd.DataFrame()
    return pd.DataFrame(rows), abnormal


def industry_seed_scores(
    universe: pd.DataFrame, seeds: pd.DataFrame, industry_column="sic2"
) -> pd.DataFrame:
    """Propagate signed seed scores only to firms sharing the same industry."""
    rows = []
    seed_industry = seeds.merge(
        universe[["permco", industry_column]],
        left_on="firm_id",
        right_on="permco",
        how="left",
        validate="many_to_one",
    )
    for channel, group in seed_industry.groupby("channel", sort=True):
        weights = group["signed_score"] * group["confidence"]
        mapping = (
            pd.DataFrame({industry_column: group[industry_column], "weighted": weights})
            .dropna()
            .groupby(industry_column)["weighted"]
            .sum()
        )
        denominator = (
            pd.DataFrame({industry_column: group[industry_column], "weight": weights.abs()})
            .dropna()
            .groupby(industry_column)["weight"]
            .sum()
        )
        score_map = mapping.div(denominator).replace([np.inf, -np.inf], np.nan)
        frame = universe[["permco", "firm_name", industry_column]].copy()
        frame["representation"] = "industry_similarity"
        frame["channel"] = channel
        frame["propagated_score"] = frame[industry_column].map(score_map).fillna(0.0)
        frame["is_seed"] = frame["permco"].isin(seeds["firm_id"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
