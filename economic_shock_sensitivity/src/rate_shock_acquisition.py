from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def rollback(db) -> None:
    try:
        db.connection.rollback()
    except Exception:
        pass


def first_queryable(db, candidates):
    errors = []
    for candidate in candidates:
        try:
            db.raw_sql(candidate["probe"])
            return candidate
        except Exception as error:
            errors.append({
                "candidate": candidate["name"],
                "error_type": type(error).__name__,
                "message": str(error).splitlines()[0][:240],
            })
            rollback(db)
    raise RuntimeError(f"No CRSP daily source was queryable: {errors}")


def build_event_trading_map(events: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates).dropna().drop_duplicates().sort_values())
    rows = []
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.event_date)
        location = int(dates.searchsorted(event_date, side="left"))
        # Unscheduled announcements can occur on weekends. Their event return
        # begins on the first trading session on or after the announcement.
        if location >= len(dates):
            raise ValueError(f"Trading calendar does not extend beyond {event_date.date()}.")
        effective_event_date = dates[location]
        nontrading_announcement = effective_event_date != event_date
        if location == 0 or location + 1 >= len(dates):
            raise ValueError(f"Trading calendar does not bracket {event_date.date()}.")
        for role, position in [("minus_1", location - 1), ("event_0", location), ("plus_1", location + 1)]:
            rows.append({
                "event_date": event_date,
                "effective_event_date": effective_event_date,
                "nontrading_announcement": nontrading_announcement,
                "calendar_days_to_market_open": int((effective_event_date - event_date).days),
                "shock_stmt": float(event.shock_stmt),
                "shock_me": float(event.shock_me),
                "role": role,
                "trading_date": dates[position],
            })
    result = pd.DataFrame(rows)
    if result.duplicated(["event_date", "role"]).any():
        raise AssertionError("Event/role mapping is not unique.")
    return result


def sql_date_list(values) -> str:
    dates = sorted(pd.to_datetime(pd.Series(values).dropna().unique()))
    return ",".join(f"'{pd.Timestamp(value).date().isoformat()}'" for value in dates)


def aggregate_event_returns(security: pd.DataFrame, event_map: pd.DataFrame, market: pd.DataFrame):
    frame = security.copy()
    for column in ["permno", "permco", "ret", "prc", "shrout"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["permno", "permco", "date"]).copy()
    frame["market_equity"] = frame["prc"].abs() * frame["shrout"]

    expanded = event_map.merge(frame, left_on="trading_date", right_on="date", how="left")
    values = expanded.pivot_table(
        index=["event_date", "shock_stmt", "shock_me", "permco", "permno"],
        columns="role", values=["ret", "market_equity"], aggfunc="first",
    )
    values.columns = [f"{value}_{role}" for value, role in values.columns]
    values = values.reset_index()
    for column in ["ret_event_0", "ret_plus_1", "market_equity_minus_1"]:
        if column not in values:
            values[column] = np.nan
    values["security_return_00"] = values["ret_event_0"]
    values["security_return_01"] = (1 + values["ret_event_0"]) * (1 + values["ret_plus_1"]) - 1
    values["lag_weight"] = values["market_equity_minus_1"].where(values["market_equity_minus_1"] > 0)

    def weighted(group, outcome):
        valid = group[outcome].notna() & group["lag_weight"].notna()
        if not valid.any():
            return np.nan
        return np.average(group.loc[valid, outcome], weights=group.loc[valid, "lag_weight"])

    firm_rows = []
    for keys, group in values.groupby(["event_date", "shock_stmt", "shock_me", "permco"], sort=False):
        firm_rows.append({
            "event_date": keys[0], "shock_stmt": keys[1], "shock_me": keys[2], "permco": int(keys[3]),
            "firm_return_00": weighted(group, "security_return_00"),
            "firm_return_01": weighted(group, "security_return_01"),
            "lag_market_equity": group["lag_weight"].sum(min_count=1),
            "underlying_permnos": int(group["permno"].nunique()),
        })
    firms = pd.DataFrame(firm_rows)

    market_frame = market.copy()
    market_frame["date"] = pd.to_datetime(market_frame["date"])
    market_frame["market_return"] = pd.to_numeric(market_frame["market_return"], errors="coerce")
    market_event = event_map.merge(market_frame, left_on="trading_date", right_on="date", how="left")
    market_wide = market_event.pivot(index="event_date", columns="role", values="market_return").reset_index()
    market_wide["market_return_00"] = market_wide["event_0"]
    market_wide["market_return_01"] = (1 + market_wide["event_0"]) * (1 + market_wide["plus_1"]) - 1
    firms = firms.merge(market_wide[["event_date", "market_return_00", "market_return_01"]], on="event_date", how="left", validate="many_to_one")
    firms["abnormal_return_00"] = firms["firm_return_00"] - firms["market_return_00"]
    firms["abnormal_return_01"] = firms["firm_return_01"] - firms["market_return_01"]
    return firms, values


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
