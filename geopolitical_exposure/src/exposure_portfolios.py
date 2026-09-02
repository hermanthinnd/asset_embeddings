from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


POSITIVE_CHANNELS = ["defense_demand", "oil_gas_production"]
NEGATIVE_CHANNELS = ["russia_ukraine_operations_revenue", "sanctions_sensitive_supply_chain", "shipping_logistics"]
CAR_COLUMNS = ["car_0_1", "car_0_5", "car_1_20"]


def percentile_strength(frame: pd.DataFrame, channels: list[str], positive: bool) -> pd.DataFrame:
    subset = frame.loc[frame["channel"].isin(channels)].copy()
    subset["strength"] = subset.groupby("channel")["propagated_score"].rank(pct=True, ascending=positive)
    # For positive channels, high raw scores should have high strength. For negative channels,
    # more-negative raw scores should have high strength.
    if positive:
        subset["strength"] = subset.groupby("channel")["propagated_score"].rank(pct=True, ascending=True)
    else:
        subset["strength"] = subset.groupby("channel")["propagated_score"].rank(pct=True, ascending=False)
    best = subset.sort_values("strength").groupby("permco", as_index=False).tail(1)
    return best[["permco", "firm_name", "channel", "strength", "propagated_score"]]


def build_score_universe(root: Path, representation: str = "all_s12") -> tuple[pd.DataFrame, pd.DataFrame]:
    expansion = root / "embedding_expansion_v2"
    seeds = pd.read_parquet(expansion / "directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds["firm_id"], errors="coerce").astype("Int64")
    seed_ids = set(seeds["firm_id"].dropna())
    scores = pd.read_parquet(expansion / "embedding_propagated_scores.parquet")
    scores = scores.loc[scores["representation"].eq(representation)].copy()
    scores["permco"] = pd.to_numeric(scores["permco"], errors="coerce").astype("Int64")
    positive = percentile_strength(scores, POSITIVE_CHANNELS, True).rename(
        columns={"channel": "positive_channel", "strength": "positive_strength", "propagated_score": "positive_raw_score"}
    )
    negative = percentile_strength(scores, NEGATIVE_CHANNELS, False).rename(
        columns={"channel": "negative_channel", "strength": "negative_strength", "propagated_score": "negative_raw_score"}
    )
    universe = positive.merge(negative.drop(columns="firm_name"), on="permco", how="inner", validate="one_to_one")
    universe["signed_rank_score"] = universe["positive_strength"] - universe["negative_strength"]
    universe["is_any_seed"] = universe["permco"].isin(seed_ids)

    characteristics = pd.read_parquet(expansion / "characteristic_propagated_scores.parquet")
    characteristics["permco"] = pd.to_numeric(characteristics["permco"], errors="coerce").astype("Int64")
    cpos = percentile_strength(characteristics, POSITIVE_CHANNELS, True).rename(
        columns={"channel": "positive_channel", "strength": "positive_strength", "propagated_score": "positive_raw_score"}
    )
    cneg = percentile_strength(characteristics, NEGATIVE_CHANNELS, False).rename(
        columns={"channel": "negative_channel", "strength": "negative_strength", "propagated_score": "negative_raw_score"}
    )
    characteristic_universe = cpos.merge(cneg.drop(columns="firm_name"), on="permco", validate="one_to_one")
    characteristic_universe["signed_rank_score"] = characteristic_universe["positive_strength"] - characteristic_universe["negative_strength"]
    characteristic_universe["is_any_seed"] = characteristic_universe["permco"].isin(seed_ids)
    return universe, characteristic_universe


def select_members(root: Path, count: int = 25) -> pd.DataFrame:
    embedding, characteristics = build_score_universe(root)
    cars = pd.read_parquet(root / "event_data/market_model_cars.parquet")
    controls = pd.read_parquet(root.parent / "remaining_benchmarks/return_comovement/characteristics/2021q4.parquet")
    eligible = set(
        cars.dropna(subset=CAR_COLUMNS + ["beta"])["permco"]
        .loc[lambda x: x.isin(controls.dropna(subset=["beta_60m_paper"])["permco"])]
        .astype(int)
    )
    embedding = embedding.loc[embedding["permco"].isin(eligible)].copy()
    characteristics = characteristics.loc[characteristics["permco"].isin(eligible)].copy()
    seeds = pd.read_parquet(root / "embedding_expansion_v2/directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds["firm_id"], errors="coerce").astype("Int64")
    names = embedding[["permco", "firm_name"]].drop_duplicates()
    direct = seeds.groupby("firm_id", as_index=False)["signed_score"].mean().merge(names, left_on="firm_id", right_on="permco", how="left")
    direct = direct.loc[direct["firm_id"].isin(eligible)].copy()
    direct_long = direct.loc[direct.signed_score > 0].nlargest(count, "signed_score")
    direct_short = direct.loc[direct.signed_score < 0].nsmallest(count, "signed_score")

    def take(source: pd.DataFrame, model: str) -> pd.DataFrame:
        pool = source.loc[~source.is_any_seed].copy()
        long = pool.nlargest(count, "signed_rank_score").assign(side="long")
        short = pool.loc[~pool.permco.isin(long.permco)].nsmallest(count, "signed_rank_score").assign(side="short")
        return pd.concat([long, short], ignore_index=True).assign(portfolio=model, selection_score=lambda x: x.signed_rank_score)

    rows = []
    rows.append(pd.concat([direct_long.assign(side="long"), direct_short.assign(side="short")])
                .assign(portfolio="direct_text", selection_score=lambda x: x.signed_score)
                [["portfolio", "side", "permco", "firm_name", "selection_score"]])
    for source, model in [(embedding, "embedding_expansion"), (characteristics, "characteristic_expansion")]:
        rows.append(take(source, model)[["portfolio", "side", "permco", "firm_name", "selection_score"]])

    half = count // 2
    direct_part = pd.concat([direct_long.nlargest(half, "signed_score").assign(side="long"),
                             direct_short.nsmallest(half, "signed_score").assign(side="short")])
    emb_pool = embedding.loc[~embedding.is_any_seed].copy()
    emb_long = emb_pool.nlargest(count - half, "signed_rank_score").assign(side="long")
    emb_short = emb_pool.loc[~emb_pool.permco.isin(emb_long.permco)].nsmallest(count - half, "signed_rank_score").assign(side="short")
    combined = pd.concat([
        direct_part.assign(selection_score=lambda x: x.signed_score),
        pd.concat([emb_long, emb_short]).assign(selection_score=lambda x: x.signed_rank_score),
    ]).assign(portfolio="combined")
    rows.append(combined[["portfolio", "side", "permco", "firm_name", "selection_score"]])
    return pd.concat(rows, ignore_index=True)


def beta_neutral_weights(frame: pd.DataFrame, cap: float = 0.15) -> np.ndarray:
    side = frame["side"].to_numpy()
    beta = frame["beta"].to_numpy(float)
    target = np.where(side == "long", 1 / (side == "long").sum(), -1 / (side == "short").sum())
    bounds = [(0, cap) if value == "long" else (-cap, 0) for value in side]
    constraints = [
        {"type": "eq", "fun": lambda w: w[side == "long"].sum() - 1},
        {"type": "eq", "fun": lambda w: w[side == "short"].sum() + 1},
        {"type": "eq", "fun": lambda w: float(w @ beta)},
    ]
    result = minimize(lambda w: float(np.square(w - target).sum()), target, method="SLSQP",
                      bounds=bounds, constraints=constraints, options={"maxiter": 2000, "ftol": 1e-12})
    return result.x if result.success else target


def evaluate_portfolios(root: Path, members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cars = pd.read_parquet(root / "event_data/market_model_cars.parquet")
    controls = pd.read_parquet(root.parent / "remaining_benchmarks/return_comovement/characteristics/2021q4.parquet")
    industry = pd.read_parquet(root / "event_data/crsp_point_in_time_industry_2021q4.parquet")
    data = members.merge(cars, on="permco", how="left", validate="many_to_one").merge(
        controls[["permco", "beta_60m_paper", "log_market_equity"]], on="permco", how="left", validate="many_to_one"
    ).merge(industry[["permco", "sic2"]], on="permco", how="left", validate="many_to_one")
    data["beta"] = data["beta_60m_paper"].fillna(data["beta"])
    rows = []
    weight_rows = []
    for portfolio, group in data.groupby("portfolio"):
        group = group.dropna(subset=["beta"] + CAR_COLUMNS).copy()
        equal = np.where(group.side.eq("long"), 1 / group.side.eq("long").sum(), -1 / group.side.eq("short").sum())
        neutral = beta_neutral_weights(group)
        for scheme, weights in [("equal_weight", equal), ("beta_neutral", neutral)]:
            row = {"portfolio": portfolio, "weighting": scheme, "names": len(group),
                   "long_names": int(group.side.eq("long").sum()), "short_names": int(group.side.eq("short").sum()),
                   "net_exposure": float(weights.sum()), "gross_exposure": float(np.abs(weights).sum()),
                   "portfolio_beta": float(weights @ group.beta.to_numpy(float))}
            for outcome in CAR_COLUMNS:
                row[outcome] = float(weights @ group[outcome].to_numpy(float))
            rows.append(row)
            temp = group[["portfolio", "side", "permco", "firm_name", "selection_score", "beta", "sic2"] + CAR_COLUMNS].copy()
            temp["weighting"] = scheme; temp["weight"] = weights
            weight_rows.append(temp)
    return pd.DataFrame(rows), pd.concat(weight_rows, ignore_index=True)


def contributor_audit(root: Path, members: pd.DataFrame, neighbors: int = 3) -> pd.DataFrame:
    embeddings = pd.read_parquet(root.parent / "active_passive_extension/benchmarks/all_s12_control/2021q4/rs_l_min/asset_embeddings_10d.parquet")
    columns = [c for c in embeddings if c.startswith("embedding_")]
    matrix = embeddings[columns].to_numpy(float)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    lookup = {int(p): i for i, p in enumerate(embeddings.permco)}
    seeds = pd.read_parquet(root / "embedding_expansion_v2/directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds.firm_id, errors="coerce").astype("Int64")
    seed_names = embeddings[["permco", "firm_name"]].rename(columns={"permco": "firm_id", "firm_name": "seed_name"})
    seeds = seeds.merge(seed_names, on="firm_id", how="left")
    results = []
    candidates = members.loc[members.portfolio.eq("embedding_expansion")]
    for row in candidates.itertuples():
        eligible = seeds.loc[seeds.signed_score.gt(0) if row.side == "long" else seeds.signed_score.lt(0)].copy()
        if int(row.permco) not in lookup: continue
        eligible = eligible.loc[eligible.firm_id.map(lambda x: int(x) in lookup if pd.notna(x) else False)].copy()
        eligible["cosine_similarity"] = [float(matrix[lookup[int(row.permco)]] @ matrix[lookup[int(x)]]) for x in eligible.firm_id]
        for rank, seed in enumerate(eligible.nlargest(neighbors, "cosine_similarity").itertuples(), 1):
            results.append({"candidate_permco": row.permco, "candidate_name": row.firm_name, "side": row.side,
                            "neighbor_rank": rank, "seed_permco": seed.firm_id, "seed_name": seed.seed_name,
                            "seed_channel": seed.channel, "seed_signed_score": seed.signed_score,
                            "cosine_similarity": seed.cosine_similarity})
    return pd.DataFrame(results)
