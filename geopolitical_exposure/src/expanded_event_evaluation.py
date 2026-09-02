"""Evaluation helpers for the frozen expanded geopolitical-exposure scores."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests


def evaluate_model_scores(evaluation, firm_controls, control_columns, car_columns):
    rows = []
    for (representation, channel), group in evaluation.groupby(["representation", "channel"], sort=True):
        frame = (group.drop(columns=["sic2", "siccd"], errors="ignore")
                 .merge(firm_controls[["permco", "sic2"] + control_columns],
                        on="permco", how="left", validate="many_to_one"))
        for column in control_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].fillna(frame[column].median())
        dummies = pd.get_dummies(frame.sic2.astype("string").fillna("missing"),
                                 prefix="sic2", drop_first=True, dtype=float)
        base_x = pd.concat([frame[control_columns].reset_index(drop=True),
                            dummies.reset_index(drop=True)], axis=1)
        base_x = sm.add_constant(base_x.astype(float), has_constant="add")
        score = pd.to_numeric(frame.propagated_score, errors="coerce").astype("float64").reset_index(drop=True)
        score_sd = score.std(ddof=0)
        score_z = (score - score.mean()) / score_sd if score_sd > 0 else score * 0
        for outcome in car_columns:
            y = pd.to_numeric(frame[outcome], errors="coerce").astype("float64").reset_index(drop=True)
            valid = y.notna() & score_z.notna()
            rho, rho_p = (spearmanr(score[valid], y[valid])
                          if score[valid].nunique() > 1 else (np.nan, np.nan))
            ranks = score[valid].rank(pct=True, method="average")
            low, high = y[valid][ranks <= .2], y[valid][ranks >= .8]
            spread = high.mean() - low.mean()
            spread_p = (ttest_ind(high, low, equal_var=False, nan_policy="omit").pvalue
                        if len(low) and len(high) else np.nan)
            base = sm.OLS(y[valid], base_x.loc[valid]).fit(cov_type="HC3")
            augmented_x = base_x.loc[valid].copy()
            augmented_x["exposure_score_z"] = score_z[valid]
            augmented = sm.OLS(y[valid], augmented_x).fit(cov_type="HC3")
            rows.append({
                "representation": representation, "channel": channel, "outcome": outcome,
                "firms": int(valid.sum()), "spearman_rho": rho, "spearman_pvalue": rho_p,
                "top_minus_bottom_car": spread, "spread_pvalue": spread_p,
                "score_coefficient": augmented.params["exposure_score_z"],
                "score_hc3_pvalue": augmented.pvalues["exposure_score_z"],
                "base_adjusted_r2": base.rsquared_adj,
                "model_adjusted_r2": augmented.rsquared_adj,
                "incremental_adjusted_r2": augmented.rsquared_adj - base.rsquared_adj,
            })
    result = pd.DataFrame(rows)
    for pvalue, qvalue in [("spearman_pvalue", "spearman_bh_qvalue"),
                           ("score_hc3_pvalue", "score_hc3_bh_qvalue"),
                           ("spread_pvalue", "spread_bh_qvalue")]:
        result[qvalue] = np.nan
        valid = result[pvalue].notna()
        if valid.any():
            result.loc[valid, qvalue] = multipletests(result.loc[valid, pvalue], method="fdr_bh")[1]
    return result


def direct_seed_reactions(seeds, cars, car_columns):
    panel = seeds.merge(cars, left_on="firm_id", right_on="permco", how="left", validate="many_to_one")
    rows = []
    for channel, group in panel.groupby("channel", sort=True):
        for outcome in car_columns:
            valid = group[["signed_score", outcome]].dropna()
            rho, pvalue = (spearmanr(valid.signed_score, valid[outcome])
                           if len(valid) >= 3 and valid.signed_score.nunique() > 1 else (np.nan, np.nan))
            rows.append({"channel": channel, "outcome": outcome, "seed_firms": len(valid),
                         "spearman_rho": rho, "spearman_pvalue": pvalue})
    return panel, pd.DataFrame(rows)


def evaluate_placebos(placebos, cars, car_columns, actual_results):
    merged = placebos.merge(cars[["permco"] + car_columns], on="permco", how="inner")
    rows = []
    for (repetition, channel), group in merged.groupby(["repetition", "channel"], sort=True):
        for outcome in car_columns:
            valid = group[["propagated_score", outcome]].dropna()
            rho = (spearmanr(valid.propagated_score, valid[outcome]).statistic
                   if valid.propagated_score.nunique() > 1 else np.nan)
            rows.append({"repetition": repetition, "channel": channel,
                         "outcome": outcome, "spearman_rho": rho})
    correlations = pd.DataFrame(rows)
    actual = actual_results[actual_results.representation.eq("all_s12")][
        ["channel", "outcome", "spearman_rho"]].rename(columns={"spearman_rho": "actual_spearman_rho"})
    summary = actual.merge(
        correlations.groupby(["channel", "outcome"]).spearman_rho
        .agg(placebo_mean="mean", placebo_std="std").reset_index(),
        on=["channel", "outcome"], how="left")
    for index, row in summary.iterrows():
        null = correlations.query("channel == @row.channel and outcome == @row.outcome").spearman_rho.dropna().abs()
        summary.loc[index, "empirical_two_sided_pvalue"] = (
            (1 + (null >= abs(row.actual_spearman_rho)).sum()) / (1 + len(null)) if len(null) else np.nan
        )
    return correlations, summary
