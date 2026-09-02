"""Build static, presentation-ready figures for the V2 geopolitical extension summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "summary_assets_v2"
ASSETS.mkdir(parents=True, exist_ok=True)
BLUE = "#2463A7"
LIGHT_BLUE = "#72A6D8"
ORANGE = "#E68635"
GREEN = "#4E9A71"
RED = "#C74B50"
GRAY = "#6B7280"


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def labels(ax, fmt="{:.0f}", padding=3):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding, fontsize=9)


plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Corpus funnel
stages = ["Selected\nfirms", "Firms with\nfilings", "Queued\nfirms", "Directly labeled\nfirms", "Directional\nseed firms"]
values = [300, 298, 163, 90, 89]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(stages, values, color=[BLUE, BLUE, LIGHT_BLUE, ORANGE, GREEN])
labels(ax)
ax.set_ylabel("Firms")
ax.set_title("From frozen firm universe to directional text seeds")
ax.grid(axis="y", alpha=.2)
finish(fig, "01_corpus_funnel.png")

# Accepted exposure channels
panel = pd.read_parquet(ROOT / "text_expansion_v2/results/expanded_exposure_panel_final.parquet")
counts = panel.groupby("channel").size().sort_values()
pretty = {
    "russia_ukraine_operations_revenue": "Russia/Ukraine operations",
    "defense_demand": "Defense demand",
    "sanctions_sensitive_supply_chain": "Sanctions-sensitive supply chain",
    "shipping_logistics": "Shipping/logistics",
    "oil_gas_production": "Oil and gas",
    "agriculture_fertilizer": "Agriculture/fertilizer",
}
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([pretty.get(x, x) for x in counts.index], counts.values, color=BLUE)
for y, value in enumerate(counts.values):
    ax.text(value + .6, y, f"{value}", va="center", fontsize=9)
ax.set_xlabel("Accepted document-level exposures")
ax.set_title("Audited text exposures by scenario channel")
ax.set_xlim(0, counts.max() * 1.15)
ax.grid(axis="x", alpha=.2)
finish(fig, "02_channel_counts.png")

# Best outcome-free LOO representation by channel
loo = pd.read_parquet(ROOT / "embedding_expansion_v2/all_leave_one_out_summary.parquet")
best = loo.sort_values("spearman_correlation").groupby("channel", as_index=False).tail(1).sort_values("spearman_correlation")
fig, ax = plt.subplots(figsize=(10, 5))
colors = [GREEN if x > 0 else RED for x in best.spearman_correlation]
bars = ax.barh([pretty.get(x, x) for x in best.channel], best.spearman_correlation, color=colors)
for bar, (_, row) in zip(bars, best.iterrows()):
    value = row.spearman_correlation
    ax.text(value + (.015 if value >= 0 else -.015), bar.get_y() + bar.get_height()/2,
            f"{value:.3f}  ({row.representation})", va="center",
            ha="left" if value >= 0 else "right", fontsize=9)
ax.axvline(0, color="black", linewidth=.8)
ax.set_xlabel("Leave-one-seed-out Spearman correlation")
ax.set_title("Best internal recovery model differs by exposure channel")
ax.grid(axis="x", alpha=.2)
finish(fig, "03_loo_best_by_channel.png")

# All-S12 actual versus placebo means
placebo = pd.read_parquet(ROOT / "event_study_v2/all_s12_placebo_summary.parquet")
windows = ["car_0_1", "car_0_5", "car_1_20"]
channels = list(pretty)[:5]
plot = placebo[placebo.channel.isin(channels)].copy()
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, outcome in zip(axes, windows):
    part = plot[plot.outcome.eq(outcome)].set_index("channel").reindex(channels)
    y = np.arange(len(part)); width = .36
    ax.barh(y - width/2, part.actual_spearman_rho, height=width, color=BLUE, label="Actual seeds")
    ax.barh(y + width/2, part.placebo_mean, height=width, color=GRAY, label="Placebo mean")
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_title(outcome.replace("car_", "CAR [").replace("_", ",") + "]")
    ax.set_yticks(y, [pretty.get(x, x) for x in channels])
    ax.grid(axis="x", alpha=.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle("All-S12 event correlations do not separate cleanly from shuffled-seed placebos", fontsize=14)
finish(fig, "04_event_actual_vs_placebo.png")

# V1-V2 all-S12 event comparison
comparison = pd.read_parquet(ROOT / "event_study_v2/v1_v2_event_comparison.parquet")
shared = ["defense_demand", "oil_gas_production", "russia_ukraine_operations_revenue"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, outcome in zip(axes, windows):
    part = comparison[comparison.outcome.eq(outcome)].set_index("channel").reindex(shared)
    x = np.arange(len(part)); width = .36
    ax.bar(x - width/2, part.spearman_rho_v1, width, color=LIGHT_BLUE, label="V1: 50-firm pilot")
    ax.bar(x + width/2, part.spearman_rho_v2, width, color=ORANGE, label="V2: expanded labels")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(x, ["Defense", "Oil & gas", "Russia/\nUkraine"])
    ax.set_title(outcome.replace("car_", "CAR [").replace("_", ",") + "]")
    ax.grid(axis="y", alpha=.2)
    labels(ax, "%.3f", 2)
axes[0].set_ylabel("All-S12 Spearman correlation")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Several original event associations are unstable after label expansion", fontsize=14)
finish(fig, "05_v1_v2_event_comparison.png")

# Joint-screen overview
decision = pd.read_parquet(ROOT / "event_study_v2/model_decision_table.parquet")
summary = pd.Series({
    "Specifications": len(decision),
    "Nominal rank p < 0.05": int(decision.nominal_rank_signal.sum()),
    "FDR rank q < 0.10": int(decision.fdr_rank_signal.sum()),
    "FDR controlled coefficient": int(decision.fdr_incremental_signal.sum()),
    "Passed joint screen": int(decision.passes_joint_screen.sum()),
})
fig, ax = plt.subplots(figsize=(10, 4.8))
colors = [GRAY, LIGHT_BLUE, BLUE, ORANGE, RED]
ax.bar(summary.index, summary.values, color=colors)
labels(ax)
ax.set_ylabel("Number of specifications")
ax.set_title("The complete 105-specification grid yields no joint-screen success")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=.2)
finish(fig, "06_decision_funnel.png")

print(f"Saved summary figures to {ASSETS}")
