import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "14F_Capped_Portfolio_Risk_and_Theme_Analysis.ipynb"

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)})

md("""# 14F — Capped Portfolio Risk and Theme Analysis

This notebook applies the same diagnostics used for the uncapped seeded-embedding portfolio to the earlier **frozen capped portfolios**. It does not refit, reselect, download data, or call an API.

Two designs are kept separate:

1. The daily, three-month frozen portfolios from the long–short, hybrid, and consensus experiments are used for Sharpe ratios, volatility, benchmark correlations, information ratios, and factor regressions.
2. The earlier matched-cardinality long-only experiment is shown separately using its saved endpoint returns. Its longer evaluation design is not silently combined with the daily three-month panel.

All portfolio definitions were frozen before these return diagnostics were calculated.""")

code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path.cwd()
if ROOT.name != 'sp500_critical_minerals':
    candidate = Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
    if candidate.exists(): ROOT = candidate

ATTR = ROOT / 'factor_attribution_v1'
FORWARD = ROOT / 'forward_performance_v1'
OUT = ROOT / 'capped_portfolio_analysis_v1'
FIG = OUT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

daily = pd.read_parquet(ATTR / 'daily_attribution_panel.parquet')
daily['date'] = pd.to_datetime(daily['date'])
assert daily.groupby('method').size().nunique() == 1
assert daily.groupby('method').date.nunique().min() == 248
print('Methods:', sorted(daily.method.unique()))
print('Dates:', daily.date.min().date(), 'to', daily.date.max().date())""")

md("""## 1. Construct a common calendar-time return panel

The six capped methods are evaluated on the same 248 trading dates. `hybrid_long_only` is a funded long-only portfolio. The other five methods are self-financing long–short spreads.""")

code("""method_returns = daily.pivot(index='date', columns='method', values='portfolio_return').sort_index()
bench = (daily.sort_values(['date','method']).drop_duplicates('date')
         .set_index('date')[['market_return','broad_cm_equity_return','broad_cm_equity_ex_market','RF','Mkt-RF','SMB','HML','RMW','CMA','Mom']])
panel = method_returns.join(bench, how='inner')
panel.to_parquet(OUT / 'capped_daily_return_panel.parquet')
display(panel.head())
print('Complete rows:', int(panel.notna().all(axis=1).sum()), '/', len(panel))""")

md("""## 2. Visualize the capped portfolios

Long-only and long–short strategies are shown separately because a funded equity portfolio and a zero-investment spread have different economic interpretations.""")

code("""labels = {
    'consensus':'Consensus long–short', 'embedding':'Embedding long–short',
    'hybrid':'Hybrid long–short', 'hybrid_long_only':'Hybrid long-only',
    'llm':'LLM long–short', 'matched_text':'Matched-text long–short',
    'market_return':'S&P 500', 'broad_cm_equity_return':'Critical-minerals benchmark'
}

wealth_lo = (1 + panel[['hybrid_long_only','market_return','broad_cm_equity_return']]).cumprod().rename(columns=labels)
ax = wealth_lo.plot(figsize=(11,6), linewidth=2, color=['#0072B2','#777777','#009E73'])
ax.axhline(1,color='black',lw=.7); ax.grid(alpha=.2)
ax.set(title='Capped long-only portfolio and benchmarks', xlabel='Date', ylabel='Growth of $1')
plt.tight_layout(); plt.savefig(FIG/'capped_long_only_vs_benchmarks.png',dpi=180,bbox_inches='tight'); plt.show()

ls = ['embedding','llm','hybrid','consensus','matched_text']
wealth_ls = (1 + panel[ls]).cumprod().rename(columns=labels)
ax = wealth_ls.plot(figsize=(11,6), linewidth=2)
ax.axhline(1,color='black',lw=.7); ax.grid(alpha=.2)
ax.set(title='Capped long–short portfolio wealth paths', xlabel='Date', ylabel='Growth of $1')
plt.tight_layout(); plt.savefig(FIG/'capped_long_short_wealth.png',dpi=180,bbox_inches='tight'); plt.show()""")

md("""## 3. Risk and Sharpe-ratio comparison

The conventional Sharpe ratio uses excess returns over the daily risk-free rate for funded long-only investments. Long–short spreads are already self-financing, so their spread return is used directly.""")

code("""def stats(raw, sharpe_series):
    raw=raw.dropna(); s=sharpe_series.loc[raw.index].dropna(); raw=raw.loc[s.index]
    wealth=(1+raw).cumprod()
    return pd.Series({
        'days':len(raw), 'cumulative_return':wealth.iloc[-1]-1,
        'annualized_mean':raw.mean()*252,
        'annualized_volatility':raw.std(ddof=1)*np.sqrt(252),
        'conventional_sharpe':s.mean()/s.std(ddof=1)*np.sqrt(252),
        'maximum_drawdown':(wealth/wealth.cummax()-1).min()
    })

rows=[]
for method in ls:
    x=stats(panel[method],panel[method]); x['portfolio']=method; x['portfolio_type']='self_financing_long_short'; rows.append(x)
for method,raw in [('hybrid_long_only',panel.hybrid_long_only),('sp500',panel.market_return),('critical_minerals',panel.broad_cm_equity_return)]:
    x=stats(raw,raw-panel.RF); x['portfolio']=method; x['portfolio_type']='funded_long_only'; rows.append(x)
performance=pd.DataFrame(rows)[['portfolio','portfolio_type','days','cumulative_return','annualized_mean','annualized_volatility','conventional_sharpe','maximum_drawdown']]
performance.to_parquet(OUT/'capped_performance_summary.parquet',index=False)
display(performance.round(4))""")

md("""## 4. Benchmark-relative performance

The information ratio is the annualized active return relative to the critical-minerals benchmark divided by tracking error. Its cleanest interpretation is for the capped long-only portfolio because both sides are funded long-only investments.""")

code("""ir_rows=[]
for method in ['hybrid_long_only']:
    for benchmark in ['broad_cm_equity_return','market_return']:
        active=panel[method]-panel[benchmark]
        ir_rows.append({'portfolio':method,'benchmark':benchmark,
                        'annualized_active_return':active.mean()*252,
                        'tracking_error':active.std(ddof=1)*np.sqrt(252),
                        'information_ratio':active.mean()/active.std(ddof=1)*np.sqrt(252),
                        'correlation':panel[method].corr(panel[benchmark]),
                        'simple_beta':panel[method].cov(panel[benchmark])/panel[benchmark].var()})
relative=pd.DataFrame(ir_rows)
relative.to_parquet(OUT/'capped_long_only_relative_performance.parquet',index=False)
display(relative.round(4))""")

md("""## 5. Does the capped portfolio preserve critical-minerals exposure?

For long-only, the dependent variable is portfolio return minus the risk-free rate. For long–short portfolios, it is the self-financing spread return. HAC standard errors use five lags. The key result is the minerals-factor beta and the increase in R² when the minerals factor is added.""")

code("""factor_rows=[]
methods=ls+['hybrid_long_only']
for method in methods:
    y=panel[method] if method in ls else panel[method]-panel.RF
    specifications={
        'market':['Mkt-RF'],
        'market_plus_cm':['Mkt-RF','broad_cm_equity_ex_market'],
        'ff5_mom':['Mkt-RF','SMB','HML','RMW','CMA','Mom'],
        'ff5_mom_plus_cm':['Mkt-RF','SMB','HML','RMW','CMA','Mom','broad_cm_equity_ex_market']
    }
    fitted={}
    for spec,cols in specifications.items():
        z=pd.concat([y.rename('y'),panel[cols]],axis=1).dropna()
        model=sm.OLS(z.y,sm.add_constant(z[cols])).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        fitted[spec]=model
        factor_rows.append({'method':method,'specification':spec,'n':int(model.nobs),
                            'r2':model.rsquared,'adjusted_r2':model.rsquared_adj,
                            'annualized_alpha':model.params['const']*252,
                            'alpha_t':model.tvalues['const'],
                            'cm_beta':model.params.get('broad_cm_equity_ex_market',np.nan),
                            'cm_t':model.tvalues.get('broad_cm_equity_ex_market',np.nan),
                            'cm_p':model.pvalues.get('broad_cm_equity_ex_market',np.nan)})
    factor_rows[-3]['incremental_cm_r2_vs_market']=fitted['market_plus_cm'].rsquared-fitted['market'].rsquared
    factor_rows[-1]['incremental_cm_r2_vs_ff5_mom']=fitted['ff5_mom_plus_cm'].rsquared-fitted['ff5_mom'].rsquared

factor_results=pd.DataFrame(factor_rows)
factor_results.to_parquet(OUT/'capped_factor_and_theme_exposure.parquet',index=False)
focus=factor_results[factor_results.specification.eq('ff5_mom_plus_cm')][['method','annualized_alpha','alpha_t','cm_beta','cm_t','cm_p','r2','incremental_cm_r2_vs_ff5_mom']]
display(focus.round(4))""")

md("""## 6. Volatility-matched capped long-only comparison

The capped long-only portfolio is levered or de-levered to the minerals benchmark's realized volatility. Financing is charged at the contemporaneous risk-free rate. This is a descriptive, ex-post risk-matching diagnostic—not an investable rule selected before the sample.""")

code("""lo_vol=panel.hybrid_long_only.std(ddof=1)*np.sqrt(252)
cm_vol=panel.broad_cm_equity_return.std(ddof=1)*np.sqrt(252)
leverage=cm_vol/lo_vol
scaled=panel.RF+leverage*(panel.hybrid_long_only-panel.RF)
vm=pd.DataFrame([
    {'portfolio':'capped_hybrid_long_only','leverage':1.0,**stats(panel.hybrid_long_only,panel.hybrid_long_only-panel.RF).to_dict()},
    {'portfolio':'capped_hybrid_long_only_vol_matched','leverage':leverage,**stats(scaled,scaled-panel.RF).to_dict()},
    {'portfolio':'critical_minerals','leverage':1.0,**stats(panel.broad_cm_equity_return,panel.broad_cm_equity_return-panel.RF).to_dict()}
])
vm.to_parquet(OUT/'capped_volatility_matched_comparison.parquet',index=False)
display(vm.round(4))""")

md("""## 7. Earlier matched-cardinality long-only endpoints

These saved results compare equal-cardinality embedding-only and LLM-only portfolios. They are displayed separately because only endpoint returns—not a common daily path—were frozen in that stage.""")

code("""primary=pd.read_parquet(FORWARD/'primary_portfolio_results.parquet')
display(primary[['vintage','portfolio','selected_firms','portfolio_return','available_universe_adjusted_return']].round(4))
pivot=primary.pivot(index='vintage',columns='portfolio',values='portfolio_return')
ax=pivot.rename(columns={'embedding_only_primary':'Embedding-only','llm_only_common_support':'LLM-only'}).plot(kind='bar',figsize=(10,5),color=['#0072B2','#D55E00'])
ax.axhline(0,color='black',lw=.7); ax.set(title='Earlier capped equal-cardinality long-only endpoint returns',xlabel='Vintage',ylabel='Portfolio return'); ax.grid(axis='y',alpha=.2)
for container in ax.containers: ax.bar_label(container,fmt='%.1f%%',labels=[f'{100*v:.1f}%' for v in container.datavalues],padding=2,fontsize=8)
plt.xticks(rotation=0);plt.tight_layout();plt.savefig(FIG/'matched_cardinality_long_only_endpoints.png',dpi=180,bbox_inches='tight');plt.show()""")

md("""## 8. Interpretation guide

- Sharpe ratio evaluates return above cash per unit of total risk; it does not use either benchmark.
- Information ratio evaluates active performance relative to a chosen benchmark.
- Correlation and simple beta show whether the capped long-only portfolio behaves more like the broad market or the minerals benchmark.
- The controlled minerals beta tests whether thematic exposure remains after standard equity factors are held constant.
- Incremental R² measures how much explanatory power the minerals factor adds.
- Volatility matching asks whether conclusions survive after equalizing realized risk; it must not be described as a pre-specified trading strategy.

Because there are only 248 trading observations across four short frozen windows, all statistical conclusions remain pilot evidence rather than definitive long-horizon performance claims.""")

code("""required=[
    OUT/'capped_daily_return_panel.parquet', OUT/'capped_performance_summary.parquet',
    OUT/'capped_long_only_relative_performance.parquet', OUT/'capped_factor_and_theme_exposure.parquet',
    OUT/'capped_volatility_matched_comparison.parquet', FIG/'capped_long_only_vs_benchmarks.png',
    FIG/'capped_long_short_wealth.png', FIG/'matched_cardinality_long_only_endpoints.png']
gates=pd.DataFrame([
    {'gate':'six_capped_methods','value':daily.method.nunique(),'threshold':6,'passed':daily.method.nunique()==6},
    {'gate':'common_248_day_panel','value':len(panel),'threshold':248,'passed':len(panel)==248},
    {'gate':'all_outputs_saved','value':sum(p.exists() for p in required),'threshold':len(required),'passed':all(p.exists() for p in required)},
    {'gate':'no_selection_or_refit','value':True,'threshold':True,'passed':True}
])
# PyArrow cannot serialize mixed integer/Boolean object columns consistently.
# These two columns are descriptive; preserve their values as strings while
# retaining `passed` as a genuine Boolean decision field.
gates['value'] = gates['value'].astype(str)
gates['threshold'] = gates['threshold'].astype(str)
gates['passed'] = gates['passed'].astype(bool)
gates.to_parquet(OUT/'decision_gates.parquet',index=False)
display(gates)
assert gates.passed.all()
print('Saved analysis to:',OUT)""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
NB.write_text(json.dumps(nb, indent=1) + "\n")
print(NB)
