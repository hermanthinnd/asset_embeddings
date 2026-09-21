import json
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'14A_Critical_Minerals_Factor_Construction.ipynb'
nb=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
cells=[]
def md(x):cells.append(nbf.v4.new_markdown_cell(x))
def code(x):cells.append(nbf.v4.new_code_cell(x))

md("""# 14A — Independent critical-minerals theme factor

## Objective

Construct and freeze an external daily state variable for the critical-minerals investment theme. This separates three questions:

1. Does a firm have predicted thematic exposure?
2. Does its stock subsequently move with the external theme factor?
3. Does it earn abnormal returns?

This notebook addresses only the factor. It does not estimate firm betas, inspect portfolio performance, or modify the frozen text and embedding scores.

The primary factor is an investable equity-theme factor—not a pure mineral-price shock. It captures the joint market valuation of copper miners, the lithium/battery chain, and rare-earth/strategic-metals firms after subtracting the broad US equity market.""")

md("""## 1. Setup and immutable inputs""")
code("""from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name!='sp500_critical_minerals':
    candidate=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
    if candidate.exists(): ROOT=candidate
OUT=ROOT/'thematic_factor_v1'; RAW=OUT/'raw'; MANIFESTS=OUT/'manifests'
for p in [OUT,RAW,MANIFESTS]:p.mkdir(exist_ok=True)

SIGNALS=ROOT/'long_short_hybrid_v1'/'signed_signal_panel.parquet'
assert SIGNALS.exists()
signal_hash=hashlib.sha256(SIGNALS.read_bytes()).hexdigest()
signals=pd.read_parquet(SIGNALS)
display(signals.groupby('vintage').agg(firms=('permco','nunique')).reset_index())
print({'signal_sha256':signal_hash,'returns_opened':False})""")

md("""## 2. Freeze the factor contract before acquisition

### Primary factor

For trading day \(d\):

\[
CM_d=\frac{1}{3}(R^{COPX}_d+R^{LIT}_d+R^{REMX}_d)-R^{SPY}_d.
\]

- `COPX`: copper-miner equity sleeve.
- `LIT`: lithium and battery-technology value-chain sleeve.
- `REMX`: rare-earth and strategic-metals equity sleeve.
- `SPY`: broad-market benchmark.

Each sleeve receives equal weight. Missing sleeves are not silently reweighted: all four returns must exist.

### Frozen robustness factors

- `producer_ex_market`: equal-weight COPX and REMX minus SPY.
- `base_metals_raw`: raw DBB total return; the beta regression controls for the equity market separately.
- `materials_ex_market`: XME minus SPY.

The primary result must remain the three-sleeve factor. Robustness definitions cannot replace it based on results.""")
code("""CONTRACT={
 'version':'critical_minerals_thematic_factor_v1',
 'frozen_before_download':True,
 'primary_name':'broad_cm_equity_ex_market',
 'primary_tickers':['COPX','LIT','REMX'],
 'market_ticker':'SPY',
 'primary_formula':'equal_weight(COPX,LIT,REMX)-SPY',
 'robustness':{
   'producer_ex_market':{'tickers':['COPX','REMX'],'formula':'equal_weight(COPX,REMX)-SPY'},
   'base_metals_raw':{'tickers':['DBB'],'formula':'DBB'},
   'materials_ex_market':{'tickers':['XME'],'formula':'XME-SPY'}},
 'frequency':'daily_close_to_close_total_return',
 'prices':'split_and_distribution_adjusted',
 'start':'2022-01-01','end_exclusive':'2026-01-01','common_data_cutoff':'2025-12-31',
 'missing_policy':'complete_case_no_dynamic_reweighting',
 'winsorization':'none',
 'standardization':'none_for_regressions',
 'signal_inputs_accessed':False,
 'firm_returns_accessed':False,
 'portfolio_results_accessed':False,
 'interpretation':'investable equity-theme factor; not a pure commodity-price shock',
}
contract_path=MANIFESTS/'thematic_factor_contract.json'
if contract_path.exists():
    old=json.loads(contract_path.read_text())
    assert old==CONTRACT,'A different factor contract already exists. Version changes explicitly; do not overwrite.'
else:
    contract_path.write_text(json.dumps(CONTRACT,indent=2,sort_keys=True)+'\\n')
contract_hash=hashlib.sha256(contract_path.read_bytes()).hexdigest()
print({'contract_sha256':contract_hash,'frozen_before_download':True})""")

md("""## 3. Acquire and checkpoint adjusted prices

This uses Yahoo Finance only as a reproducible pilot data source. It makes no OpenAI calls. If the checkpoint exists, the notebook performs no network request.""")
code("""price_path=RAW/'theme_etf_adjusted_close.parquet'
tickers=sorted(set(CONTRACT['primary_tickers']+[CONTRACT['market_ticker'],'DBB','XME']))
if price_path.exists():
    prices=pd.read_parquet(price_path)
    prices.index=pd.to_datetime(prices.index)
    download_status='checkpoint_loaded'
else:
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError('Install yfinance in this notebook kernel with: %pip install yfinance') from error
    downloaded=yf.download(tickers,start=CONTRACT['start'],end=CONTRACT['end_exclusive'],
                           auto_adjust=False,progress=True,threads=True)
    if isinstance(downloaded.columns,pd.MultiIndex):
        prices=downloaded['Adj Close'].copy()
    else:
        prices=downloaded[['Adj Close']].rename(columns={'Adj Close':tickers[0]})
    prices=prices.sort_index().reindex(columns=tickers)
    prices.index.name='date'
    prices.to_parquet(price_path)
    download_status='downloaded'
display(pd.DataFrame([{'status':download_status,'trading_days':len(prices),'first_date':prices.index.min(),
                       'last_date':prices.index.max(),'tickers':','.join(prices.columns)}]))""")

md("""## 4. Data-quality gates""")
code("""coverage=prices.notna().agg(['count','mean']).T.reset_index().rename(columns={'index':'ticker','count':'observations','mean':'coverage'})
returns=prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)
gates=pd.DataFrame([
 {'gate':'all_required_tickers_present','value':len(set(tickers)&set(prices.columns)),'threshold':len(tickers),'passed':set(tickers).issubset(prices.columns)},
 {'gate':'primary_complete_days_at_least_500','value':int(returns[CONTRACT['primary_tickers']+['SPY']].notna().all(axis=1).sum()),'threshold':500,'passed':returns[CONTRACT['primary_tickers']+['SPY']].notna().all(axis=1).sum()>=500},
 {'gate':'covers_all_2024','value':int(prices.index.min()<=pd.Timestamp('2024-01-02') and prices.index.max()>=pd.Timestamp('2024-12-31')),'threshold':1,'passed':prices.index.min()<=pd.Timestamp('2024-01-02') and prices.index.max()>=pd.Timestamp('2024-12-31')},
])
display(coverage);display(gates)
assert gates.passed.all(),'Factor acquisition failed a frozen data-quality gate.'""")

md("""## 5. Construct the primary and robustness factors""")
code("""factor=pd.DataFrame(index=returns.index)
factor['broad_cm_equity_return']=returns[['COPX','LIT','REMX']].mean(axis=1,skipna=False)
factor['market_return']=returns['SPY']
factor['broad_cm_equity_ex_market']=factor.broad_cm_equity_return-factor.market_return
factor['producer_ex_market']=returns[['COPX','REMX']].mean(axis=1,skipna=False)-returns.SPY
factor['base_metals_raw']=returns.DBB
factor['materials_ex_market']=returns.XME-returns.SPY
factor=factor.dropna(subset=['broad_cm_equity_ex_market']).reset_index()
factor.to_parquet(OUT/'daily_critical_minerals_factors.parquet',index=False)

summary=factor.drop(columns='date').agg(['count','mean','std','min','max']).T
summary['annualized_mean']=summary['mean']*252
summary['annualized_volatility']=summary['std']*np.sqrt(252)
display(summary)
display(factor.drop(columns='date').corr().round(3))""")

md("""## 6. Freeze quarterly estimation windows

Each 2024 quarter-end signal receives:

- a trailing 252-trading-day historical window ending no later than the signal cutoff;
- a forward 252-trading-day window beginning strictly after the cutoff.

The next notebook will estimate betas using daily CRSP firm returns. Overlapping forward windows are reported by vintage and are not treated as independent experiments.""")
code("""cutoffs={'2024q1':'2024-03-31','2024q2':'2024-06-30','2024q3':'2024-09-30','2024q4':'2024-12-31'}
dates=pd.DatetimeIndex(factor.date.sort_values().unique());rows=[]
for vintage,cutoff in cutoffs.items():
    cutoff=pd.Timestamp(cutoff)
    hist=dates[dates<=cutoff][-252:]
    fwd=dates[dates>cutoff][:252]
    rows.append({'vintage':vintage,'signal_cutoff':cutoff,'historical_start':hist.min() if len(hist) else pd.NaT,
                 'historical_end':hist.max() if len(hist) else pd.NaT,'historical_days':len(hist),
                 'forward_start':fwd.min() if len(fwd) else pd.NaT,'forward_end':fwd.max() if len(fwd) else pd.NaT,
                 'forward_days_available':len(fwd),'forward_target_days':252})
windows=pd.DataFrame(rows)
windows.to_parquet(MANIFESTS/'frozen_beta_windows.parquet',index=False)
display(windows)""")

md("""## 7. Final manifest and next-stage handoff""")
code("""artifacts=[contract_path,price_path,OUT/'daily_critical_minerals_factors.parquet',MANIFESTS/'frozen_beta_windows.parquet']
inventory=pd.DataFrame([{'artifact':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts])
inventory.to_parquet(MANIFESTS/'artifact_inventory.parquet',index=False)
manifest={'stage':'independent_critical_minerals_factor_v1','primary_factor':CONTRACT['primary_name'],
          'contract_sha256':contract_hash,'signal_sha256':signal_hash,'observations':len(factor),
          'first_date':str(factor.date.min().date()),'last_date':str(factor.date.max().date()),
          'firm_returns_accessed':False,'portfolio_results_accessed':False}
(MANIFESTS/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n')
display(inventory)
print('Saved factor stage:',OUT)
print('Next: acquire matched daily CRSP firm returns and estimate historical/forward thematic betas.')""")

md("""## Interpretation boundary

This factor is suitable for testing whether predicted exposure corresponds to realized sensitivity to the traded critical-minerals theme. Because its components are equity ETFs, it may contain industry, duration, currency, and equity-style variation. The next stage must include the broad market and conventional controls, and report narrower robustness factors. A later pure commodity-price factor would strengthen the design if reliable point-in-time mineral-price series become available.""")

nb['cells']=cells
nbf.write(nb,OUT)
print(OUT)
