import json
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'14B_Firm_Thematic_Beta_Estimation.ipynb'
nb=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
cells=[]
def md(x):cells.append(nbf.v4.new_markdown_cell(x))
def code(x):cells.append(nbf.v4.new_code_cell(x))

md("""# 14B — Firm-level critical-minerals thematic beta

## Primary question

Do frozen text and holdings-embedding exposure scores predict firms' realized sensitivity to an independently constructed critical-minerals equity-theme factor?

For firm \(i\), vintage \(t\), and daily return \(d\):

\[
R_{i,d}=\alpha_{i,t}+\beta^M_{i,t}R^{SPY}_d+\beta^{CM}_{i,t}CM_d+\epsilon_{i,d}.
\]

The economically relevant outcome is \(\beta^{CM}\), not unconditional return. Historical and forward betas are kept separate. This notebook does not load the earlier portfolio-performance results.""")

md("""## 1. Setup and frozen design""")
code("""from pathlib import Path
import getpass,hashlib,json,time
import numpy as np
import pandas as pd
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name!='sp500_critical_minerals':
    candidate=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
    if candidate.exists():ROOT=candidate
FACTOR=ROOT/'thematic_factor_v1'; OUT=ROOT/'thematic_beta_v1'; RAW=OUT/'raw'; MANIFESTS=OUT/'manifests'
for p in [OUT,RAW,MANIFESTS]:p.mkdir(exist_ok=True)

DESIGN={'version':'firm_thematic_beta_v1','primary_factor':'broad_cm_equity_ex_market',
 'robustness_factors':['producer_ex_market','base_metals_raw','materials_ex_market'],
 'return_frequency':'daily','historical_window_days':252,'forward_target_days':252,
 'common_cutoff':'2025-12-31','minimum_observations':126,
 'time_series_model':'OLS: firm return ~ intercept + SPY return + thematic factor',
 'multiple_share_class_aggregation':'lagged-market-equity weighted PERMNO return within PERMCO',
 'missing_return_policy':'retain CRSP numeric observations only; no imputation',
 'portfolio_results_accessed':False,'scenario_13a_scores_accessed':False}
design_path=MANIFESTS/'thematic_beta_design.json'
if design_path.exists():assert json.loads(design_path.read_text())==DESIGN
else:design_path.write_text(json.dumps(DESIGN,indent=2,sort_keys=True)+'\\n')
design_hash=hashlib.sha256(design_path.read_bytes()).hexdigest()
print({'design_sha256':design_hash,'portfolio_results_accessed':False})""")

md("""## 2. Input audit

Run 14A after its cutoff-alignment revision before continuing.""")
code("""paths={'factor':FACTOR/'daily_critical_minerals_factors.parquet',
 'windows':FACTOR/'manifests'/'frozen_beta_windows.parquet',
 'signals':ROOT/'long_short_hybrid_v1'/'signed_signal_panel.parquet',
 'bridge':ROOT/'data'/'processed'/'sp500_point_in_time_identifier_bridge.parquet'}
audit=pd.DataFrame([{'input':k,'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None} for k,p in paths.items()])
display(audit);assert audit.exists.all(),'Run corrected 14A or restore a missing frozen input.'
factor=pd.read_parquet(paths['factor']);factor['date']=pd.to_datetime(factor.date)
windows=pd.read_parquet(paths['windows']);signals=pd.read_parquet(paths['signals'])
bridge=pd.read_parquet(paths['bridge'])
assert factor.date.max()<=pd.Timestamp(DESIGN['common_cutoff'])
display(windows)""")

md("""## 3. Connect to WRDS only when the checkpoint is absent""")
code("""security_path=RAW/'crsp_daily_security_returns_2023_2025.parquet'
needs_wrds=not security_path.exists()
if needs_wrds:
    import wrds
    username=input('WRDS username: ').strip();password=getpass.getpass('WRDS password: ')
    db=wrds.Connection(wrds_username=username,wrds_password=password,autoconnect=False,verbose=True,
      wrds_connect_args={'sslmode':'require','connect_timeout':30,'application_name':'cm_thematic_beta'})
    db.connect();display(db.raw_sql('select 1 as connection_test'))
else:print('CRSP checkpoint exists; WRDS connection is not required.')""")

md("""## 4. Acquire the minimal CRSP daily extract

The union contains only securities appearing in one of the four point-in-time S&P 500 universes. Queries are checkpointed as one combined file.""")
code("""mapping=bridge[['permno','permco']].dropna().drop_duplicates().astype({'permno':int,'permco':int})
assert mapping.groupby('permno').permco.nunique().max()==1
permnos=sorted(mapping.permno.unique())

def rollback():
    try:db.connection.rollback()
    except Exception:pass

if security_path.exists():
    security=pd.read_parquet(security_path);source='local_checkpoint'
else:
    candidates=[
     {'table':'crsp_a_stock.wrds_dsfv2_query','date':'dlycaldt','ret':'dlyret','prc':'dlyprc','shrout':'shrout'},
     {'table':'crsp_a_stock.dsf_v2','date':'dlycaldt','ret':'dlyret','prc':'dlyprc','shrout':'shrout'},
     {'table':'crsp_a_stock.dsf','date':'date','ret':'ret','prc':'prc','shrout':'shrout'},
     {'table':'crsp.dsf','date':'date','ret':'ret','prc':'prc','shrout':'shrout'}]
    spec=None;errors=[]
    for candidate in candidates:
        try:
            db.raw_sql(f\"select permno,{candidate['date']},{candidate['ret']},{candidate['prc']},{candidate['shrout']} from {candidate['table']} limit 1\")
            spec=candidate;break
        except Exception as error:
            errors.append((candidate['table'],type(error).__name__));rollback()
    if spec is None:raise RuntimeError(f'No daily CRSP table queryable: {errors}')
    frames=[]
    for start in range(0,len(permnos),250):
        ids=','.join(map(str,permnos[start:start+250]))
        q=f\"\"\"select permno,{spec['date']} as date,{spec['ret']} as ret,{spec['prc']} as prc,{spec['shrout']} as shrout
        from {spec['table']} where {spec['date']} between '2023-03-01' and '2025-12-31' and permno in ({ids})\"\"\"
        frames.append(db.raw_sql(q,date_cols=['date']));print(f'batch {start//250+1}/{int(np.ceil(len(permnos)/250))} complete')
    security=pd.concat(frames,ignore_index=True);security.to_parquet(security_path,index=False);source=spec['table']
print({'source':source,'rows':len(security),'permnos':security.permno.nunique()})""")

md("""## 5. Clean and aggregate security returns to firms""")
code("""security['date']=pd.to_datetime(security.date)
for c in ['permno','ret','prc','shrout']:security[c]=pd.to_numeric(security[c],errors='coerce')
security=security.merge(mapping,on='permno',how='inner',validate='many_to_one')
security=security.sort_values(['permno','date']).drop_duplicates(['permno','date'],keep='last')
security['market_equity']=security.prc.abs()*security.shrout
security['lag_market_equity']=security.groupby('permno').market_equity.shift(1)
valid=security.dropna(subset=['ret']).copy()

def aggregate_firm(g):
    w=g.lag_market_equity.where(g.lag_market_equity.gt(0))
    value=np.average(g.ret,weights=w) if w.notna().all() and w.sum()>0 else g.ret.mean()
    return pd.Series({'firm_return':value,'underlying_permnos':g.permno.nunique()})
firm_daily=(valid.groupby(['permco','date'],as_index=False).apply(aggregate_firm,include_groups=False).reset_index())
if 'level_2' in firm_daily:firm_daily=firm_daily.drop(columns='level_2')
firm_daily.to_parquet(OUT/'crsp_daily_firm_returns.parquet',index=False)
display(pd.DataFrame([{'rows':len(firm_daily),'firms':firm_daily.permco.nunique(),'first_date':firm_daily.date.min(),
                       'last_date':firm_daily.date.max(),'duplicate_firm_dates':int(firm_daily.duplicated(['permco','date']).sum())}]))""")

md("""## 6. Estimate historical and forward betas

The same regression is estimated separately for each firm, vintage, window, and frozen factor. No beta is reported with fewer than 126 observations.""")
code("""factor_names=[DESIGN['primary_factor']]+DESIGN['robustness_factors'];rows=[]
for w in windows.itertuples(index=False):
    ranges={'historical':(pd.Timestamp(w.historical_start),pd.Timestamp(w.historical_end)),
            'forward':(pd.Timestamp(w.forward_start),min(pd.Timestamp(w.forward_end),pd.Timestamp(DESIGN['common_cutoff'])))}
    vintage_firms=set(signals.loc[signals.vintage.eq(w.vintage),'permco'].astype(int))
    for window,(start,end) in ranges.items():
        base=firm_daily[firm_daily.permco.isin(vintage_firms)&firm_daily.date.between(start,end)].merge(factor,on='date',how='inner')
        for permco,g in base.groupby('permco'):
            for factor_name in factor_names:
                z=g[['firm_return','market_return',factor_name]].dropna()
                if len(z)<DESIGN['minimum_observations']:
                    rows.append({'vintage':w.vintage,'window':window,'permco':permco,'factor':factor_name,'observations':len(z),'alpha':np.nan,'market_beta':np.nan,'thematic_beta':np.nan,'r2':np.nan});continue
                X=np.column_stack([np.ones(len(z)),z.market_return,z[factor_name]])
                coef=np.linalg.lstsq(X,z.firm_return.to_numpy(),rcond=None)[0];pred=X@coef
                sst=((z.firm_return-z.firm_return.mean())**2).sum();sse=((z.firm_return-pred)**2).sum()
                rows.append({'vintage':w.vintage,'window':window,'permco':permco,'factor':factor_name,'observations':len(z),
                             'alpha':coef[0],'market_beta':coef[1],'thematic_beta':coef[2],'r2':1-sse/sst if sst>0 else np.nan})
betas=pd.DataFrame(rows);betas.to_parquet(OUT/'firm_thematic_betas.parquet',index=False)
coverage=(betas.groupby(['vintage','window','factor'],as_index=False).agg(firms=('permco','nunique'),
          estimated_betas=('thematic_beta','count'),median_observations=('observations','median'),median_r2=('r2','median')))
display(coverage);assert betas[betas.factor.eq(DESIGN['primary_factor'])].groupby(['vintage','window']).thematic_beta.count().min()>=350""")

md("""## 7. Blinded descriptive diagnostics

These diagnostics inspect beta estimation quality only. Exposure scores are not joined until 14C.""")
code("""primary=betas[betas.factor.eq(DESIGN['primary_factor'])]
display(primary.groupby(['vintage','window']).thematic_beta.describe(percentiles=[.01,.1,.5,.9,.99]))
stability=(primary.pivot_table(index=['vintage','permco'],columns='window',values='thematic_beta').dropna())
print('Historical-forward beta correlation:',stability.historical.corr(stability.forward))
gates=pd.DataFrame([
 {'gate':'common_cutoff_respected','value':int(firm_daily.date.max()<=pd.Timestamp(DESIGN['common_cutoff'])),'threshold':1,'passed':firm_daily.date.max()<=pd.Timestamp(DESIGN['common_cutoff'])},
 {'gate':'minimum_350_primary_betas_each_cell','value':int(primary.groupby(['vintage','window']).thematic_beta.count().min()),'threshold':350,'passed':primary.groupby(['vintage','window']).thematic_beta.count().min()>=350},
 {'gate':'portfolio_results_unopened','value':0,'threshold':0,'passed':True}])
display(gates);gates.to_parquet(MANIFESTS/'beta_estimation_gates.parquet',index=False);assert gates.passed.all()""")

md("""## 8. Manifest and handoff""")
code("""artifacts=[design_path,security_path,OUT/'crsp_daily_firm_returns.parquet',OUT/'firm_thematic_betas.parquet',MANIFESTS/'beta_estimation_gates.parquet']
inventory=pd.DataFrame([{'artifact':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts])
inventory.to_parquet(MANIFESTS/'artifact_inventory.parquet',index=False)
manifest={'stage':'firm_thematic_beta_estimation_v1','design_sha256':design_hash,'primary_factor':DESIGN['primary_factor'],
 'beta_rows':len(betas),'primary_estimated':int(betas.loc[betas.factor.eq(DESIGN['primary_factor']),'thematic_beta'].count()),
 'common_cutoff':DESIGN['common_cutoff'],'portfolio_results_accessed':False,'exposure_scores_joined':False}
(MANIFESTS/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n')
display(inventory);print('Saved:',OUT)
print('Next: 14C joins frozen scores only after these beta estimates are checkpointed.')""")

nb['cells']=cells;nbf.write(nb,OUT);print(OUT)
