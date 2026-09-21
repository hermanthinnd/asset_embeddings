from pathlib import Path
import nbformat as nbf
R=Path(__file__).resolve().parent;T=R/'14D_Factor_Controlled_Portfolio_Attribution.ipynb';n=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});c=[]
def M(x):c.append(nbf.v4.new_markdown_cell(x))
def X(x):c.append(nbf.v4.new_code_cell(x))
M('''# 14D — Factor-controlled portfolio attribution

Do the frozen text, embedding, hybrid, and consensus portfolios merely load on conventional equity risks and the independent critical-minerals theme factor, or is a residual intercept left? This attribution stage never reselects securities or changes weights. It uses daily CRSP returns over the exact frozen three-month windows, with the concatenated calendar-time series as the primary test and each vintage as a stability diagnostic.''')
M('## 1. Setup and frozen design')
X('''from pathlib import Path
import hashlib,io,json,urllib.request,zipfile
import numpy as np,pandas as pd,statsmodels.api as sm
from IPython.display import display
ROOT=Path.cwd()
if ROOT.name!='sp500_critical_minerals':
 q=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
 if q.exists():ROOT=q
OUT=ROOT/'factor_attribution_v1';RAW=OUT/'raw';MAN=OUT/'manifests'
for q in [OUT,RAW,MAN]:q.mkdir(exist_ok=True)
DESIGN={'version':'factor_controlled_portfolio_attribution_v1','frequency':'daily','windows':'exact frozen three-month forward windows','primary_panel':'concatenated calendar time by method','secondary_panel':'each vintage','long_short_y':'raw self-financing return','long_only_y':'return minus RF','controls':'daily FF5 plus momentum plus frozen 14A theme factor','HAC_lags':5,'no_reselection':True}
dp=MAN/'attribution_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2,sort_keys=True)+'\\n')
dh=hashlib.sha256(dp.read_bytes()).hexdigest();print({'design_sha256':dh,'regressions_run':False})''')
M('## 2. Load frozen memberships, windows, returns, and thematic factors')
X('''mp=ROOT/'long_short_hybrid_v1/frozen_portfolio_membership.parquet';cp=ROOT/'consensus_agreement_v1/frozen_consensus_membership.parquet';rp=ROOT/'thematic_beta_v1/crsp_daily_firm_returns.parquet';fp=ROOT/'thematic_factor_v1/daily_critical_minerals_factors.parquet';sp=ROOT/'config/sp500_experiment_design_v1.json';paths=[mp,cp,rp,fp,sp]
audit=pd.DataFrame([{'path':str(q),'exists':q.exists(),'sha256':hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None} for q in paths]);display(audit);assert audit.exists.all()
m=pd.concat([pd.read_parquet(mp)[['vintage','method','leg','permco','weight']],pd.read_parquet(cp)[['vintage','method','leg','permco','weight']]],ignore_index=True);m['permco']=pd.to_numeric(m.permco,errors='coerce').astype('Int64');m=m.groupby(['vintage','method','leg','permco'],as_index=False).weight.sum()
d=pd.read_parquet(rp);d['date']=pd.to_datetime(d.date);d['permco']=pd.to_numeric(d.permco,errors='coerce').astype('Int64');d['firm_return']=pd.to_numeric(d.firm_return,errors='coerce');d=d.dropna(subset=['permco','date','firm_return']).drop_duplicates(['permco','date'])
f=pd.read_parquet(fp);f['date']=pd.to_datetime(f.date);cfg=json.loads(sp.read_text());w=pd.DataFrame([{'vintage':v['label'],'month':pd.Period(x,'M')} for v in cfg['vintages'] for x in v['forward_months']]);display(m.groupby(['vintage','method']).agg(names=('permco','nunique'),net=('weight','sum'),gross=('weight',lambda x:x.abs().sum())).reset_index())''')
M('''## 3. Reconstruct daily frozen returns

Missing returns contribute zero to the raw calculation but are not hidden: observed gross-weight coverage is audited and no surviving firm is upweighted.''')
X('''dates=d[['date']].drop_duplicates();dates['month']=dates.date.dt.to_period('M');dates=dates.merge(w,on='month');g=m.merge(dates,on='vintage').merge(d,on=['permco','date'],how='left',validate='many_to_one');g['contribution']=g.weight*g.firm_return;g['absolute']=g.weight.abs();g['observed']=np.where(g.firm_return.notna(),g.absolute,0)
daily=g.groupby(['vintage','method','date'],as_index=False).agg(portfolio_return=('contribution','sum'),target_gross=('absolute','sum'),observed_gross=('observed','sum'),names=('permco','nunique'),observed_names=('firm_return','count'));daily['coverage']=daily.observed_gross/daily.target_gross;coverage=daily.groupby(['vintage','method'],as_index=False).agg(days=('date','nunique'),minimum_coverage=('coverage','min'),mean_coverage=('coverage','mean'));display(coverage);assert coverage.days.min()>=55;assert coverage.minimum_coverage.min()>=.90;daily.to_parquet(OUT/'daily_frozen_portfolio_returns.parquet',index=False)''')
M('''## 4. Official daily Fama–French controls

Small ZIP files from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) are checkpointed locally.''')
X('''URL={'ff5':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip','mom':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'}
def get(k):
 p=RAW/f'{k}_daily.zip'
 if not p.exists():p.write_bytes(urllib.request.urlopen(urllib.request.Request(URL[k],headers={'User-Agent':'Academic research'}),timeout=60).read())
 return p
def parse(p,needle):
 with zipfile.ZipFile(p) as z:s=z.read([a for a in z.namelist() if a.lower().endswith('.csv')][0]).decode('utf-8-sig',errors='replace').splitlines()
 h=next(i for i,a in enumerate(s) if needle in a.replace(' ',''));rows=[]
 for a in s[h+1:]:
  k=a.split(',')[0].strip()
  if not(len(k)==8 and k.isdigit()):break
  rows.append(a)
 x=pd.read_csv(io.StringIO(s[h]+'\\n'+'\\n'.join(rows))).rename(columns=lambda a:str(a).strip());x=x.rename(columns={x.columns[0]:'yyyymmdd'});x['date']=pd.to_datetime(x.yyyymmdd.astype(str),format='%Y%m%d');return x.drop(columns='yyyymmdd')
std=parse(get('ff5'),'Mkt-RF').merge(parse(get('mom'),'Mom'),on='date');std.columns=[str(a).strip() for a in std.columns]
for a in ['Mkt-RF','SMB','HML','RMW','CMA','RF','Mom']:std[a]=pd.to_numeric(std[a],errors='coerce')/100
std=std[['date','Mkt-RF','SMB','HML','RMW','CMA','Mom','RF']];std.to_parquet(OUT/'daily_standard_factors.parquet',index=False);display(std.tail())''')
M('## 5. Assemble vintage and primary calendar-time panels')
X('''panel=daily.merge(std,on='date',validate='many_to_one').merge(f,on='date',validate='many_to_one');panel['dependent_return']=np.where(panel.method.eq('hybrid_long_only'),panel.portfolio_return-panel.RF,panel.portfolio_return);panel.to_parquet(OUT/'daily_attribution_panel.parquet',index=False)
fc=['Mkt-RF','SMB','HML','RMW','CMA','Mom','RF','broad_cm_equity_ex_market','producer_ex_market','base_metals_raw','materials_ex_market'];cal=panel.groupby(['method','date'],as_index=False).agg(portfolio_return=('portfolio_return','mean'),dependent_return=('dependent_return','mean'),live_vintages=('vintage','nunique'),**{a:(a,'first') for a in fc});cal.to_parquet(OUT/'calendar_time_attribution_panel.parquet',index=False);display(cal.groupby('method').agg(days=('date','nunique'),mean_return=('portfolio_return','mean')).reset_index())''')
M('''## 6. Frozen attribution regressions

Daily alpha is annualized by 252. HAC standard errors allow short-run dependence. A remaining alpha is not proof of skill because the history is short and transaction costs are absent.''')
X('''S={'mean_only':[],'capm':['Mkt-RF'],'capm_plus_cm':['Mkt-RF','broad_cm_equity_ex_market'],'ff5':['Mkt-RF','SMB','HML','RMW','CMA'],'ff5_mom':['Mkt-RF','SMB','HML','RMW','CMA','Mom'],'ff5_mom_cm':['Mkt-RF','SMB','HML','RMW','CMA','Mom','broad_cm_equity_ex_market']}
def fit(z,pn,v,m):
 out=[]
 for spec,cols in S.items():
  a=z[['dependent_return']+cols].dropna();r=sm.OLS(a.dependent_return,sm.add_constant(a[cols],has_constant='add')).fit(cov_type='HAC',cov_kwds={'maxlags':5,'use_correction':True});ci=r.conf_int().loc['const'];out.append({'panel':pn,'vintage':v,'method':m,'specification':spec,'days':len(a),'daily_alpha':r.params['const'],'annualized_alpha':252*r.params['const'],'alpha_ci_low':252*ci.iloc[0],'alpha_ci_high':252*ci.iloc[1],'alpha_hac_t':r.tvalues['const'],'alpha_p_value':r.pvalues['const'],'r2':r.rsquared,'cm_beta':r.params.get('broad_cm_equity_ex_market',np.nan),'cm_beta_hac_t':r.tvalues.get('broad_cm_equity_ex_market',np.nan)})
 return out
rows=[]
for (v,m),a in panel.groupby(['vintage','method']):rows+=fit(a,'vintage',v,m)
for m,a in cal.groupby('method'):rows+=fit(a,'calendar_time','all_windows',m)
res=pd.DataFrame(rows);res.to_parquet(OUT/'factor_attribution_results.parquet',index=False);display(res[(res.panel.eq('calendar_time'))&res.specification.isin(['mean_only','ff5_mom','ff5_mom_cm'])][['method','specification','days','annualized_alpha','alpha_ci_low','alpha_ci_high','alpha_hac_t','alpha_p_value','cm_beta','cm_beta_hac_t','r2']].round(4))''')
M('## 7. Incremental critical-minerals attribution')
X('''b=res[res.specification.eq('ff5_mom')].set_index(['panel','vintage','method']);a=res[res.specification.eq('ff5_mom_cm')].set_index(['panel','vintage','method']);inc=pd.DataFrame({'alpha_before_cm':b.daily_alpha,'alpha_after_cm':a.daily_alpha,'cm_beta':a.cm_beta,'cm_beta_hac_t':a.cm_beta_hac_t,'r2_before_cm':b.r2,'r2_after_cm':a.r2}).reset_index();inc['annualized_alpha_change']=252*(inc.alpha_after_cm-inc.alpha_before_cm);inc['incremental_r2']=inc.r2_after_cm-inc.r2_before_cm;inc.to_parquet(OUT/'critical_minerals_incremental_attribution.parquet',index=False);display(inc[inc.panel.eq('calendar_time')].round(4))''')
M('## 8. Frozen thematic-factor robustness')
X('''rr=[]
for pn,groups in [('vintage',panel.groupby(['vintage','method'])),('calendar_time',cal.groupby('method'))]:
 for key,z in groups:
  v,m=(key if pn=='vintage' else ('all_windows',key))
  for tf in ['broad_cm_equity_ex_market','producer_ex_market','base_metals_raw','materials_ex_market']:
   cols=['Mkt-RF','SMB','HML','RMW','CMA','Mom',tf];a=z[['dependent_return']+cols].dropna();r=sm.OLS(a.dependent_return,sm.add_constant(a[cols],has_constant='add')).fit(cov_type='HAC',cov_kwds={'maxlags':5,'use_correction':True});rr.append({'panel':pn,'vintage':v,'method':m,'theme_factor':tf,'days':len(a),'annualized_alpha':252*r.params['const'],'alpha_hac_t':r.tvalues['const'],'theme_beta':r.params[tf],'theme_beta_hac_t':r.tvalues[tf],'r2':r.rsquared})
rob=pd.DataFrame(rr);rob.to_parquet(OUT/'thematic_factor_robustness.parquet',index=False);display(rob[rob.panel.eq('calendar_time')].round(4))''')
M('## 9. Gates and reproducibility manifest')
X('''p=res[(res.panel.eq('calendar_time'))&res.specification.eq('ff5_mom_cm')];gates=pd.DataFrame([{'gate':'six_frozen_methods','value':p.method.nunique(),'threshold':6,'passed':p.method.nunique()==6},{'gate':'minimum_calendar_days','value':p.days.min(),'threshold':230,'passed':p.days.min()>=230},{'gate':'four_vintages','value':res[res.panel.eq('vintage')].vintage.nunique(),'threshold':4,'passed':res[res.panel.eq('vintage')].vintage.nunique()==4},{'gate':'minimum_coverage','value':coverage.minimum_coverage.min(),'threshold':.90,'passed':coverage.minimum_coverage.min()>=.90},{'gate':'no_reselection','value':0,'threshold':0,'passed':True}]);gates.to_parquet(MAN/'decision_gates.parquet',index=False);display(gates)
arts=[dp,OUT/'daily_frozen_portfolio_returns.parquet',OUT/'daily_standard_factors.parquet',OUT/'daily_attribution_panel.parquet',OUT/'calendar_time_attribution_panel.parquet',OUT/'factor_attribution_results.parquet',OUT/'critical_minerals_incremental_attribution.parquet',OUT/'thematic_factor_robustness.parquet',MAN/'decision_gates.parquet'];pd.DataFrame([{'artifact':str(a.relative_to(ROOT)),'sha256':hashlib.sha256(a.read_bytes()).hexdigest()} for a in arts]).to_parquet(MAN/'artifact_inventory.parquet',index=False);(MAN/'manifest.json').write_text(json.dumps({'stage':DESIGN['version'],'design_sha256':dh,'methods':sorted(p.method.unique()),'calendar_days':int(p.days.min()),'all_gates_passed':bool(gates.passed.all()),'reselected':False},indent=2,sort_keys=True)+'\\n');print('Saved:',OUT)''')
M('''## Interpretation boundary

Critical-minerals beta measures sensitivity to an investable equity-theme factor, not proof of a physical supply-chain link. Alpha reduction after adding it indicates thematic attribution. Remaining alpha is exploratory—not proof of skill—given the short sample, absent transaction costs, and the fact that this attribution follows earlier pilot analysis. Robustness factors were frozen in 14A and cannot replace the primary factor based on preferred results.''')
n['cells']=c;nbf.write(n,T);print(T)
