from pathlib import Path
import nbformat as nbf

R=Path(__file__).resolve().parent
n=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
def md(x): n.cells.append(nbf.v4.new_markdown_cell(x))
def c(x): n.cells.append(nbf.v4.new_code_cell(x))

md('''# 12G — Locked Forward-Performance Evaluation

This notebook opens outcomes only after text extraction, embedding propagation, portfolio membership, and placebos have been frozen. The primary comparison uses equal-sized, equal-weighted LLM-only and embedding-only portfolios. Embedding performance is also tested against 100 matched-cardinality placebo portfolios.''')
md('''## 1. Setup and immutable paths

All inputs are local checkpoints. This stage makes no OpenAI API calls and requires no WRDS connection.''')
c('''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd().resolve()
if ROOT.name!='sp500_critical_minerals': ROOT=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
CONFIG=ROOT/'config'; PROP=ROOT/'embedding_propagation_v1'; OUT=ROOT/'forward_performance_v1'; OUT.mkdir(exist_ok=True)
RETURNS=ROOT.parent/'quarterly_sec_pilot'/'forward_performance_2025_2026'/'data'/'processed'/'crsp_firm_monthly_through_cutoff.parquet'
UNIVERSE=ROOT/'data'/'processed'/'sp500_point_in_time_membership_with_crsp_ids.parquet'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def savej(x,p): p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\\n')
print('Root:',ROOT); print('Return checkpoint:',RETURNS.exists())''')

md('''## 2. Verify the frozen design before opening returns

The hashes bind this analysis to the exact portfolios and placebo draws created by 12F. A thematic index is not added after outcome access because that would permit benchmark selection with hindsight.''')
c('''contract_path=CONFIG/'forward_performance_v1.json'; contract=json.loads(contract_path.read_text())
gates=pd.read_parquet(PROP/'embedding_propagation_gates.parquet'); assert gates.passed.all()
primary_path=PROP/'primary_equal_cardinality_portfolios.parquet'; placebo_path=PROP/'matched_placebo_portfolios_100.parquet'
audit=pd.DataFrame([{'input':x,'path':str(p),'sha256':sha(p)} for x,p in [('contract',contract_path),('primary',primary_path),('placebos',placebo_path),('returns',RETURNS)]])
audit.to_parquet(OUT/'input_hash_audit.parquet',index=False)
display(gates); display(audit[['input','sha256']])''')

md('''## 3. Build the primary three-month outcome

The months were declared in the original design. A firm needs all three monthly observations. Returns are compounded within firm first and then averaged across firms, ensuring each selected firm has one portfolio weight.''')
c('''design=json.loads((CONFIG/'sp500_experiment_design_v1.json').read_text())
windows=pd.DataFrame([{'vintage':v['label'],'month':pd.Period(m,'M')} for v in design['vintages'] for m in v['forward_months']])
r=pd.read_parquet(RETURNS); r['permco']=pd.to_numeric(r.permco,errors='coerce').astype('Int64'); r['month']=pd.to_datetime(r.eom).dt.to_period('M'); r['eom_ret']=pd.to_numeric(r.eom_ret,errors='coerce')
r=r.dropna(subset=['permco','month','eom_ret']).drop_duplicates(['permco','month'])
u=pd.read_parquet(UNIVERSE); u['permco']=pd.to_numeric(u.permco,errors='coerce').astype('Int64'); u=u[['vintage','permco']].dropna().drop_duplicates()
fw=u.merge(windows,on='vintage',how='inner').merge(r[['permco','month','eom_ret']],on=['permco','month'],how='left',validate='many_to_one')
fb=(fw.groupby(['vintage','permco'],as_index=False).agg(months=('eom_ret','count'),buy_hold_return=('eom_ret',lambda x:np.prod(1+x.dropna())-1 if x.notna().any() else np.nan)))
fb['complete']=fb.months.eq(3)
benchmark=(fb[fb.complete].groupby('vintage',as_index=False).agg(sp500_equal_weight_return=('buy_hold_return','mean'),benchmark_firms=('permco','nunique')))
universe_coverage=(fb.groupby('vintage',as_index=False).agg(universe_firms=('permco','nunique'),complete_return_firms=('complete','sum'))); universe_coverage['coverage']=universe_coverage.complete_return_firms/universe_coverage.universe_firms
display(universe_coverage.merge(benchmark,on='vintage'))''')

md('''## 4. Primary LLM-versus-embedding comparison

The portfolios have equal cardinality within vintage and are disjoint. The matched absolute-return difference is primary. The available-universe average is shown only as a descriptive common-support diagnostic because the local checkpoint does not cover every S&P 500 firm.''')
c('''p=pd.read_parquet(primary_path); p['permco']=pd.to_numeric(p.permco,errors='coerce').astype('Int64'); pp=p.merge(fb,on=['vintage','permco'],how='left',validate='many_to_one')
def summarize(g):
    z=g[g.complete]
    return pd.Series({'selected_firms':g.permco.nunique(),'observed_firms':z.permco.nunique(),'portfolio_return':z.buy_hold_return.mean()})
primary=pp.groupby(['vintage','portfolio'],as_index=False).apply(summarize,include_groups=False).reset_index(drop=True)
primary=primary.merge(benchmark,on='vintage',validate='many_to_one').rename(columns={'sp500_equal_weight_return':'available_universe_return'}); primary['available_universe_adjusted_return']=primary.portfolio_return-primary.available_universe_return
w=primary.pivot(index='vintage',columns='portfolio',values='portfolio_return').reset_index(); w['embedding_minus_llm']=w.embedding_only_primary-w.llm_only_common_support
primary.to_parquet(OUT/'primary_portfolio_results.parquet',index=False); w.to_parquet(OUT/'primary_paired_comparison.parquet',index=False)
display(primary); display(w)''')

md('''## 5. Matched-placebo inference

Every placebo has the same cardinality as the embedding-only portfolio in its vintage. The pooled test preserves placebo draw identity across quarters and compares four-vintage mean returns.''')
c('''pl=pd.read_parquet(placebo_path); pl['permco']=pd.to_numeric(pl.permco,errors='coerce').astype('Int64'); xp=pl.merge(fb,on=['vintage','permco'],how='left',validate='many_to_one')
placebo=xp.groupby(['vintage','placebo_draw'],as_index=False).apply(summarize,include_groups=False).reset_index(drop=True).rename(columns={'portfolio_return':'placebo_return'})
obs=primary[primary.portfolio.eq('embedding_only_primary')][['vintage','portfolio_return']].rename(columns={'portfolio_return':'embedding_return'})
placebo=placebo.merge(obs,on='vintage',validate='many_to_one')
def ps(g):
    o=g.embedding_return.iloc[0]; x=g.placebo_return.dropna(); return pd.Series({'embedding_return':o,'placebo_mean':x.mean(),'placebo_q05':x.quantile(.05),'placebo_q50':x.quantile(.5),'placebo_q95':x.quantile(.95),'embedding_percentile':(x<o).mean(),'one_sided_empirical_p':(1+(x>=o).sum())/(1+len(x)),'draws':len(x)})
by_vintage=placebo.groupby('vintage',as_index=False).apply(ps,include_groups=False).reset_index(drop=True)
pooled=placebo.groupby('placebo_draw',as_index=False).placebo_return.mean().rename(columns={'placebo_return':'mean_placebo_return'}); om=obs.embedding_return.mean(); x=pooled.mean_placebo_return
pooled_summary=pd.DataFrame([{'vintages':4,'embedding_mean_return':om,'placebo_mean_return':x.mean(),'placebo_q05':x.quantile(.05),'placebo_q95':x.quantile(.95),'embedding_percentile':(x<om).mean(),'one_sided_empirical_p':(1+(x>=om).sum())/(1+len(x))}])
placebo.to_parquet(OUT/'matched_placebo_returns.parquet',index=False); by_vintage.to_parquet(OUT/'matched_placebo_summary.parquet',index=False); pooled.to_parquet(OUT/'pooled_placebo_distribution.parquet',index=False); pooled_summary.to_parquet(OUT/'pooled_placebo_summary.parquet',index=False)
display(by_vintage); display(pooled_summary)''')

md('''## 6. Coverage, gates, and visualization

The chart is descriptive; matched-placebo inference is the formal test. All portfolio and benchmark coverage gates must pass.''')
c('''diag=primary[['vintage','portfolio','selected_firms','observed_firms']].copy(); diag['return_coverage']=diag.observed_firms/diag.selected_firms
card=primary.pivot(index='vintage',columns='portfolio',values='selected_firms')
fg=pd.DataFrame([
 {'gate':'four_vintages','value':primary.vintage.nunique(),'threshold':4,'passed':primary.vintage.nunique()==4},
 {'gate':'equal_primary_cardinality','value':int((card.embedding_only_primary==card.llm_only_common_support).sum()),'threshold':4,'passed':bool((card.embedding_only_primary==card.llm_only_common_support).all())},
 {'gate':'minimum_primary_return_coverage','value':diag.return_coverage.min(),'threshold':.90,'passed':diag.return_coverage.min()>=.90},
 {'gate':'minimum_descriptive_universe_coverage','value':universe_coverage.coverage.min(),'threshold':.80,'passed':universe_coverage.coverage.min()>=.80},
 {'gate':'one_hundred_placebos_each','value':placebo.groupby('vintage').placebo_draw.nunique().min(),'threshold':100,'passed':placebo.groupby('vintage').placebo_draw.nunique().min()==100}])
diag.to_parquet(OUT/'return_coverage_diagnostics.parquet',index=False); fg.to_parquet(OUT/'forward_performance_gates.parquet',index=False)
plot=primary.pivot(index='vintage',columns='portfolio',values='portfolio_return'); ax=plot.plot.bar(figsize=(10,5),color=['#D55E00','#0072B2']); ax.axhline(0,color='black',lw=.8); ax.set_ylabel('Three-month buy-and-hold return'); ax.set_xlabel('Formation vintage'); ax.set_title('Direct LLM versus embedding-only discoveries')
for q in ax.containers: ax.bar_label(q,fmt='%.1%%',padding=3,fontsize=8)
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/'primary_portfolio_returns.png',dpi=180,bbox_inches='tight'); plt.show()
display(diag); display(fg); assert fg.passed.all()''')

md('''## 7. Reproducibility manifest and interpretation boundary

With four vintages, this is a disciplined proof-of-concept rather than definitive time-series evidence. No factor alpha is claimed from three observations per vintage, and the cached return checkpoint does not expose a separate delisting-return field.''')
c('''arts=[]
for f in sorted(OUT.glob('*')):
    if f.is_file() and f.name!='forward_performance_manifest.json': arts.append({'artifact':f.name,'rows':len(pd.read_parquet(f)) if f.suffix=='.parquet' else None,'bytes':f.stat().st_size,'sha256':sha(f)})
manifest={'stage':'locked_forward_performance_v1','contract_sha256':sha(contract_path),'all_gates_passed':bool(fg.passed.all()),'vintages':sorted(primary.vintage.unique()),'limitations':['four vintages','three-month primary horizon','no separate delisting-return field','no pre-frozen thematic index'],'artifacts':arts}
savej(manifest,OUT/'forward_performance_manifest.json'); display(pd.DataFrame(arts)); print('Saved:',OUT)''')

nbf.write(n,R/'12G_Locked_Forward_Performance_Evaluation.ipynb')
