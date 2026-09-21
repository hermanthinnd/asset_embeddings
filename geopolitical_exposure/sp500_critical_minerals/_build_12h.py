from pathlib import Path
import nbformat as nbf
R=Path(__file__).resolve().parent
n=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
M=lambda x:n.cells.append(nbf.v4.new_markdown_cell(x)); C=lambda x:n.cells.append(nbf.v4.new_code_cell(x))
M('''# 12H — Long–Short and Hybrid Portfolio Completion

This stage completes portfolio methods already declared in the original experiment design. It evaluates direct-text, embedding-only, and hybrid long–short portfolios, plus a practical 50/50-sleeve hybrid long-only portfolio. Because 12G outcomes have already been viewed, these are labeled pre-specified-method completion results rather than a newly blinded confirmation.''')
M('''## 1. Setup and frozen contract

The contract fixes capital allocation, scoring, cardinalities, and placebo counts. No parameter is selected using returns.''')
C('''from pathlib import Path
import json,hashlib,numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path.cwd().resolve()
if ROOT.name!='sp500_critical_minerals': ROOT=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
PROJECT=ROOT.parents[1]; CONFIG=ROOT/'config'; OUT=ROOT/'long_short_hybrid_v1'; OUT.mkdir(exist_ok=True)
contract_path=CONFIG/'long_short_hybrid_v1.json'; contract=json.loads(contract_path.read_text())
Q=pd.read_parquet(ROOT/'quarterly_exposure_v1'/'quarterly_firm_exposure_panel.parquet')
NULL=pd.read_parquet(ROOT/'embedding_propagation_v1'/'random_pair_null_summary.parquet').set_index('vintage')
RET=ROOT.parent/'quarterly_sec_pilot'/'forward_performance_2025_2026'/'data'/'processed'/'crsp_firm_monthly_through_cutoff.parquet'
def z(x): x=pd.Series(x,dtype=float); s=x.std(ddof=0); return (x-x.mean())/s if s>0 else x*0
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
print(contract['status'])''')
M('''## 2. Reconstruct positive and negative embedding propagation

Positive and negative seeds are propagated separately under the frozen 12F similarity threshold. Every direct LLM seed is excluded from embedding-only candidates. A candidate’s signed embedding signal is its strongest positive propagated score minus its strongest negative propagated score.''')
C('''rows=[]; frames={}; dims=[f'embedding_{i:03d}' for i in range(1,11)]
for v in sorted(Q.vintage.unique()):
 p=PROJECT/'full_replication_80q'/'embeddings'/v/'rs_l_min'/'asset_embeddings_normalized_labeled.parquet'; f=pd.read_parquet(p)[['permco']+dims].drop_duplicates('permco'); X=f[dims].to_numpy(float); norm=np.linalg.norm(X,axis=1); keep=np.isfinite(X).all(1)&(norm>0); f=f.loc[keep].reset_index(drop=True); X=X[keep]/norm[keep,None]; frames[v]=(f,X)
 u=Q[Q.vintage.eq(v)].copy(); universe=set(u.permco); eligible=u.positive_seed|u.negative_seed; u['assigned_direction']=np.where(eligible&(u.net_score>0),'positive',np.where(eligible&(u.net_score<0),'negative','none')); all_seeds=set(u.loc[eligible,'permco']); idx=np.flatnonzero(f.permco.isin(universe)); pc=f.loc[idx,'permco'].astype(int).to_numpy(); loc={int(x):i for i,x in enumerate(f.permco)}; threshold=float(NULL.loc[v,'q95'])
 for direction in ['positive','negative']:
  scores=u.loc[u.assigned_direction.eq(direction)].set_index('permco').net_score.abs().to_dict()
  for seed,ss in scores.items():
   if int(seed) not in loc: continue
   sims=np.einsum('ij,j->i',X[idx],X[loc[int(seed)]]); accepted=0
   for j in np.argsort(-sims):
    cand=int(pc[j]); sim=float(sims[j])
    if cand in all_seeds or sim<=threshold: continue
    rows.append({'vintage':v,'direction':direction,'seed_permco':int(seed),'candidate_permco':cand,'similarity':sim,'weighted_score':sim*float(ss)}); accepted+=1
    if accepted==contract['neighbors_per_seed']: break
prop=pd.DataFrame(rows); prop.to_parquet(OUT/'bidirectional_seed_nominations.parquet',index=False)
display(prop.groupby(['vintage','direction']).agg(seeds=('seed_permco','nunique'),candidates=('candidate_permco','nunique')).reset_index())''')
M('''## 3. Freeze direct, embedding, and hybrid portfolios

Long and short legs are independently equal weighted. Each long–short portfolio therefore has +100% gross long exposure and −100% gross short exposure. The hybrid long-only portfolio assigns 50% to each positive sleeve.''')
C('''members=[]; score_panels=[]
for v in sorted(Q.vintage.unique()):
 u=Q[Q.vintage.eq(v)].copy(); f,X=frames[v]; embedded=set(f.permco.astype(int)); eligible=u.positive_seed|u.negative_seed; pos=u[eligible&(u.net_score>0)&u.permco.isin(embedded)].copy(); neg=u[eligible&(u.net_score<0)&u.permco.isin(embedded)].copy(); np_,nn=len(pos),len(neg)
 for leg,g in [('long',pos),('short',neg)]:
  for r in g.itertuples(): members.append({'vintage':v,'method':'llm','leg':leg,'permco':int(r.permco),'signal':abs(r.net_score),'weight':(1/len(g))*(1 if leg=='long' else -1)})
 pv=prop[prop.vintage.eq(v)]; wide=pv.pivot_table(index='candidate_permco',columns='direction',values='weighted_score',aggfunc='max',fill_value=0).reset_index()
 for col in ['positive','negative']:
  if col not in wide: wide[col]=0.0
 wide['embedding_signed']=wide.positive-wide.negative
 elong=wide[wide.embedding_signed>0].nlargest(np_,'embedding_signed'); eshort=wide[wide.embedding_signed<0].nsmallest(nn,'embedding_signed'); assert len(elong)==np_ and len(eshort)==nn
 for leg,g in [('long',elong),('short',eshort)]:
  for r in g.itertuples(): members.append({'vintage':v,'method':'embedding','leg':leg,'permco':int(r.candidate_permco),'signal':r.embedding_signed,'weight':(1/len(g))*(1 if leg=='long' else -1)})
 panel=u[['permco','net_score']].merge(wide[['candidate_permco','embedding_signed']],left_on='permco',right_on='candidate_permco',how='left'); panel['embedding_signed']=panel.embedding_signed.fillna(0); panel['hybrid_score']=(z(panel.net_score)+z(panel.embedding_signed))/2; panel['vintage']=v; score_panels.append(panel[['vintage','permco','net_score','embedding_signed','hybrid_score']])
 hlong=panel.nlargest(np_,'hybrid_score'); hshort=panel.nsmallest(nn,'hybrid_score')
 for leg,g in [('long',hlong),('short',hshort)]:
  for r in g.itertuples(): members.append({'vintage':v,'method':'hybrid','leg':leg,'permco':int(r.permco),'signal':r.hybrid_score,'weight':(1/len(g))*(1 if leg=='long' else -1)})
 for r in pos.itertuples(): members.append({'vintage':v,'method':'hybrid_long_only','leg':'long_text','permco':int(r.permco),'signal':r.net_score,'weight':.5/np_})
 for r in elong.itertuples(): members.append({'vintage':v,'method':'hybrid_long_only','leg':'long_embedding','permco':int(r.candidate_permco),'signal':r.embedding_signed,'weight':.5/np_})
members=pd.DataFrame(members); scores=pd.concat(score_panels,ignore_index=True); members.to_parquet(OUT/'frozen_portfolio_membership.parquet',index=False); scores.to_parquet(OUT/'signed_signal_panel.parquet',index=False)
display(members.groupby(['vintage','method','leg']).agg(firms=('permco','nunique'),weight=('weight','sum')).reset_index())''')
M('''## 4. Evaluate three-month returns and leg attribution

Returns use exactly the same frozen months and firm-level buy-and-hold construction as 12G. Gross long, gross short, and net long–short returns are retained separately.''')
C('''design=json.loads((CONFIG/'sp500_experiment_design_v1.json').read_text()); windows=pd.DataFrame([{'vintage':v['label'],'month':pd.Period(m,'M')} for v in design['vintages'] for m in v['forward_months']]); r=pd.read_parquet(RET); r['permco']=pd.to_numeric(r.permco,errors='coerce').astype('Int64');r['month']=pd.to_datetime(r.eom).dt.to_period('M');r['eom_ret']=pd.to_numeric(r.eom_ret,errors='coerce');r=r.dropna(subset=['permco','month','eom_ret']).drop_duplicates(['permco','month'])
need=members[['vintage','permco']].drop_duplicates().merge(windows,on='vintage').merge(r[['permco','month','eom_ret']],on=['permco','month'],how='left'); bhr=need.groupby(['vintage','permco'],as_index=False).agg(months=('eom_ret','count'),buy_hold_return=('eom_ret',lambda x:np.prod(1+x.dropna())-1)); bhr['complete']=bhr.months.eq(3)
x=members.merge(bhr,on=['vintage','permco'],how='left')
def leg_summary(g):
 z=g[g.complete]; target=float(g.weight.sum())
 return pd.Series({'selected':g.permco.nunique(),'observed':z.permco.nunique(),'target_weight':target,'weighted_return':target*z.buy_hold_return.mean()})
legs=x.groupby(['vintage','method','leg'],as_index=False).apply(leg_summary,include_groups=False).reset_index(drop=True)
results=[]
for (v,m),g in legs.groupby(['vintage','method']):
 if m=='hybrid_long_only': ret=g.weighted_return.sum(); longret=ret; shortret=np.nan
 else:
  longret=float(g.loc[g.leg.eq('long'),'weighted_return'].sum()); shortret=float(g.loc[g.leg.eq('short'),'weighted_return'].sum()); ret=longret+shortret
 results.append({'vintage':v,'method':m,'long_contribution':longret,'short_contribution':shortret,'portfolio_return':ret})
results=pd.DataFrame(results); legs.to_parquet(OUT/'leg_attribution.parquet',index=False);results.to_parquet(OUT/'portfolio_results.parquet',index=False)
display(legs);display(results.pivot(index='vintage',columns='method',values='portfolio_return'))''')
M('''## 5. Matched random long–short placebos and decision gates

Placebos draw disjoint long and short sets from the embedded, non-seed universe with the same quarterly leg sizes. This tests whether each long–short result is unusual relative to equally concentrated random portfolios.''')
C('''rng=np.random.default_rng(contract['random_seed']); prs=[]
for v in sorted(Q.vintage.unique()):
 u=Q[Q.vintage.eq(v)]; f,_=frames[v]; seeds=set(u.loc[u.positive_seed|u.negative_seed,'permco']); complete=set(bhr.loc[bhr.vintage.eq(v)&bhr.complete,'permco'].astype(int)); pool=np.array(sorted((set(f.permco.astype(int))&set(u.permco.astype(int))&complete)-seeds)); nl=int(members.query("vintage==@v and method=='llm' and leg=='long'").permco.nunique()); ns=int(members.query("vintage==@v and method=='llm' and leg=='short'").permco.nunique()); br=bhr[bhr.vintage.eq(v)&bhr.complete].set_index('permco').buy_hold_return
 for d in range(100):
  pick=rng.choice(pool,nl+ns,replace=False); lr=br.reindex(pick[:nl]).mean(); sr=br.reindex(pick[nl:]).mean(); prs.append({'vintage':v,'draw':d,'placebo_long_short':lr-sr})
placebo=pd.DataFrame(prs); actual=results[results.method.isin(['llm','embedding','hybrid'])]
summ=[]
for zrow in actual.itertuples():
 q=placebo[placebo.vintage.eq(zrow.vintage)].placebo_long_short.dropna(); o=zrow.portfolio_return; summ.append({'vintage':zrow.vintage,'method':zrow.method,'return':o,'placebo_mean':q.mean(),'percentile':(q<o).mean(),'upper_tail_p':(1+(q>=o).sum())/(1+len(q)),'two_sided_p':min(1,2*min((1+(q>=o).sum())/(1+len(q)),(1+(q<=o).sum())/(1+len(q))))})
ps=pd.DataFrame(summ); placebo.to_parquet(OUT/'matched_long_short_placebos.parquet',index=False);ps.to_parquet(OUT/'placebo_inference.parquet',index=False)
coverage=x.groupby(['vintage','method'],as_index=False).agg(selected=('permco','size'),observed=('complete','sum'));coverage['coverage']=coverage.observed/coverage.selected
overlap=max(len(set(g.loc[g.leg.eq('long'),'permco'])&set(g.loc[g.leg.eq('short'),'permco'])) for _,g in members[members.method.isin(['llm','embedding','hybrid'])].groupby(['vintage','method']))
gates=pd.DataFrame([{'gate':'four_vintages','value':results.vintage.nunique(),'threshold':4,'passed':results.vintage.nunique()==4},{'gate':'four_methods','value':results.method.nunique(),'threshold':4,'passed':results.method.nunique()==4},{'gate':'zero_long_short_overlap','value':overlap,'threshold':0,'passed':overlap==0},{'gate':'complete_return_coverage','value':coverage.coverage.min(),'threshold':.9,'passed':coverage.coverage.min()>=.9},{'gate':'one_hundred_placebos','value':placebo.groupby('vintage').draw.nunique().min(),'threshold':100,'passed':placebo.groupby('vintage').draw.nunique().min()==100}]);gates.to_parquet(OUT/'decision_gates.parquet',index=False)
display(ps);display(gates);assert gates.passed.all()''')
M('''## 6. Summary and manifest

Interpret long and short legs separately. Strong net performance caused only by short-leg losses has a different economic meaning from successful positive-theme discovery.''')
C('''summary=results.groupby('method',as_index=False).agg(mean_return=('portfolio_return','mean'),median_return=('portfolio_return','median'),positive_quarters=('portfolio_return',lambda x:int((x>0).sum())));display(summary);summary.to_parquet(OUT/'cross_vintage_summary.parquet',index=False)
manifest={'stage':'long_short_hybrid_completion_v1','contract_sha256':sha(contract_path),'outcomes_previously_viewed':True,'parameters_tuned_after_12g':False,'all_gates_passed':bool(gates.passed.all()),'artifacts':[p.name for p in sorted(OUT.glob('*')) if p.is_file()]};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n');print('Saved:',OUT)''')
nbf.write(n,R/'12H_Long_Short_and_Hybrid_Evaluation.ipynb')
