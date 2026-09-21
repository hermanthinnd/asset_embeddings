from pathlib import Path
import nbformat as nbf
R=Path(__file__).resolve().parent;n=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
M=lambda x:n.cells.append(nbf.v4.new_markdown_cell(x));C=lambda x:n.cells.append(nbf.v4.new_code_cell(x))
M('''# 12I — Text–Embedding Consensus Portfolios

This exploratory robustness test asks whether direct textual exposure becomes more useful when independently corroborated by holdings-embedding proximity. A target firm cannot validate itself: its embedding support is calculated only from other directionally resolved LLM seeds. This stage was requested after 12G and 12H outcomes were viewed and is not presented as a blinded confirmatory result.''')
M('''## 1. Setup and frozen consensus contract

The 12F similarity threshold and original quarterly text scores remain unchanged. Thin consensus legs are reported rather than repaired through outcome-informed threshold changes.''')
C('''from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
ROOT=Path.cwd().resolve()
if ROOT.name!='sp500_critical_minerals':ROOT=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
PROJECT=ROOT.parents[1];CONFIG=ROOT/'config';OUT=ROOT/'consensus_agreement_v1';OUT.mkdir(exist_ok=True)
contract_path=CONFIG/'consensus_agreement_v1.json';contract=json.loads(contract_path.read_text());Q=pd.read_parquet(ROOT/'quarterly_exposure_v1/quarterly_firm_exposure_panel.parquet');NULL=pd.read_parquet(ROOT/'embedding_propagation_v1/random_pair_null_summary.parquet').set_index('vintage');RET=ROOT.parent/'quarterly_sec_pilot'/'forward_performance_2025_2026'/'data'/'processed'/'crsp_firm_monthly_through_cutoff.parquet';dims=[f'embedding_{i:03d}' for i in range(1,11)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
print(contract['status'])''')
M('''## 2. Leave-one-seed-out embedding corroboration

For every text-labeled target, the strongest positive support and strongest negative support are calculated from other seeds above the frozen q95 cosine threshold. Their difference is the signed embedding-support score. Agreement requires matching nonzero signs.''')
C('''support=[]
for v in sorted(Q.vintage.unique()):
 u=Q[Q.vintage.eq(v)].copy();eligible=u.positive_seed|u.negative_seed;s=u[eligible&u.net_score.ne(0)].copy();p=PROJECT/'full_replication_80q'/'embeddings'/v/'rs_l_min'/'asset_embeddings_normalized_labeled.parquet';f=pd.read_parquet(p)[['permco']+dims].drop_duplicates('permco');X=f[dims].to_numpy(float);norm=np.linalg.norm(X,axis=1);keep=np.isfinite(X).all(1)&(norm>0);f=f.loc[keep].reset_index(drop=True);X=X[keep]/norm[keep,None];loc={int(x):i for i,x in enumerate(f.permco)};threshold=float(NULL.loc[v,'q95'])
 for target in s.itertuples():
  if int(target.permco) not in loc:continue
  pos=[];neg=[]
  for seed in s.itertuples():
   if seed.permco==target.permco or int(seed.permco) not in loc:continue
   sim=float(np.dot(X[loc[int(target.permco)]],X[loc[int(seed.permco)]]))
   if sim<=threshold:continue
   val=sim*abs(float(seed.net_score));(pos if seed.net_score>0 else neg).append(val)
  ps=max(pos) if pos else 0.;ns=max(neg) if neg else 0.;es=ps-ns;td=np.sign(target.net_score);ed=np.sign(es)
  support.append({'vintage':v,'permco':int(target.permco),'net_text_score':float(target.net_score),'text_direction':int(td),'positive_embedding_support':ps,'negative_embedding_support':ns,'embedding_support':es,'embedding_direction':int(ed),'consensus':bool(ed!=0 and ed==td),'threshold':threshold})
support=pd.DataFrame(support);support.to_parquet(OUT/'leave_one_seed_out_support.parquet',index=False)
summary=support.groupby('vintage',as_index=False).agg(labeled_embedded_firms=('permco','nunique'),consensus_firms=('consensus','sum'),consensus_long=('consensus',lambda x:int(((x)&(support.loc[x.index,'text_direction']>0)).sum())),consensus_short=('consensus',lambda x:int(((x)&(support.loc[x.index,'text_direction']<0)).sum())),no_embedding_support=('embedding_direction',lambda x:int((x==0).sum())))
display(summary);summary.to_parquet(OUT/'consensus_coverage.parquet',index=False)''')
M('''## 3. Freeze consensus and matched text-only portfolios

The matched text comparator uses exactly the same long- and short-leg counts as consensus, selecting the strongest absolute text scores within direction. This separates the value of agreement from the mechanical effect of concentration.''')
C('''members=[]
for v,g in support.groupby('vintage'):
 con=g[g.consensus];nl=int((con.text_direction>0).sum());ns=int((con.text_direction<0).sum())
 sets={'consensus':(con[con.text_direction>0],con[con.text_direction<0]),'matched_text':(g[g.text_direction>0].nlargest(nl,'net_text_score'),g[g.text_direction<0].nsmallest(ns,'net_text_score'))}
 for method,(lg,sg) in sets.items():
  for leg,z,sign in [('long',lg,1),('short',sg,-1)]:
   for r in z.itertuples():members.append({'vintage':v,'method':method,'leg':leg,'permco':r.permco,'weight':sign/len(z),'text_score':r.net_text_score,'embedding_support':r.embedding_support})
members=pd.DataFrame(members);members.to_parquet(OUT/'frozen_consensus_membership.parquet',index=False)
display(members.groupby(['vintage','method','leg']).agg(firms=('permco','nunique'),weight=('weight','sum')).reset_index())''')
M('''## 4. Apply the frozen three-month returns

Firm returns are compounded first. Equal weights are re-normalized over complete-return firms within each leg. Long-only and dollar-neutral long–short results are both retained.''')
C('''design=json.loads((CONFIG/'sp500_experiment_design_v1.json').read_text());windows=pd.DataFrame([{'vintage':v['label'],'month':pd.Period(m,'M')} for v in design['vintages'] for m in v['forward_months']]);r=pd.read_parquet(RET);r['permco']=pd.to_numeric(r.permco,errors='coerce').astype('Int64');r['month']=pd.to_datetime(r.eom).dt.to_period('M');r['eom_ret']=pd.to_numeric(r.eom_ret,errors='coerce');r=r.dropna(subset=['permco','month','eom_ret']).drop_duplicates(['permco','month']);need=members[['vintage','permco']].drop_duplicates().merge(windows,on='vintage').merge(r[['permco','month','eom_ret']],on=['permco','month'],how='left');bhr=need.groupby(['vintage','permco'],as_index=False).agg(months=('eom_ret','count'),buy_hold_return=('eom_ret',lambda x:np.prod(1+x.dropna())-1));bhr['complete']=bhr.months.eq(3);x=members.merge(bhr,on=['vintage','permco'],how='left')
def ls(g):
 l=g[(g.leg=='long')&g.complete].buy_hold_return;s=g[(g.leg=='short')&g.complete].buy_hold_return;return pd.Series({'long_firms':len(l),'short_firms':len(s),'long_only_return':l.mean(),'short_underlying_return':s.mean(),'long_short_return':l.mean()-s.mean()})
results=x.groupby(['vintage','method'],as_index=False).apply(ls,include_groups=False).reset_index(drop=True);results.to_parquet(OUT/'consensus_portfolio_results.parquet',index=False);display(results)''')
M('''## 5. Matched labeled-firm placebos

Within each text direction, 100 random portfolios use the same consensus leg sizes. These placebos test whether agreement selects unusually successful firms relative to other firms already labeled in the same direction.''')
C('''rng=np.random.default_rng(contract['random_seed']);rows=[]
for v,g in support.groupby('vintage'):
 con=g[g.consensus];nl=int((con.text_direction>0).sum());ns=int((con.text_direction<0).sum());br=bhr[(bhr.vintage==v)&bhr.complete].set_index('permco').buy_hold_return;lp=np.array(sorted(set(g.loc[g.text_direction>0,'permco'])&set(br.index)));sp=np.array(sorted(set(g.loc[g.text_direction<0,'permco'])&set(br.index)))
 for d in range(100):
  l=rng.choice(lp,nl,replace=False);s=rng.choice(sp,ns,replace=False);rows.append({'vintage':v,'draw':d,'long_only_return':br.reindex(l).mean(),'long_short_return':br.reindex(l).mean()-br.reindex(s).mean()})
placebo=pd.DataFrame(rows);placebo.to_parquet(OUT/'matched_consensus_placebos.parquet',index=False);actual=results[results.method.eq('consensus')];inf=[]
for a in actual.itertuples():
 for outcome in ['long_only_return','long_short_return']:
  z=placebo.loc[placebo.vintage.eq(a.vintage),outcome];o=getattr(a,outcome);inf.append({'vintage':a.vintage,'outcome':outcome,'actual':o,'placebo_mean':z.mean(),'percentile':(z<o).mean(),'upper_tail_p':(1+(z>=o).sum())/(1+len(z)),'two_sided_p':min(1,2*min((1+(z>=o).sum())/(1+len(z)),(1+(z<=o).sum())/(1+len(z))))})
inference=pd.DataFrame(inf);inference.to_parquet(OUT/'consensus_placebo_inference.parquet',index=False);display(inference)''')
M('''## 6. Decision gates and interpretation boundary

The minimum long-leg size gate is descriptive: two firms are enough to execute the requested robustness check, but not enough for a diversified investment claim.''')
C('''overlap=max(len(set(g.loc[g.leg.eq('long'),'permco'])&set(g.loc[g.leg.eq('short'),'permco'])) for _,g in members.groupby(['vintage','method']));gates=pd.DataFrame([{'gate':'four_vintages','value':results.vintage.nunique(),'threshold':4,'passed':results.vintage.nunique()==4},{'gate':'zero_leg_overlap','value':overlap,'threshold':0,'passed':overlap==0},{'gate':'minimum_consensus_long_firms','value':summary.consensus_long.min(),'threshold':2,'passed':summary.consensus_long.min()>=2},{'gate':'minimum_consensus_short_firms','value':summary.consensus_short.min(),'threshold':10,'passed':summary.consensus_short.min()>=10},{'gate':'one_hundred_placebos','value':placebo.groupby('vintage').draw.nunique().min(),'threshold':100,'passed':placebo.groupby('vintage').draw.nunique().min()==100}]);gates.to_parquet(OUT/'decision_gates.parquet',index=False);display(gates);assert gates.passed.all();manifest={'stage':'text_embedding_consensus_v1','contract_sha256':sha(contract_path),'outcomes_previously_viewed':True,'all_gates_passed':bool(gates.passed.all()),'minimum_long_firms':int(summary.consensus_long.min()),'minimum_short_firms':int(summary.consensus_short.min())};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n');print('Saved:',OUT)''')
nbf.write(n,R/'12I_Text_Embedding_Consensus_Portfolios.ipynb')
