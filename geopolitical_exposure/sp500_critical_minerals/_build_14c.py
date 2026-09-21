import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'14C_Exposure_Scores_vs_Thematic_Beta.ipynb'
nb=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}});cells=[]
def md(x):cells.append(nbf.v4.new_markdown_cell(x))
def code(x):cells.append(nbf.v4.new_code_cell(x))

md("""# 14C — Do exposure scores predict realized thematic beta?

## Primary hypothesis

Holdings-embedding exposure predicts forward critical-minerals thematic sensitivity beyond direct SEC-text exposure, historical thematic beta, and conventional firm characteristics.

This notebook opens the frozen beta outcome only after saving the prediction design. Results are reported separately by 2024 vintage because the forward 252-day windows overlap and must not be presented as independent time-series replications.""")

md("""## 1. Setup and preregistered model design""")
code("""from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path.cwd()
if ROOT.name!='sp500_critical_minerals':
 c=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
 if c.exists():ROOT=c
PROJECT=ROOT.parents[1];OUT=ROOT/'thematic_beta_prediction_v1';MANIFESTS=OUT/'manifests';OUT.mkdir(exist_ok=True);MANIFESTS.mkdir(exist_ok=True)
CONTROLS=['beta_60m_paper','log_market_equity','log_book_to_market','at_gr1_paper','gp_at_paper','momentum_12_1']
DESIGN={'version':'thematic_beta_prediction_v1','outcome':'forward broad_cm_equity_ex_market beta','historical_predictor':'historical beta for same factor',
 'controls':CONTROLS,'signal_columns':{'text':'net_score','embedding':'embedding_signed','hybrid':'hybrid_score'},
 'primary_nested_comparison':'baseline_plus_text_plus_embedding minus baseline_plus_text',
 'outer_cv':'five-fold shuffled within vintage','outer_seed':20240906,'inner_model':'RidgeCV','alphas':[0.01,0.1,1,10,100,1000],
 'preprocessing':'median imputation and standardization fitted inside each outer training fold',
 'permutation_placebo':'1000 within-vintage permutations of embedding score; partial-correlation statistic',
 'report_vintages_separately':True,'pool_overlapping_windows':False,'industry_primary':'omitted because frozen SIC coverage is only 25%; required robustness later',
 'portfolio_returns_accessed':False,'scenario_13a_scores_accessed':False}
dp=MANIFESTS/'prediction_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2,sort_keys=True)+'\\n')
design_hash=hashlib.sha256(dp.read_bytes()).hexdigest();print({'design_sha256':design_hash,'outcome_opened':False})""")

md("""## 2. Load and align frozen inputs""")
code("""signal_path=ROOT/'long_short_hybrid_v1'/'signed_signal_panel.parquet';beta_path=ROOT/'thematic_beta_v1'/'firm_thematic_betas.parquet'
input_paths=[signal_path,beta_path]+[PROJECT/f'remaining_benchmarks/return_comovement/characteristics/2024q{q}.parquet' for q in range(1,5)]
audit=pd.DataFrame([{'path':str(p),'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None} for p in input_paths])
display(audit);assert audit.exists.all()
signals=pd.read_parquet(signal_path)
betas=pd.read_parquet(beta_path);betas=betas[betas.factor.eq('broad_cm_equity_ex_market')]
wide=betas.pivot_table(index=['vintage','permco'],columns='window',values='thematic_beta').reset_index().rename(columns={'historical':'historical_beta','forward':'forward_beta'})
chars=[]
for q in range(1,5):
 d=pd.read_parquet(PROJECT/f'remaining_benchmarks/return_comovement/characteristics/2024q{q}.parquet');d['vintage']=f'2024q{q}';chars.append(d)
chars=pd.concat(chars,ignore_index=True)
panel=(signals.merge(wide,on=['vintage','permco'],how='inner',validate='one_to_one').merge(chars,on=['vintage','permco'],how='inner',validate='one_to_one'))
panel=panel.dropna(subset=['historical_beta','forward_beta']).reset_index(drop=True)
panel.to_parquet(OUT/'model_panel.parquet',index=False)
display(panel.groupby('vintage').agg(firms=('permco','nunique'),text_nonzero=('net_score',lambda x:int(x.ne(0).sum())),embedding_nonzero=('embedding_signed',lambda x:int(x.ne(0).sum())),forward_beta=('forward_beta','count')).reset_index())""")

md("""## 3. Univariate rank relationships

These are descriptive and do not replace the controlled nested-model test.""")
code("""rows=[]
for vintage,g in panel.groupby('vintage'):
 for horizon in ['historical_beta','forward_beta']:
  for method,column in DESIGN['signal_columns'].items():
   rho,p=spearmanr(g[column],g[horizon],nan_policy='omit');rows.append({'vintage':vintage,'beta_window':horizon,'method':method,'spearman_rho':rho,'p_value':p,'firms':len(g)})
rank_results=pd.DataFrame(rows);rank_results.to_parquet(OUT/'score_beta_rank_correlations.parquet',index=False)
display(rank_results[rank_results.beta_window.eq('forward_beta')].round(4))""")

md("""## 4. Nested out-of-sample model comparison

Every specification uses the same firm sample and the same outer folds within each vintage. Alpha selection occurs only inside the outer training sample.""")
code("""BASE=['historical_beta']+CONTROLS
MODELS={'historical_beta_only':['historical_beta'],'characteristics_only':CONTROLS,'baseline':BASE,
 'baseline_plus_text':BASE+['net_score'],'baseline_plus_embedding':BASE+['embedding_signed'],
 'baseline_plus_text_embedding':BASE+['net_score','embedding_signed'],'baseline_plus_hybrid':BASE+['hybrid_score']}
pred_rows=[];summary=[]
for vintage,g0 in panel.groupby('vintage',sort=True):
 g=g0.reset_index(drop=True);folds=list(KFold(n_splits=5,shuffle=True,random_state=DESIGN['outer_seed']+int(vintage[-1])).split(g))
 for model,cols in MODELS.items():
  pred=np.full(len(g),np.nan);alphas=[]
  for fold,(train,test) in enumerate(folds):
   pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',RidgeCV(alphas=DESIGN['alphas'],cv=5))])
   pipe.fit(g.loc[train,cols],g.loc[train,'forward_beta']);pred[test]=pipe.predict(g.loc[test,cols]);alphas.append(pipe.named_steps['ridge'].alpha_)
   for idx in test:pred_rows.append({'vintage':vintage,'permco':g.loc[idx,'permco'],'model':model,'fold':fold,'actual':g.loc[idx,'forward_beta'],'prediction':pred[idx]})
  summary.append({'vintage':vintage,'model':model,'firms':len(g),'oos_r2':r2_score(g.forward_beta,pred),'prediction_correlation':np.corrcoef(g.forward_beta,pred)[0,1],'median_selected_alpha':float(np.median(alphas))})
predictions=pd.DataFrame(pred_rows);cv=pd.DataFrame(summary)
predictions.to_parquet(OUT/'outer_fold_predictions.parquet',index=False);cv.to_parquet(OUT/'nested_model_oos_results.parquet',index=False)
display(cv.pivot(index='model',columns='vintage',values='oos_r2').assign(mean=lambda x:x.mean(axis=1)).sort_values('mean',ascending=False).round(4))""")

md("""## 5. Incremental embedding contribution""")
code("""wide_cv=cv.pivot(index='vintage',columns='model',values='oos_r2')
incremental=pd.DataFrame({'vintage':wide_cv.index,
 'embedding_beyond_baseline':wide_cv.baseline_plus_embedding-wide_cv.baseline,
 'text_beyond_baseline':wide_cv.baseline_plus_text-wide_cv.baseline,
 'embedding_beyond_text':wide_cv.baseline_plus_text_embedding-wide_cv.baseline_plus_text,
 'hybrid_beyond_baseline':wide_cv.baseline_plus_hybrid-wide_cv.baseline}).reset_index(drop=True)
incremental.to_parquet(OUT/'incremental_oos_r2.parquet',index=False);display(incremental.round(4));display(incremental.drop(columns='vintage').agg(['mean','median','min','max']).round(4))""")

md("""## 6. Embedding permutation placebo

Within each vintage, residualize forward beta and embedding exposure against the baseline-plus-text predictors. Compare their observed residual correlation with 1,000 permutations of the embedding residual. This tests incremental association without reusing the portfolio-return exercise.""")
code("""rng=np.random.default_rng(DESIGN['outer_seed']);prows=[];observed=[]
for vintage,g0 in panel.groupby('vintage',sort=True):
 g=g0.reset_index(drop=True);Xcols=BASE+['net_score'];Xs=StandardScaler().fit_transform(g[Xcols].to_numpy(float));X=np.column_stack([np.ones(len(g)),Xs])
 y=g.forward_beta.to_numpy(float);e=g.embedding_signed.to_numpy(float)
 yr=y-X@np.linalg.lstsq(X,y,rcond=None)[0];er=e-X@np.linalg.lstsq(X,e,rcond=None)[0]
 stat=np.corrcoef(yr,er)[0,1];null=[]
 for draw in range(1000):null.append(np.corrcoef(yr,rng.permutation(er))[0,1])
 null=np.asarray(null);p=(1+(np.abs(null)>=abs(stat)).sum())/(len(null)+1)
 observed.append({'vintage':vintage,'observed_partial_correlation':stat,'placebo_mean':null.mean(),'placebo_sd':null.std(ddof=1),'two_sided_p':p})
 prows.extend({'vintage':vintage,'draw':i,'partial_correlation':v} for i,v in enumerate(null))
placebos=pd.DataFrame(prows);placebo_summary=pd.DataFrame(observed)
placebos.to_parquet(OUT/'embedding_permutation_placebos.parquet',index=False);placebo_summary.to_parquet(OUT/'embedding_permutation_summary.parquet',index=False)
display(placebo_summary.round(4))""")

md("""## 7. Exposure ordering diagnostics

For embedding and hybrid scores, use within-vintage quintiles. For the sparse text score, preserve the economically meaningful negative/zero/positive groups instead of arbitrarily breaking tied zero scores.""")
code("""rows=[]
for vintage,g0 in panel.groupby('vintage'):
 g=g0.copy()
 g['text_bucket']=np.select([g.net_score.lt(0),g.net_score.gt(0)],['negative','positive'],default='zero')
 for bucket,z in g.groupby('text_bucket'):
  rows.append({'vintage':vintage,'method':'text','bucket':bucket,'firms':len(z),'mean_score':z.net_score.mean(),'mean_forward_beta':z.forward_beta.mean()})
 for method,col in [('embedding','embedding_signed'),('hybrid','hybrid_score')]:
  # Deterministically break tied scores by PERMCO for this descriptive
  # equal-count diagnostic; the underlying score itself is not changed.
  g=g.sort_values([col,'permco'],kind='mergesort').copy()
  g['bucket']=pd.qcut(np.arange(len(g)),5,labels=['Q1','Q2','Q3','Q4','Q5'])
  for bucket,z in g.groupby('bucket',observed=True):rows.append({'vintage':vintage,'method':method,'bucket':str(bucket),'firms':len(z),'mean_score':z[col].mean(),'mean_forward_beta':z.forward_beta.mean()})
ordering=pd.DataFrame(rows);ordering.to_parquet(OUT/'exposure_bucket_beta_ordering.parquet',index=False)
display(ordering.groupby(['method','bucket']).agg(mean_beta=('mean_forward_beta','mean'),mean_firms=('firms','mean')).reset_index().round(4))""")

md("""## 8. Decision table and manifest

Evidence for incremental embedding information requires positive mean incremental OOS \(R^2\), improvement in at least three of four vintages, and a same-signed partial correlation in at least three vintages. Statistical significance is reported, not required mechanically in this four-vintage pilot.""")
code("""delta=incremental.embedding_beyond_text
signs=np.sign(placebo_summary.observed_partial_correlation.replace(0,np.nan)).dropna()
same_sign=max((signs>0).sum(),(signs<0).sum()) if len(signs) else 0
gates=pd.DataFrame([
 {'gate':'positive_mean_incremental_oos_r2','value':delta.mean(),'threshold':0,'passed':delta.mean()>0},
 {'gate':'positive_incremental_r2_in_three_vintages','value':int((delta>0).sum()),'threshold':3,'passed':(delta>0).sum()>=3},
 {'gate':'partial_correlation_same_sign_three_vintages','value':int(same_sign),'threshold':3,'passed':same_sign>=3},
 {'gate':'portfolio_results_unopened','value':0,'threshold':0,'passed':True}])
gates.to_parquet(MANIFESTS/'decision_gates.parquet',index=False);display(gates)
artifacts=[dp,OUT/'model_panel.parquet',OUT/'score_beta_rank_correlations.parquet',OUT/'nested_model_oos_results.parquet',OUT/'incremental_oos_r2.parquet',OUT/'embedding_permutation_summary.parquet',OUT/'exposure_bucket_beta_ordering.parquet',MANIFESTS/'decision_gates.parquet']
inventory=pd.DataFrame([{'artifact':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]);inventory.to_parquet(MANIFESTS/'artifact_inventory.parquet',index=False)
manifest={'stage':'exposure_scores_vs_thematic_beta_v1','design_sha256':design_hash,'panel_rows':len(panel),'vintages':sorted(panel.vintage.unique()),'mean_embedding_beyond_text_oos_r2':float(delta.mean()),'all_primary_gates_passed':bool(gates.passed.all()),'portfolio_results_accessed':False}
(MANIFESTS/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n');print('Saved:',OUT)""")

md("""## Interpretation boundary

A positive result means the frozen score predicts sensitivity to the traded critical-minerals equity theme. It does not prove a physical supply-chain relationship, causal exposure, or abnormal-return alpha. A negative result is also informative: it would indicate that holdings proximity does not add thematic-beta information beyond direct text, persistent beta, and conventional characteristics in this pilot.""")

nb['cells']=cells;nbf.write(nb,OUT);print(OUT)
