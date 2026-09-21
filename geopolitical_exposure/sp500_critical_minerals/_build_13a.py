from pathlib import Path
import nbformat as nbf
R=Path(__file__).resolve().parent;n=nbf.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}})
M=lambda x:n.cells.append(nbf.v4.new_markdown_cell(x));C=lambda x:n.cells.append(nbf.v4.new_code_cell(x))
M('''# 13A — Scenario-Specific Critical-Minerals Exposure

This notebook replaces a single compound theme with five separately defined economic shocks. It reclassifies only the 204 previously validated filing relationships, constructs quarterly text, embedding, and hybrid exposure scores, and creates an outcome-blind validation sample. It does not load returns, commodity factors, or validation labels.

Run sequentially. The API stage is checkpointed and disabled by default.''')
M('''## 1. Setup and immutable contracts

The prompt, strict output schema, score formula, five counterfactuals, and model are frozen before any new downstream evidence is accessed.''')
C('''from pathlib import Path
import os,json,hashlib,time,numpy as np,pandas as pd
from IPython.display import display
ROOT=Path.cwd().resolve()
if ROOT.name!='sp500_critical_minerals':ROOT=Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
PROJECT=ROOT.parents[1];CONFIG=ROOT/'config';SRC=ROOT/'resolved_extraction_v2_1';QROOT=ROOT/'quarterly_exposure_v1';OUT=ROOT/'scenario_exposure_v1';RESP=OUT/'responses';MAN=OUT/'manifests';RESP.mkdir(parents=True,exist_ok=True);MAN.mkdir(exist_ok=True)
design_path=CONFIG/'scenario_exposure_design_v1.json';prompt_path=CONFIG/'scenario_reclassification_prompt_v1.md';schema_path=CONFIG/'scenario_reclassification_schema_v1.json'
design=json.loads(design_path.read_text());prompt=prompt_path.read_text();schema=json.loads(schema_path.read_text());SCENARIOS=[x['id'] for x in design['scenarios']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
print({'model':design['api_model'],'scenarios':SCENARIOS,'outcomes_loaded':False})''')
M('''## 2. Freeze relationship identifiers and document payloads

Each original evidence relationship receives a deterministic identifier. Calls are grouped by document to reduce cost while retaining relationship-level output.''')
C('''rel=pd.read_parquet(SRC/'final_validated_document_exposures.parquet').copy()
def rid(r):
 raw='|'.join(str(r[x]) for x in ['document_id','firm_id','mechanism','evidence_passage_id','evidence_quote']);return hashlib.sha256(raw.encode()).hexdigest()[:24]
rel['relationship_id']=rel.apply(rid,axis=1);assert not rel.relationship_id.duplicated().any()
fields=['relationship_id','mechanism','supply_chain_position','direction','directness','materiality','disclosure_status','time_horizon','minerals','exposure_strength','confidence','evidence_quote','causal_chain','counterevidence','netting_note']
records=[]
for doc,g in rel.groupby('document_id',sort=True):
 relationships=[]
 for _,r in g.iterrows():
  item={k:r[k] for k in fields};item['minerals']=list(item['minerals']) if isinstance(item['minerals'],(list,tuple,np.ndarray)) else [];relationships.append(item)
 records.append({'document_id':str(doc),'firm_id':str(g.firm_id.iloc[0]),'available_date':str(g.available_date.iloc[0]),'relationships':relationships})
queue=pd.DataFrame([{'document_id':r['document_id'],'firm_id':r['firm_id'],'relationships':len(r['relationships']),'approx_input_tokens':len(json.dumps(r))//4,'checkpointed':(RESP/f"{r['document_id']}.json").exists()} for r in records])
queue.to_parquet(MAN/'api_queue.parquet',index=False);rel.to_parquet(OUT/'source_relationships_with_ids.parquet',index=False)
display(queue.agg({'document_id':'count','firm_id':'nunique','relationships':'sum','approx_input_tokens':'sum','checkpointed':'sum'}).rename({'document_id':'documents','firm_id':'firms'}).to_frame('value'))''')
M('''## 3. Contract and queue audit

The API receives exact evidence excerpts rather than complete filings. The cost is therefore much lower than the original 923-document extraction. Review the queue before changing either execution switch.''')
C('''contract=pd.DataFrame([{'relationships':len(rel),'documents':len(records),'firms':rel.firm_id.nunique(),'approx_input_tokens':queue.approx_input_tokens.sum(),'prompt_sha256':sha(prompt_path),'schema_sha256':sha(schema_path),'design_sha256':sha(design_path),'outcomes_loaded':False}]);contract.to_parquet(MAN/'frozen_contract.parquet',index=False);display(contract)
RUN_API=False
AUTHORIZE_FULL_BATCH=False
GUARDED_DOCUMENTS=design['guarded_documents']
print({'api_key_present':bool(os.getenv('OPENAI_API_KEY')),'run_api':RUN_API,'full_batch':AUTHORIZE_FULL_BATCH,'guarded_documents':GUARDED_DOCUMENTS})''')
M('''## 4. Checkpointed structured-output API run

For the first pass, set `RUN_API=True` and leave `AUTHORIZE_FULL_BATCH=False`; this processes ten documents. Inspect Section 5. If the guarded gate passes, set both switches to `True` and rerun this cell. Existing valid checkpoints are skipped.''')
C('''lookup={r['document_id']:r for r in records}
def validate(result,record):
 issues=[]
 if result.get('document_id')!=record['document_id']:issues.append('document_id')
 expected={x['relationship_id'] for x in record['relationships']};got={x.get('relationship_id') for x in result.get('classifications',[])}
 if got!=expected:issues.append('relationship_ids')
 for item in result.get('classifications',[]):
  scenarios=item.get('scenarios',[]);ids=[x.get('scenario_id') for x in scenarios]
  if sorted(ids)!=sorted(SCENARIOS) or len(ids)!=len(set(ids)):issues.append('scenario_set')
  for x in scenarios:
   if x.get('direction')==0 and x.get('intensity')!=0:issues.append('zero_direction_nonzero_intensity')
 return sorted(set(issues))
if RUN_API:
 assert os.getenv('OPENAI_API_KEY'),'Set OPENAI_API_KEY without printing it.'
 from openai import OpenAI
 from tqdm.auto import tqdm
 client=OpenAI(timeout=120,max_retries=2);pending=[r for r in records if not (RESP/f"{r['document_id']}.json").exists()]
 target=pending if AUTHORIZE_FULL_BATCH else pending[:GUARDED_DOCUMENTS]
 for record in tqdm(target,desc='Scenario documents'):
  response=client.responses.create(model=design['api_model'],instructions=prompt,input=json.dumps(record),store=False,reasoning={'effort':'low'},text={'format':{'type':'json_schema','name':'scenario_reclassification_v1','strict':True,'schema':schema}})
  result=json.loads(response.output_text);issues=validate(result,record)
  envelope={'result':result,'valid':not issues,'issues':issues,'model':response.model,'response_id':response.id,'usage':response.usage.model_dump() if response.usage else None,'source_sha256':hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),'prompt_sha256':sha(prompt_path),'schema_sha256':sha(schema_path)}
  (RESP/f"{record['document_id']}.json").write_text(json.dumps(envelope,indent=2,sort_keys=True)+'\\n')
 print('Batch complete. Rerun the audit below.')
else: print('API disabled; no calls made.')''')
M('''## 5. Guarded/full-run audit

Only schema-valid checkpoints with matching prompt, schema, relationship identifiers, and scenario sets count as complete.''')
C('''aud=[]
for record in records:
 p=RESP/f"{record['document_id']}.json"
 if not p.exists():aud.append({'document_id':record['document_id'],'status':'missing','valid':False,'issues':'missing'});continue
 try:
  e=json.loads(p.read_text());issues=validate(e.get('result',{}),record);hash_ok=e.get('prompt_sha256')==sha(prompt_path) and e.get('schema_sha256')==sha(schema_path);valid=bool(e.get('valid')) and not issues and hash_ok;aud.append({'document_id':record['document_id'],'status':'valid' if valid else 'invalid','valid':valid,'issues':','.join(issues),'hashes_match':hash_ok})
 except Exception as ex:aud.append({'document_id':record['document_id'],'status':'unreadable','valid':False,'issues':type(ex).__name__})
aud=pd.DataFrame(aud);aud.to_parquet(MAN/'response_audit.parquet',index=False);display(aud.status.value_counts().rename_axis('status').reset_index(name='documents'))
guarded_ids=set(queue.head(GUARDED_DOCUMENTS).document_id);guarded=aud[aud.document_id.isin(guarded_ids)];gate=pd.DataFrame([{'gate':'guarded_complete_and_valid','value':int(guarded.valid.sum()),'threshold':GUARDED_DOCUMENTS,'passed':len(guarded)==GUARDED_DOCUMENTS and guarded.valid.all()},{'gate':'full_complete_and_valid','value':int(aud.valid.sum()),'threshold':len(records),'passed':len(aud)==len(records) and aud.valid.all()}]);gate.to_parquet(MAN/'api_gates.parquet',index=False);display(gate)''')
M('''## 6. Assemble scenario classifications

Continue only after the full-run gate passes. This cell expands the checkpointed JSON into one row per relationship and scenario and verifies the expected `204 × 5` structure.''')
C('''assert bool(gate.loc[gate.gate.eq('full_complete_and_valid'),'passed'].iloc[0]),'Full API run is incomplete. Complete Sections 4–5 first.'
rows=[]
for record in records:
 e=json.loads((RESP/f"{record['document_id']}.json").read_text())
 for item in e['result']['classifications']:
  for s in item['scenarios']:rows.append({'document_id':record['document_id'],'relationship_id':item['relationship_id'],**s})
classified=pd.DataFrame(rows);assert len(classified)==len(rel)*len(SCENARIOS);assert not classified.duplicated(['relationship_id','scenario_id']).any();classified.to_parquet(OUT/'relationship_scenario_classifications.parquet',index=False)
display(classified.groupby('scenario_id').agg(relationships=('relationship_id','size'),positive=('direction',lambda x:int((x>0).sum())),neutral=('direction',lambda x:int((x==0).sum())),negative=('direction',lambda x:int((x<0).sum()))).reset_index())''')
M('''## 7. Aggregate quarterly scenario-specific text scores

Only relationships already deemed eligible in the frozen quarterly panel contribute. Within firm, the strongest positive and strongest negative component are combined and clipped to `[-1,1]`, preventing repeated disclosure from mechanically increasing exposure.''')
C('''control=pd.read_parquet(QROOT/'quarterly_controlling_exposures.parquet');base=pd.read_parquet(QROOT/'quarterly_firm_exposure_panel.parquet')[['vintage','cutoff','permco','ticker','comnam','cik']].drop_duplicates();key=rel[['relationship_id','document_id','firm_id','mechanism','evidence_passage_id','evidence_quote']]
control=control.merge(key,on=['document_id','firm_id','mechanism','evidence_passage_id','evidence_quote'],how='left',validate='many_to_one');assert control.relationship_id.notna().all();x=control.merge(classified,on=['document_id','relationship_id'],how='left',validate='many_to_many')
dw=design['directness_weights'];mw=design['materiality_weights'];sw=design['disclosure_status_weights'];x['component']=x.direction_y*x.intensity*x.confidence_y*x.directness.map(dw).fillna(0)*x.materiality.map(mw).fillna(0)*x.disclosure_status.map(sw).fillna(0)*x.eligible.astype(float)
def agg(g):
 pos=g.loc[g.component>0,'component'];neg=g.loc[g.component<0,'component'];return np.clip((pos.max() if len(pos) else 0)+(neg.min() if len(neg) else 0),-1,1)
ts=x.groupby(['vintage','permco','scenario_id']).apply(agg,include_groups=False).rename('text_score').reset_index();grid=base.assign(_k=1).merge(pd.DataFrame({'scenario_id':SCENARIOS,'_k':1}),on='_k').drop(columns='_k');scores=grid.merge(ts,on=['vintage','permco','scenario_id'],how='left');scores.text_score=scores.text_score.fillna(0);scores.to_parquet(OUT/'quarterly_scenario_text_scores.parquet',index=False)
display(scores.groupby(['vintage','scenario_id']).agg(positive=('text_score',lambda z:int((z>0).sum())),neutral=('text_score',lambda z:int((z==0).sum())),negative=('text_score',lambda z:int((z<0).sum()))).reset_index())''')
M('''## 8. Propagate each scenario through holdings embeddings

Similarity below the frozen random-pair q95 contributes zero. Seed firms use leave-one-seed-out support. Exact zero is preserved as economically meaningful neutrality.''')
C('''null=pd.read_parquet(ROOT/'embedding_propagation_v1'/'random_pair_null_summary.parquet').set_index('vintage');erows=[];dims=[f'embedding_{i:03d}' for i in range(1,11)]
for v in sorted(scores.vintage.unique()):
 p=PROJECT/'full_replication_80q'/'embeddings'/v/'rs_l_min'/'asset_embeddings_normalized_labeled.parquet';f=pd.read_parquet(p)[['permco']+dims].drop_duplicates('permco');A=f[dims].to_numpy(float);norm=np.linalg.norm(A,axis=1);keep=np.isfinite(A).all(1)&(norm>0);f=f.loc[keep].reset_index(drop=True);A=A[keep]/norm[keep,None];loc={int(p):i for i,p in enumerate(f.permco)};threshold=float(null.loc[v,'q95'])
 for scenario,g in scores[scores.vintage.eq(v)].groupby('scenario_id'):
  seed=g[g.text_score.ne(0)&g.permco.isin(loc)].copy();pos=seed[seed.text_score>0];neg=seed[seed.text_score<0]
  for target in g.itertuples():
   if int(target.permco) not in loc:erows.append({'vintage':v,'permco':target.permco,'scenario_id':scenario,'embedding_score':np.nan,'positive_support':np.nan,'negative_support':np.nan});continue
   def side(z):
    vals=[];weights=[]
    for s in z.itertuples():
     if s.permco==target.permco:continue
     sim=float(np.dot(A[loc[int(target.permco)]],A[loc[int(s.permco)]]));adj=max(0,(sim-threshold)/(1-threshold));
     if adj>0:vals.append(adj*abs(s.text_score));weights.append(abs(s.text_score))
    return sum(vals)/sum(weights) if weights else 0.
   ps=side(pos);ns=side(neg);erows.append({'vintage':v,'permco':target.permco,'scenario_id':scenario,'embedding_score':np.clip(ps-ns,-1,1),'positive_support':ps,'negative_support':ns})
embed=pd.DataFrame(erows);scores=scores.merge(embed,on=['vintage','permco','scenario_id'],how='left',validate='one_to_one');scores['hybrid_score']=.5*scores.text_score+.5*scores.embedding_score;scores.to_parquet(OUT/'quarterly_scenario_all_scores.parquet',index=False)
display(scores.groupby(['vintage','scenario_id']).agg(text_nonzero=('text_score',lambda x:int((x!=0).sum())),embedding_nonzero=('embedding_score',lambda x:int((x.fillna(0)!=0).sum())),embedding_coverage=('embedding_score','count')).reset_index())''')
M('''## 9. Freeze buckets and blinded validation candidates

The sample emphasizes direct seeds, embedding-only tails, consensus, disagreement, and neutral cases. It contains predicted fields only in a restricted key; the review packet uses anonymous case identifiers.''')
C('''long=scores.melt(id_vars=['vintage','permco','scenario_id','ticker','comnam','cik'],value_vars=['text_score','embedding_score','hybrid_score'],var_name='method',value_name='score');long['bucket']=pd.cut(long.score,[-np.inf,-.05,.05,np.inf],labels=['negative','neutral','positive']);long.to_parquet(OUT/'quarterly_scenario_rankings.parquet',index=False)
q4=scores[scores.vintage.eq('2024q4')].copy();q4['case_type']=np.select([(q4.text_score!=0)&(np.sign(q4.text_score)==np.sign(q4.embedding_score)),(q4.text_score!=0)&(q4.embedding_score!=0)&(np.sign(q4.text_score)!=np.sign(q4.embedding_score)),(q4.text_score==0)&(q4.embedding_score>.05),(q4.text_score==0)&(q4.embedding_score<-.05)],['consensus','disagreement','embedding_only_positive','embedding_only_negative'],default='neutral');q4['priority']=q4.hybrid_score.abs();sample=(q4.sort_values(['case_type','priority'],ascending=[True,False]).groupby('case_type',group_keys=False).head(20).sort_values('priority',ascending=False).head(design['validation_sample_target']).copy());sample['case_id']=[f'CMV{i:03d}' for i in range(1,len(sample)+1)];key=sample[['case_id','vintage','permco','scenario_id','ticker','comnam','text_score','embedding_score','hybrid_score','case_type']];packet=sample[['case_id','vintage','scenario_id']].assign(validation_direction=pd.NA,validation_intensity=pd.NA,evidence_source=pd.NA,evidence_date=pd.NA,review_notes=pd.NA)
key.to_parquet(MAN/'restricted_validation_key.parquet',index=False);packet.to_csv(OUT/'blinded_validation_review.csv',index=False);display(sample.case_type.value_counts().rename_axis('case_type').reset_index(name='cases'))''')
M('''## 10. Final integrity gates and manifest

These outputs remain outcome-blind. The next stage may collect independent validation evidence only after all gates pass.''')
C('''gates=pd.DataFrame([{'gate':'five_scenarios_per_relationship','value':classified.groupby('relationship_id').scenario_id.nunique().min(),'threshold':5,'passed':classified.groupby('relationship_id').scenario_id.nunique().min()==5},{'gate':'four_vintages','value':scores.vintage.nunique(),'threshold':4,'passed':scores.vintage.nunique()==4},{'gate':'scores_bounded','value':max(scores.text_score.abs().max(),scores.embedding_score.abs().max(),scores.hybrid_score.abs().max()),'threshold':1,'passed':max(scores.text_score.abs().max(),scores.embedding_score.abs().max(),scores.hybrid_score.abs().max())<=1},{'gate':'seed_self_exclusion_declared','value':1,'threshold':1,'passed':True},{'gate':'outcomes_unopened','value':0,'threshold':0,'passed':True}]);gates.to_parquet(MAN/'score_gates.parquet',index=False);display(gates);assert gates.passed.all();manifest={'stage':'scenario_specific_exposure_scores_v1','design_sha256':sha(design_path),'prompt_sha256':sha(prompt_path),'schema_sha256':sha(schema_path),'relationships':len(rel),'scenarios':SCENARIOS,'firm_scenario_quarters':len(scores),'validation_cases':len(sample),'returns_loaded':False,'all_gates_passed':True};(MAN/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n');print('Saved:',OUT)''')
nbf.write(n,R/'13A_Scenario_Specific_Exposure_Validation.ipynb')
