import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "13B_Blinded_Scenario_Adjudication.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 13B — Blinded scenario adjudication and score calibration

## Purpose

This notebook tests whether the scenario-specific exposure labels created in 13A are economically defensible **before any return outcome is opened**. It creates a blinded review packet from contemporaneously available SEC filing passages, validates completed reviews, and only then decodes the predictions to measure reliability.

The reviewer sees the scenario and filing evidence, but not the company name, ticker, text score, embedding score, hybrid score, or sampling category. This prevents those model outputs from influencing the judgment.

### Review question

Conditional on the stated scenario occurring, would the disclosed business relationship have a positive, negative, neutral, or presently indeterminate effect on the firm's operating economics?

This is a construct-validation exercise. It does not use stock returns and does not establish investment performance.""")

md("""## 1. Setup and frozen review rules

Allowed judgments:

- `positive`: evidence supports a beneficial operating effect under the scenario.
- `negative`: evidence supports an adverse operating effect.
- `neutral`: evidence is relevant but does not support a directional net effect.
- `insufficient_evidence`: the supplied passages cannot support a defensible judgment.

Intensity is an integer from 0 to 3: none, weak, moderate, or strong. A neutral or insufficient case must have intensity 0. The review must cite at least one supplied passage for every positive or negative judgment.""")

code("""from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name != 'sp500_critical_minerals':
    candidate = Path('/Users/harmandeep/Desktop/Projects/Asset_Embeddings/geopolitical_exposure/sp500_critical_minerals')
    if candidate.exists(): ROOT = candidate

SOURCE = ROOT / 'scenario_exposure_v1'
OUT = ROOT / 'scenario_adjudication_v1'
MANIFESTS = OUT / 'manifests'
OUT.mkdir(exist_ok=True); MANIFESTS.mkdir(exist_ok=True)

ALLOWED = {'positive','negative','neutral','insufficient_evidence'}
DESIGN_PATH = ROOT/'config'/'scenario_exposure_design_v1.json'
DESIGN = json.loads(DESIGN_PATH.read_text())
SCENARIOS = {item['id']:item['definition'] for item in DESIGN['scenarios']}
DESIGN_SHA256 = hashlib.sha256(DESIGN_PATH.read_bytes()).hexdigest()
print({'root': str(ROOT), 'outcomes_loaded': False, 'cases_expected': 100,
       'scenario_design_sha256': DESIGN_SHA256})""")

md("""## 2. Verify the frozen sample and source integrity

The restricted key is used only to map anonymous case IDs to filing records. Prediction columns are not copied into the review packet.""")

code("""key_path = SOURCE/'manifests'/'restricted_validation_key.parquet'
review_template_path = SOURCE/'blinded_validation_review.csv'
records_path = ROOT/'data'/'processed'/'sp500_retrieval_records.jsonl'

key = pd.read_parquet(key_path)
template = pd.read_csv(review_template_path, dtype={'case_id':'string'})
records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
records_df = pd.DataFrame(records)
records_df['firm_id'] = pd.to_numeric(records_df['firm_id'], errors='coerce').astype('Int64')
records_df['available_date'] = pd.to_datetime(records_df['available_date'])

assert len(key) == 100 and key.case_id.is_unique
assert set(template.case_id) == set(key.case_id)
assert set(key.scenario_id).issubset(SCENARIOS)
assert key.vintage.eq('2024q4').all()

integrity = pd.DataFrame([
 {'gate':'one_hundred_unique_cases','value':key.case_id.nunique(),'threshold':100,'passed':key.case_id.nunique()==100},
 {'gate':'review_template_matches_key','value':len(set(template.case_id)&set(key.case_id)),'threshold':100,'passed':set(template.case_id)==set(key.case_id)},
 {'gate':'outcomes_unopened','value':0,'threshold':0,'passed':True},
])
display(integrity)
assert integrity.passed.all()""")

md("""## 3. Construct evidence packets without exposing predictions

For each anonymous case, the notebook uses only SEC passages available by the 2024 Q4 cutoff. Passages are ranked deterministically using scenario-specific vocabulary. This ranking is a retrieval aid, not a direction classifier.""")

code("""TERMS = {
 'strategic_demand_expansion': ['demand','capacity','production','expansion','critical mineral','lithium','copper','nickel','rare earth'],
 'mineral_price_increase': ['price','cost','margin','commodity','raw material','input','lithium','copper','nickel','rare earth'],
 'physical_supply_disruption': ['supply','shortage','disruption','source','supplier','inventory','availability','concentration'],
 'western_policy_support': ['subsidy','grant','tax credit','government','policy','procurement','permit','financing','inflation reduction act'],
 'western_supply_chain_reorientation': ['domestic','united states','canada','allied','localize','reshore','supply chain','china','western']
}

def passage_rank(text, scenario):
    low = str(text).lower()
    return sum(low.count(term) for term in TERMS[scenario])

rows=[]
for case in key[['case_id','permco','scenario_id']].itertuples(index=False):
    docs=records_df[(records_df.firm_id.eq(int(case.permco))) & (records_df.available_date <= pd.Timestamp('2024-12-31'))]
    candidates=[]
    for doc in docs.itertuples(index=False):
        for j,p in enumerate(doc.passages):
            text=str(p.get('text','')).strip()
            candidates.append({'document_id':doc.document_id,'document_type':doc.document_type,
                'available_date':doc.available_date,'passage_id':f'{doc.document_id}:P{j+1:02d}',
                'text':text,'retrieval_rank_score':passage_rank(text,case.scenario_id)})
    candidates=sorted(candidates,key=lambda x:(-x['retrieval_rank_score'],str(x['available_date']),x['passage_id']))[:8]
    if not candidates:
        rows.append({'case_id':case.case_id,'vintage':'2024q4','scenario_id':case.scenario_id,'scenario_definition':SCENARIOS[case.scenario_id],
                     'passage_id':pd.NA,'document_type':pd.NA,'available_date':pd.NaT,'passage_text':'NO PASSAGE AVAILABLE'})
    else:
        for x in candidates:
            rows.append({'case_id':case.case_id,'vintage':'2024q4','scenario_id':case.scenario_id,'scenario_definition':SCENARIOS[case.scenario_id],
                         'passage_id':x['passage_id'],'document_type':x['document_type'],'available_date':x['available_date'],
                         'passage_text':x['text']})

evidence=pd.DataFrame(rows)
evidence.to_parquet(OUT/'blinded_evidence_packet.parquet',index=False)
display(evidence.groupby('scenario_id').agg(cases=('case_id','nunique'),passages=('passage_id','count')).reset_index())
display(evidence.head(8))""")

md("""## 4. Create or preserve the blinded review sheet

Edit `scenario_adjudication_v1/blinded_scenario_review.csv` outside the notebook. Do not open the restricted key while reviewing.

Required fields are `validation_direction`, `validation_intensity`, `cited_passage_ids`, and `review_notes`. Separate multiple passage IDs with semicolons. Set `review_complete` to `True` only after making a final judgment.""")

code("""review_path=OUT/'blinded_scenario_review.csv'
base=(evidence[['case_id','vintage','scenario_id','scenario_definition']].drop_duplicates()
      .sort_values('case_id').reset_index(drop=True))
for col,default in [('validation_direction',''),('validation_intensity',''),('cited_passage_ids',''),
                    ('review_notes',''),('review_complete',False)]: base[col]=default

if review_path.exists():
    old=pd.read_csv(review_path,dtype={'case_id':'string'})
    editable=['validation_direction','validation_intensity','cited_passage_ids','review_notes','review_complete']
    old=old[['case_id']+[c for c in editable if c in old]].drop_duplicates('case_id')
    base=base.drop(columns=editable).merge(old,on='case_id',how='left',validate='one_to_one')
    for col,default in [('validation_direction',''),('validation_intensity',''),('cited_passage_ids',''),('review_notes',''),('review_complete',False)]:
        if col not in base: base[col]=default
base.to_csv(review_path,index=False)
print('Review sheet:',review_path)
display(base.head())""")

md("""## 5. Optional blinded AI-assisted adjudication

This is available when the researcher delegates the evidence review to a second, independently prompted model. It is **not human adjudication**. The model receives only the anonymous case ID, scenario definition, and frozen SEC passages. Predictions, firm identities, method labels, and returns are excluded.

Set `RUN_AI_REVIEW=True` to run it. Responses are checkpointed case by case, so interruption is safe. `AI_REVIEW_BATCH_SIZE` means the number of unfinished cases to process in the current run. Start with 10; after inspection, use 25 or 90.""")

code("""import os, getpass, time

RUN_AI_REVIEW = False
AI_REVIEW_BATCH_SIZE = 10
AI_REVIEW_MODEL = 'gpt-5.6-terra'
AI_RESPONSES = OUT/'ai_review_responses'; AI_RESPONSES.mkdir(exist_ok=True)

AI_REVIEW_PROMPT = '''You are an independent research adjudicator. Evaluate only the supplied, contemporaneous SEC filing passages. You do not know the firm identity, portfolio selection method, model prediction, or later returns.

Question: conditional on the precisely stated scenario occurring, what directional operating-economic effect is supported for the filing company?

Rules:
1. Use only supplied passages. Do not use outside knowledge or infer the company identity.
2. positive means a supported beneficial effect; negative means a supported adverse effect.
3. neutral means the passages address the scenario but support no directional net effect.
4. insufficient_evidence means the passages do not establish a scenario-specific causal channel. Prefer this over speculation.
5. Generic mentions of supply chains, government, commodities, inflation, ESG, or geography are insufficient unless tied to the scenario.
6. Do not infer that an industry automatically benefits or suffers.
7. Cite only passage IDs supplied in the payload. Positive and negative decisions require at least one citation.
8. Intensity is 0 for neutral/insufficient, 1 weak, 2 moderate, 3 strong.
Return a short evidence-based note, not hidden reasoning.'''

AI_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'case_id':{'type':'string'},
 'validation_direction':{'type':'string','enum':['positive','negative','neutral','insufficient_evidence']},
 'validation_intensity':{'type':'integer','enum':[0,1,2,3]},
 'cited_passage_ids':{'type':'array','items':{'type':'string'}},
 'review_notes':{'type':'string'}},
 'required':['case_id','validation_direction','validation_intensity','cited_passage_ids','review_notes']}

if RUN_AI_REVIEW:
    if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY']=getpass.getpass('OpenAI API key (input hidden): ').strip()
    from openai import OpenAI
    client=OpenAI(timeout=90,max_retries=2)
    all_pending=[cid for cid in base.case_id if not (AI_RESPONSES/f'{cid}.json').exists()]
    pending=all_pending[:AI_REVIEW_BATCH_SIZE]
    print({'total_cases':len(base),'already_checkpointed':len(base)-len(all_pending),
           'remaining_before_run':len(all_pending),'current_batch':len(pending)})
    for number,cid in enumerate(pending,1):
        target=AI_RESPONSES/f'{cid}.json'
        g=evidence[evidence.case_id.eq(cid)]
        payload={'case_id':cid,'scenario_id':g.scenario_id.iloc[0],
                 'scenario_definition':g.scenario_definition.iloc[0],
                 'passages':[{'passage_id':str(x.passage_id),'document_type':str(x.document_type),
                              'available_date':str(x.available_date)[:10],'text':str(x.passage_text)}
                             for x in g.itertuples(index=False) if pd.notna(x.passage_id)]}
        response=client.responses.create(model=AI_REVIEW_MODEL,instructions=AI_REVIEW_PROMPT,
          input=json.dumps(payload),store=False,reasoning={'effort':'low'},
          text={'format':{'type':'json_schema','name':'blinded_scenario_review','strict':True,'schema':AI_SCHEMA}})
        result=json.loads(response.output_text)
        if str(result.get('case_id')) != str(cid):
            raise ValueError(f'Response case_id mismatch: requested {cid}, received {result.get("case_id")}')
        envelope={'result':result,'model':response.model,'response_id':response.id,
                  'prompt_sha256':hashlib.sha256(AI_REVIEW_PROMPT.encode()).hexdigest(),
                  'evidence_packet_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                  'reviewer_type':'independent_ai_assisted','predictions_exposed':False,'outcomes_exposed':False,
                  'scenario_design_sha256':DESIGN_SHA256}
        target.write_text(json.dumps(envelope,indent=2,sort_keys=True)+'\\n')
        print(f'[{number}/{len(pending)}] {cid} checkpointed')

# Import all available blinded AI-review checkpoints into the editable sheet.
review=pd.read_csv(review_path,dtype={'case_id':'string'})
loaded=0
for path in sorted(AI_RESPONSES.glob('CMV*.json')):
    envelope=json.loads(path.read_text()); result=envelope['result']; cid=str(result['case_id'])
    mask=review.case_id.eq(cid)
    if mask.sum()!=1: continue
    review.loc[mask,'validation_direction']=result['validation_direction']
    review.loc[mask,'validation_intensity']=result['validation_intensity']
    review.loc[mask,'cited_passage_ids']=';'.join(result['cited_passage_ids'])
    review.loc[mask,'review_notes']='AI-assisted blinded review: '+result['review_notes']
    review.loc[mask,'review_complete']=True
    loaded+=1
review.to_csv(review_path,index=False)
print({'ai_review_checkpoints_imported':loaded,'reviewer_type':'independent_ai_assisted','human_adjudication':False})""")

md("""## 6. Review progress and mechanical validation

This section may be rerun at any time. It never decodes incomplete judgments.""")

code("""review=pd.read_csv(review_path,dtype={'case_id':'string'})
review['validation_direction']=review.validation_direction.fillna('').str.strip().str.lower()
review['validation_intensity']=pd.to_numeric(review.validation_intensity,errors='coerce')
review['review_complete']=review.review_complete.fillna(False).astype(str).str.lower().isin(['true','1','yes'])

available={case:set(g.passage_id.dropna().astype(str)) for case,g in evidence.groupby('case_id')}
def citations_valid(row):
    if row.validation_direction in {'neutral','insufficient_evidence'}: return True
    cited={x.strip() for x in str(row.cited_passage_ids).split(';') if x.strip() and x.strip().lower()!='nan'}
    return bool(cited) and cited.issubset(available.get(row.case_id,set()))

review['direction_valid']=review.validation_direction.isin(ALLOWED)
review['intensity_valid']=review.validation_intensity.isin([0,1,2,3])
review['neutral_intensity_valid']=~review.validation_direction.isin(['neutral','insufficient_evidence']) | review.validation_intensity.eq(0)
review['citation_valid']=review.apply(citations_valid,axis=1)
review['row_valid']=review[['review_complete','direction_valid','intensity_valid','neutral_intensity_valid','citation_valid']].all(axis=1)
progress=pd.DataFrame([{'cases':len(review),'completed':int(review.review_complete.sum()),'valid':int(review.row_valid.sum()),
                        'remaining':int((~review.row_valid).sum())}])
display(progress)
if (~review.row_valid).any():
    display(review.loc[~review.row_valid,['case_id','review_complete','validation_direction','validation_intensity',
                                         'direction_valid','intensity_valid','neutral_intensity_valid','citation_valid']].head(20))""")

md("""## 7. Decode only after all reviews pass

This is the first point at which model predictions and sampling categories are joined back to the judgments. No return data are used.""")

code("""if not review.row_valid.all():
    print(f'DECODING LOCKED: {int((~review.row_valid).sum())} cases remain incomplete or invalid.')
else:
    decoded=review.merge(key,on=['case_id','vintage','scenario_id'],how='left',validate='one_to_one')
    truth=decoded.validation_direction.map({'positive':1,'neutral':0,'negative':-1,'insufficient_evidence':np.nan})
    decoded['validation_sign']=truth
    decoded['text_prediction']=np.sign(decoded.text_score)
    decoded['embedding_prediction']=np.sign(decoded.embedding_score)
    decoded['hybrid_prediction']=np.sign(decoded.hybrid_score)
    decoded['text_correct']=decoded.text_prediction.eq(truth)
    decoded['embedding_correct']=decoded.embedding_prediction.eq(truth)
    decoded['hybrid_correct']=decoded.hybrid_prediction.eq(truth)
    decoded.to_parquet(OUT/'decoded_validation_results.parquet',index=False)

    usable=decoded[decoded.validation_sign.notna()].copy()
    result_rows=[]
    for grouping in ['overall','scenario_id','case_type']:
        groups=[('overall',usable)] if grouping=='overall' else usable.groupby(grouping,sort=True)
        for label,g in groups:
            for method in ['text','embedding','hybrid']:
                pred=g[f'{method}_prediction']; actual=g.validation_sign
                result_rows.append({'grouping':grouping,'group':label,'method':method,'cases':len(g),
                    'exact_direction_accuracy':float(pred.eq(actual).mean()),
                    'directional_accuracy_nonzero_truth':float(pred[actual.ne(0)].eq(actual[actual.ne(0)]).mean()) if actual.ne(0).any() else np.nan,
                    'false_direction_rate_on_neutral':float(pred[actual.eq(0)].ne(0).mean()) if actual.eq(0).any() else np.nan})
    metrics=pd.DataFrame(result_rows)
    metrics.to_parquet(OUT/'validation_reliability_metrics.parquet',index=False)
    display(metrics[metrics.grouping.eq('overall')])
    display(metrics[metrics.grouping.eq('scenario_id')])""")

md("""## 8. Reliability gates and frozen handoff

The gates are deliberately descriptive rather than retroactively optimized. They identify which scenario–method combinations have enough validation support to proceed to return tests. Thresholds must not be changed after examining returns.""")

code("""if review.row_valid.all():
    usable=decoded[decoded.validation_sign.notna()]
    scenario_metrics=metrics[metrics.grouping.eq('scenario_id')].copy()
    scenario_metrics['enough_cases']=scenario_metrics.cases.ge(10)
    scenario_metrics['directionally_reliable']=scenario_metrics.directional_accuracy_nonzero_truth.ge(0.60)
    scenario_metrics['neutrality_acceptable']=scenario_metrics.false_direction_rate_on_neutral.le(0.50) | scenario_metrics.false_direction_rate_on_neutral.isna()
    scenario_metrics['advance_to_returns']=scenario_metrics[['enough_cases','directionally_reliable','neutrality_acceptable']].all(axis=1)
    scenario_metrics.to_parquet(MANIFESTS/'scenario_method_decision_gates.parquet',index=False)
    display(scenario_metrics)

    manifest={'stage':'blinded_scenario_adjudication_v1','cases':len(decoded),
      'usable_cases':len(usable),'outcomes_loaded':False,
      'review_file_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
      'evidence_packet_sha256':hashlib.sha256((OUT/'blinded_evidence_packet.parquet').read_bytes()).hexdigest(),
      'decoding_completed':True}
    (MANIFESTS/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n')
    print('Validation complete. Frozen handoff:',OUT)
else:
    print('Complete the blinded review sheet, then rerun Sections 5–7.')""")

md("""## Interpretation discipline

- A passing validation result supports the economic meaning of the score, not return predictability.
- `insufficient_evidence` is a legitimate result and must not be silently converted to neutral.
- One-sided scenarios should remain one-sided in later portfolio tests when the evidence supports only beneficiaries or only harmed firms.
- The return evaluation must use rules frozen here and must compare text, embedding, and hybrid methods using matched portfolio sizes and weighting conventions.""")

nb["cells"] = cells
nbf.write(nb, OUT)
print(OUT)
