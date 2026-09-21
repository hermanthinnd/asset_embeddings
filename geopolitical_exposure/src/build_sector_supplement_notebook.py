"""Build the sector-enriched critical-minerals supplement notebook."""
from pathlib import Path
import nbformat as nbf

GEO_ROOT=Path(__file__).resolve().parents[1]
OUT=GEO_ROOT/'quarterly_sec_pilot'/'11B_Sector_Enriched_Supplement.ipynb'
def md(x): return nbf.v4.new_markdown_cell(x.strip()+'\n')
def code(x): return nbf.v4.new_code_cell(x.strip()+'\n')

cells=[
md("""
# Quarterly SEC–Embedding Pilot

## 11B-S — Sector-enriched critical-minerals supplement

The frozen broad-market screen produced only six firms above the exposure
threshold. This notebook preserves that sample unchanged and adds a separately
identified 50-firm exploratory supplement from industries plausibly connected
to the frozen mechanisms.

The supplement improves seed diversity; it is **not** representative and must
never be used to estimate market-wide exposure prevalence. Selection uses no
returns, text labels, company-name judgment, or embedding proximity to the six
base seeds. The identical ontology, retrieval terms, prompt, schema, model, and
point-in-time filing rules are reused.
"""),
md("""
## 1. Environment and frozen base handoff

All base decisions and outputs must exist. Supplement artifacts are stored in a
separate directory and cannot overwrite the broad-market experiment.
"""),
code("""
from pathlib import Path
import hashlib, importlib, json, os, sys
import numpy as np
import pandas as pd
from IPython.display import display

HERE=Path.cwd().resolve()
if HERE.name=='quarterly_sec_pilot': PILOT_ROOT,GEO_ROOT,ROOT=HERE,HERE.parent,HERE.parents[1]
elif HERE.name=='geopolitical_exposure': GEO_ROOT,ROOT=HERE,HERE.parent;PILOT_ROOT=GEO_ROOT/'quarterly_sec_pilot'
elif HERE.name=='Asset_Embeddings': ROOT=HERE;GEO_ROOT=ROOT/'geopolitical_exposure';PILOT_ROOT=GEO_ROOT/'quarterly_sec_pilot'
else: ROOT=(HERE/'Projects'/'Asset_Embeddings').resolve();GEO_ROOT=ROOT/'geopolitical_exposure';PILOT_ROOT=GEO_ROOT/'quarterly_sec_pilot'
sys.path.insert(0,str(GEO_ROOT/'src'))
import edgar_pipeline,document_contracts,quarterly_sec_extraction
importlib.reload(edgar_pipeline);importlib.reload(document_contracts);importlib.reload(quarterly_sec_extraction)
from edgar_pipeline import filing_metadata_for_cik,get_text,html_to_text,ticker_cik_crosswalk
from document_contracts import normalize_document,validate_documents
from quarterly_sec_extraction import build_retrieval_records,load_jsonl,run_batch,sha256_path,validate_checkpointed_responses

SUPPLEMENT=PILOT_ROOT/'sector_supplement';RAW=SUPPLEMENT/'data/raw';PROCESSED=SUPPLEMENT/'data/processed';RESULTS=SUPPLEMENT/'results';RESPONSES=RESULTS/'openai_responses';MANIFESTS=SUPPLEMENT/'manifests';CONFIG=PILOT_ROOT/'config'
for path in [RAW,PROCESSED,RESULTS,RESPONSES,MANIFESTS]:path.mkdir(parents=True,exist_ok=True)
base_manifest=json.loads((PILOT_ROOT/'manifests/extraction_manifest.json').read_text())
assert base_manifest['all_gates_passed'] and not base_manifest['outcomes_accessed']
base_sample=pd.read_parquet(PILOT_ROOT/'data/processed/quarterly_sec_pilot_100_firms_frozen.parquet')
design_path=CONFIG/'sector_supplement_design_v1.json';design=json.loads(design_path.read_text());design_hash=sha256_path(design_path)
display(pd.DataFrame([{'base_firms':base_sample.permco.nunique(),'base_exposed_firms':6,'supplement_target':design['sample_size'],'outcomes_accessed':False}]))
"""),
md("""
## 2. Build the eligible sector universe

Eligibility requires presence in all four 2024 RS-L-Min universes, a unique
exact SEC ticker–CIK link, Q1 market equity, and a CRSP SIC falling in one of six
prespecified industry buckets. The original 100 firms are excluded.
"""),
code("""
quarter_permcos=[]
for q in range(1,5):
    path=ROOT/f'full_replication_80q/embeddings/2024q{q}/rs_l_min/asset_embeddings_normalized_labeled.parquet'
    quarter_permcos.append(set(pd.read_parquet(path,columns=['permco']).permco))
common_permcos=set.intersection(*quarter_permcos)
names=pd.read_parquet(ROOT/'data/crosswalks/crsp/year=2024/quarter=4/stocknames_2024q4.parquet')
names=(names[names.permco.isin(common_permcos)].sort_values(['permco','nameenddt']).drop_duplicates('permco',keep='last'))
sec_tickers=pd.read_parquet(GEO_ROOT/'documents/raw/sec_reference/company_tickers_exchange.parquet')
bridge=ticker_cik_crosswalk(names,sec_tickers).query("cik_match_status=='unique_exact_ticker'").drop_duplicates('permco')
size=pd.read_parquet(ROOT/'remaining_benchmarks/return_comovement/characteristics/2024q1.parquet',columns=['permco','log_market_equity'])
eligible=bridge.merge(size,on='permco',how='inner',validate='one_to_one')
eligible=eligible[~eligible.permco.isin(base_sample.permco)&eligible.log_market_equity.notna()].copy()
eligible['siccd']=pd.to_numeric(eligible.siccd,errors='coerce')
conditions=[];labels=[]
for label,ranges in design['sector_buckets'].items():
    condition=pd.Series(False,index=eligible.index)
    for low,high in ranges:condition|=eligible.siccd.between(low,high)
    conditions.append(condition);labels.append(label)
eligible['sector_bucket']=np.select(conditions,labels,default='outside_scope')
eligible=eligible.query("sector_bucket!='outside_scope'").copy()
eligible['size_quintile']=pd.qcut(eligible.log_market_equity,5,labels=False,duplicates='drop')+1
display(eligible.groupby('sector_bucket').agg(firms=('permco','nunique'),median_log_me=('log_market_equity','median')).reset_index())
assert eligible.permco.nunique()>=design['sample_size']
"""),
md("""
## 3. Select and freeze 50 supplemental firms

Within each sector × size cell, firms receive a deterministic hash order.
Round-robin selection spreads the supplement across both dimensions. Company
names and filing contents are not used to choose individual firms.
"""),
code("""
selection_path=PROCESSED/'sector_supplement_50_firms_frozen.parquet'
if selection_path.exists(): supplement_firms=pd.read_parquet(selection_path);print('Loaded frozen supplement.')
else:
    eligible['stable_order']=eligible.permco.map(lambda x:hashlib.sha256(f"{design['selection_seed']}:{int(x)}".encode()).hexdigest())
    eligible=eligible.sort_values(['sector_bucket','size_quintile','stable_order'])
    eligible['stratum_rank']=eligible.groupby(['sector_bucket','size_quintile']).cumcount()
    supplement_firms=(eligible.sort_values(['stratum_rank','sector_bucket','size_quintile','stable_order']).head(design['sample_size']).copy())
    supplement_firms['selection_design_sha256']=design_hash;supplement_firms.to_parquet(selection_path,index=False)
assert supplement_firms.permco.nunique()==design['sample_size']
assert not set(supplement_firms.permco)&set(base_sample.permco)
display(supplement_firms.groupby(['sector_bucket','size_quintile']).size().unstack(fill_value=0))
display(pd.DataFrame([{'firms':len(supplement_firms),'sectors':supplement_firms.sector_bucket.nunique(),'size_quintiles':supplement_firms.size_quintile.nunique(),'selection_sha256':sha256_path(selection_path)}]))
"""),
md("""
## 4. Configure resumable SEC acquisition

Enter an identifying SEC user agent. Enable metadata first and documents second.
Existing submission JSON and prior filing HTML are reused automatically.
"""),
code("""
SEC_USER_AGENT=os.getenv('SEC_USER_AGENT','').strip()
if '@' not in SEC_USER_AGENT:SEC_USER_AGENT=input('SEC user agent (example: Academic research name@university.edu): ').strip()
if '@' not in SEC_USER_AGENT:raise ValueError('A valid SEC user agent is required.')
DOWNLOAD_SEC_METADATA=False
DOWNLOAD_SEC_DOCUMENTS=False
print('Metadata enabled:',DOWNLOAD_SEC_METADATA,'Documents enabled:',DOWNLOAD_SEC_DOCUMENTS)
"""),
md("""
## 5. Acquire filing metadata and construct vintage selections

For each cutoff, select the latest 10-K and latest subsequent 10-Q. Metadata is
cached CIK by CIK and this section is safe to rerun.
"""),
code("""
metadata_path=PROCESSED/'sec_filing_metadata_50_firms.parquet'
if metadata_path.exists():metadata=pd.read_parquet(metadata_path);print('Loaded metadata checkpoint.')
elif not DOWNLOAD_SEC_METADATA:raise RuntimeError('Set DOWNLOAD_SEC_METADATA=True in Section 4.')
else:
    frames=[];errors=[];cache=GEO_ROOT/'documents/raw/sec_submissions'
    for position,row in enumerate(supplement_firms.itertuples(index=False),1):
        try:
            frame=filing_metadata_for_cik(int(row.cik),SEC_USER_AGENT,['10-K','10-Q'],'2023-01-01','2024-12-31',cache)
            if not frame.empty:frame['permco']=int(row.permco);frames.append(frame)
        except Exception as error:errors.append({'permco':int(row.permco),'error_type':type(error).__name__,'error':str(error)})
        if position%10==0 or position==len(supplement_firms):print(f'Metadata {position}/{len(supplement_firms)}; errors={len(errors)}')
    metadata=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();pd.DataFrame(errors).to_parquet(RESULTS/'metadata_errors.parquet',index=False)
    if errors:raise RuntimeError(f'{len(errors)} metadata errors remain.')
    metadata.to_parquet(metadata_path,index=False)
metadata['filingDate']=pd.to_datetime(metadata.filingDate)
cutoffs={'2024q1':pd.Timestamp('2024-03-31'),'2024q2':pd.Timestamp('2024-06-30'),'2024q3':pd.Timestamp('2024-09-30'),'2024q4':pd.Timestamp('2024-12-31')}
rows=[]
for vintage,cutoff in cutoffs.items():
    for permco,group in metadata[metadata.filingDate.le(cutoff)].groupby('permco'):
        annual=group.query("form=='10-K'").sort_values(['filingDate','accessionNumber'])
        if annual.empty:continue
        latest=annual.iloc[-1];rows.append({'vintage':vintage,'document_role':'structural_anchor','document_cutoff':cutoff,**latest.to_dict()})
        quarterly=group[(group.form=='10-Q')&group.filingDate.gt(latest.filingDate)].sort_values(['filingDate','accessionNumber'])
        if not quarterly.empty:rows.append({'vintage':vintage,'document_role':'quarterly_update','document_cutoff':cutoff,**quarterly.iloc[-1].to_dict()})
vintage_documents=pd.DataFrame(rows).drop_duplicates(['vintage','accessionNumber']);vintage_documents['document_id']=vintage_documents.accessionNumber.astype(str)
unique_documents=vintage_documents.sort_values(['filingDate','accessionNumber']).drop_duplicates('accessionNumber')
vintage_documents.to_parquet(PROCESSED/'vintage_document_map.parquet',index=False);unique_documents.to_parquet(PROCESSED/'unique_selected_documents.parquet',index=False)
coverage=vintage_documents.groupby('vintage').agg(firms=('permco','nunique'),documents=('document_id','nunique'),latest=('filingDate','max')).reset_index();display(coverage)
"""),
md("""
## 6. Download and normalize unique filings

Raw HTML is located first in the supplement, base-pilot, V1, and V2 caches.
Only genuinely missing accessions are requested from EDGAR.
"""),
code("""
normalized_path=PROCESSED/'sec_documents_50_firms.jsonl';html_dir=RAW/'sec_filings';html_dir.mkdir(parents=True,exist_ok=True)
legacy=[PILOT_ROOT/'data/raw/sec_filings',GEO_ROOT/'documents/raw/sec_filings',GEO_ROOT/'text_expansion_v2/raw/sec_filings']
def locate(accession):
    for directory in [html_dir]+legacy:
        for suffix in ['.html','.htm']:
            path=directory/f'{accession}{suffix}'
            if path.exists():return path
    return None
if normalized_path.exists():documents=validate_documents(load_jsonl(normalized_path));print('Loaded normalized corpus.')
elif not DOWNLOAD_SEC_DOCUMENTS:raise RuntimeError('Set DOWNLOAD_SEC_DOCUMENTS=True in Section 4.')
else:
    documents=[];errors=[]
    for position,row in enumerate(unique_documents.itertuples(index=False),1):
        try:
            cached=locate(str(row.accessionNumber));html=cached.read_text(encoding='utf-8',errors='replace') if cached else get_text(row.source_uri,SEC_USER_AGENT)
            if cached is None:(html_dir/f'{row.accessionNumber}.html').write_text(html,encoding='utf-8')
            documents.append(normalize_document({'document_id':str(row.accessionNumber),'firm_id':str(int(row.permco)),'document_type':str(row.form),'document_date':pd.Timestamp(row.reportDate).date().isoformat(),'available_date':pd.Timestamp(row.filingDate).date().isoformat(),'source':'SEC EDGAR','source_uri':row.source_uri,'text':html_to_text(html)}))
        except Exception as error:errors.append({'document_id':str(row.accessionNumber),'error_type':type(error).__name__,'error':str(error)})
        if position%10==0 or position==len(unique_documents):print(f'Documents {position}/{len(unique_documents)}; errors={len(errors)}')
    pd.DataFrame(errors).to_parquet(RESULTS/'document_errors.parquet',index=False)
    with normalized_path.open('w',encoding='utf-8') as handle:
        for record in documents:handle.write(json.dumps(record,ensure_ascii=False)+'\\n')
    if errors:raise RuntimeError(f'{len(errors)} document errors remain.')
document_index=pd.DataFrame([{'document_id':d['document_id'],'firm_id':d['firm_id'],'document_type':d['document_type'],'available_date':d['available_date'],'content_sha256':d['content_sha256'],'characters':len(d['text'])} for d in documents])
document_index.to_parquet(PROCESSED/'document_index.parquet',index=False)
display(pd.DataFrame([{'documents':len(document_index),'firms':document_index.firm_id.nunique(),'median_characters':document_index.characters.median()}]))
"""),
md("""
## 7. Apply the frozen retrieval rule and preview API cost

The exact base-pilot retrieval function is reused. Empty retrieval is retained
as “not identified,” while only positive-hit documents enter the API queue.
"""),
code("""
retrieval_path=PROCESSED/'retrieval_records.jsonl';audit_path=RESULTS/'retrieval_audit.parquet'
if retrieval_path.exists():records=load_jsonl(retrieval_path);audit=pd.read_parquet(audit_path);print('Loaded retrieval checkpoint.')
else:
    records=build_retrieval_records(documents)
    with retrieval_path.open('w',encoding='utf-8') as handle:
        for record in records:handle.write(json.dumps(record,ensure_ascii=False)+'\\n')
    audit=pd.DataFrame([{'document_id':r['document_id'],'firm_id':r['firm_id'],'document_type':r['document_type'],'available_date':r['available_date'],'passage_count':len(r['passages']),'retrieved_characters':r['retrieved_characters']} for r in records]);audit.to_parquet(audit_path,index=False)
lookup={str(r['document_id']):r for r in records}
queue_path=PROCESSED/'api_queue_frozen.parquet'
if queue_path.exists():queue=pd.read_parquet(queue_path)
else:
    queue=audit.query('passage_count > 0').copy()
    queue['approx_input_tokens']=np.ceil(queue.retrieved_characters/4).astype(int)+2600;queue.to_parquet(queue_path,index=False)
estimated_input=int(queue.approx_input_tokens.sum());estimated_output=len(queue)*800
display(audit.groupby('document_type').agg(documents=('document_id','size'),hits=('passage_count',lambda x:int((x>0).sum())),characters=('retrieved_characters','sum')).reset_index())
display(pd.DataFrame([{'queue_documents':len(queue),'queue_firms':queue.firm_id.nunique(),'approx_input_tokens':estimated_input,'assumed_output_tokens':estimated_output,'terra_estimated_usd':estimated_input/1e6*2+estimated_output/1e6*12}]))
"""),
md("""
## 8. Freeze the reused extraction contract

The same GPT-5.6 Terra model, ontology, prompt, schema, and low-reasoning setting
are required for comparability. Set the API switch only after reviewing the
queue. Each run processes at most 25 new documents.
"""),
code("""
ontology_path=CONFIG/'critical_minerals_ontology_v1.json';prompt_path=CONFIG/'critical_minerals_extraction_prompt_v1.md';schema_path=CONFIG/'critical_minerals_exposure_schema_v1.json'
ontology=json.loads(ontology_path.read_text());prompt=prompt_path.read_text();schema=json.loads(schema_path.read_text())
OPENAI_MODEL='gpt-5.6-terra';API_BATCH_SIZE=25;RUN_API_BATCH=False
queue_hash=sha256_path(queue_path);contract={'stage':'sector_supplement_extraction','design_sha256':design_hash,'queue_sha256':queue_hash,'model':OPENAI_MODEL,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'schema_sha256':hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),'ontology_sha256':hashlib.sha256(json.dumps(ontology,sort_keys=True).encode()).hexdigest(),'store':False,'reasoning_effort':'low','outcomes_accessed':False}
contract_path=MANIFESTS/'api_contract.json'
if contract_path.exists() and json.loads(contract_path.read_text())!=contract:raise RuntimeError('Frozen supplement API contract mismatch.')
if not contract_path.exists():contract_path.write_text(json.dumps(contract,indent=2,sort_keys=True)+'\\n')
display(pd.DataFrame([{'model':OPENAI_MODEL,'api_key_present':bool(os.getenv('OPENAI_API_KEY')),'queued':len(queue),'checkpointed':len(list(RESPONSES.glob('*.json'))),'enabled':RUN_API_BATCH}]))
"""),
md("""
## 9. Run one resumable API batch

Set `RUN_API_BATCH=True` in Section 8. Rerun Sections 8–9 until no documents
remain. Successful responses are never repeated.
"""),
code("""
existing={path.stem for path in RESPONSES.glob('*.json')};remaining=queue[~queue.document_id.astype(str).isin(existing)].head(API_BATCH_SIZE)
display(pd.DataFrame([{'queued':len(queue),'checkpointed':len(set(queue.document_id.astype(str))&existing),'remaining_total':len(set(queue.document_id.astype(str))-existing),'next_batch':len(remaining)}]))
if not RUN_API_BATCH:print('API disabled. Set RUN_API_BATCH=True in Section 8.')
elif remaining.empty:print('Supplement extraction complete.')
else:
    failures=run_batch(remaining,lookup,ontology,schema,prompt,OPENAI_MODEL,RESPONSES);failures.to_parquet(RESULTS/'latest_failures.parquet',index=False);display(failures)
"""),
md("""
## 10. Validate and consolidate checkpointed responses

This applies the same contract and exact-quotation checks as the base screen.
Run after every batch. The notebook stops for adjudication once all responses
are present.
"""),
code("""
prompt_hash=hashlib.sha256(prompt.encode()).hexdigest();schema_hash=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest();ontology_hash=hashlib.sha256(json.dumps(ontology,sort_keys=True).encode()).hexdigest()
response_audit,exposures=validate_checkpointed_responses(queue,documents,RESPONSES,OPENAI_MODEL,prompt_hash,schema_hash,ontology_hash)
response_audit.to_parquet(RESULTS/'response_audit.parquet',index=False);exposures.to_parquet(RESULTS/'document_exposures.parquet',index=False)
completed=set(response_audit.document_id);status=pd.DataFrame([{'queued':len(queue),'responses':len(completed),'remaining':len(set(queue.document_id.astype(str))-completed),'valid':int(response_audit.valid.sum()) if len(response_audit) else 0,'invalid':int((~response_audit.valid).sum()) if len(response_audit) else 0,'exposure_records':len(exposures),'exposed_firms':exposures.firm_id.nunique() if len(exposures) else 0}]);display(status)
if len(exposures):display(exposures.groupby(['mechanism','direction']).size().rename('records').reset_index())
if len(response_audit) and (~response_audit.valid).any():display(response_audit.query('not valid'))
"""),
md("""
## 11. Freeze the handoff for evidence adjudication

When the queue is complete, save a review sheet but do not construct labels.
Every exposure will be inspected before merging the supplement with the base
sample. The base and enriched samples retain separate sample-origin fields.
"""),
code("""
row=status.iloc[0]
if row.remaining:raise RuntimeError('Complete the remaining API batches first.')
review=exposures.copy();review['review_decision']='pending';review['review_note']='';review.to_parquet(RESULTS/'evidence_review_sheet.parquet',index=False)
handoff={'stage':'sector_supplement_pending_adjudication','design_sha256':design_hash,'queue_sha256':queue_hash,'firms':len(supplement_firms),'queued_documents':len(queue),'response_documents':int(row.responses),'exposure_records':len(exposures),'outcomes_accessed':False}
(MANIFESTS/'extraction_handoff.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\\n')
display(pd.DataFrame([handoff]));print('Stop here for evidence adjudication.')
"""),
]
nb=nbf.v4.new_notebook(cells=cells);nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}}
OUT.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,OUT);print(OUT)
