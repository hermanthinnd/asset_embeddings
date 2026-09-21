"""Build Notebook 11B for quarterly SEC critical-minerals extraction."""

from pathlib import Path
import nbformat as nbf

GEO_ROOT = Path(__file__).resolve().parents[1]
OUT = GEO_ROOT / "quarterly_sec_pilot" / "11B_Quarterly_SEC_Exposure_Extraction.ipynb"


def md(value):
    return nbf.v4.new_markdown_cell(value.strip() + "\n")


def code(value):
    return nbf.v4.new_code_cell(value.strip() + "\n")


cells = [
md("""
# Quarterly SEC–Embedding Pilot

## 11B — Retrieval and critical-minerals exposure extraction

This notebook converts the frozen 100-firm, four-vintage SEC corpus into
auditable critical-minerals exposure labels. It freezes the economic ontology,
retrieval rules, prompt, schema, model, thresholds, and API queue before making
any model call.

The LLM receives only retrieved, point-in-time SEC passages. It never receives
embeddings, prices, returns, future filings, or company-name-based hints. The
output is a provisional text label—not ground truth—and exact evidence is
validated against the locally stored filing.
"""),
md("""
## 1. Load the completed corpus handoff

Notebook 11A must have passed every gate. Its design, sample, document index,
and information-panel hashes are checked before this notebook proceeds.
"""),
code("""
from pathlib import Path
import hashlib, importlib, json, os, sys
import numpy as np
import pandas as pd
from IPython.display import display

HERE = Path.cwd().resolve()
if HERE.name == 'quarterly_sec_pilot':
    PILOT_ROOT, GEO_ROOT, ROOT = HERE, HERE.parent, HERE.parents[1]
elif HERE.name == 'geopolitical_exposure':
    GEO_ROOT, ROOT = HERE, HERE.parent
    PILOT_ROOT = GEO_ROOT / 'quarterly_sec_pilot'
elif HERE.name == 'Asset_Embeddings':
    ROOT = HERE; GEO_ROOT = ROOT / 'geopolitical_exposure'; PILOT_ROOT = GEO_ROOT / 'quarterly_sec_pilot'
else:
    ROOT = (HERE / 'Projects' / 'Asset_Embeddings').resolve()
    GEO_ROOT = ROOT / 'geopolitical_exposure'; PILOT_ROOT = GEO_ROOT / 'quarterly_sec_pilot'

sys.path.insert(0, str(GEO_ROOT / 'src'))
import quarterly_sec_extraction
importlib.reload(quarterly_sec_extraction)
from quarterly_sec_extraction import build_retrieval_records, load_jsonl, run_batch, sha256_path, validate_checkpointed_responses

CONFIG = PILOT_ROOT / 'config'
PROCESSED = PILOT_ROOT / 'data' / 'processed'
MANIFESTS = PILOT_ROOT / 'manifests'
RESULTS = PILOT_ROOT / 'results' / 'extraction'
RESPONSES = RESULTS / 'openai_responses'
for path in [MANIFESTS, RESULTS, RESPONSES]: path.mkdir(parents=True, exist_ok=True)

corpus_manifest_path = MANIFESTS / 'corpus_manifest.json'
corpus_manifest = json.loads(corpus_manifest_path.read_text())
assert corpus_manifest['all_gates_passed'] and not corpus_manifest['outcomes_accessed']
documents = load_jsonl(PROCESSED / 'sec_documents_100_firms.jsonl')
document_index = pd.read_parquet(PROCESSED / 'document_index.parquet')
information_panel = pd.read_parquet(PROCESSED / 'firm_quarter_information_panel.parquet')
pilot = pd.read_parquet(PROCESSED / 'quarterly_sec_pilot_100_firms_frozen.parquet')
assert sha256_path(PROCESSED / 'document_index.parquet') == corpus_manifest['document_index_sha256']
assert sha256_path(PROCESSED / 'firm_quarter_information_panel.parquet') == corpus_manifest['information_panel_sha256']
display(pd.DataFrame([{'firms': pilot.permco.nunique(), 'documents': len(documents),
                       'firm_quarter_rows': information_panel[['vintage','permco']].drop_duplicates().shape[0],
                       'corpus_gate': corpus_manifest['all_gates_passed']}]))
"""),
md("""
## 2. Freeze the theme ontology, prompt, schema, and aggregation rules

The ontology separates twelve economic mechanisms and defines direction
relative to one fixed shock: stronger strategic demand and policy support for
secure Western critical-mineral supply chains. Mineral mention alone is not
sufficient evidence.

The extraction prompt is intentionally detailed, but it does not invite web
research or investment forecasting. The aggregation threshold is frozen before
responses are examined.
"""),
code("""
ontology_path = CONFIG / 'critical_minerals_ontology_v1.json'
prompt_path = CONFIG / 'critical_minerals_extraction_prompt_v1.md'
schema_path = CONFIG / 'critical_minerals_exposure_schema_v1.json'
ontology = json.loads(ontology_path.read_text())
prompt = prompt_path.read_text()
schema = json.loads(schema_path.read_text())

extraction_rules = {
    'contract_version': 'critical_minerals_extraction_v1',
    'ontology_sha256': sha256_path(ontology_path),
    'prompt_sha256': sha256_path(prompt_path),
    'schema_sha256': sha256_path(schema_path),
    'retrieval_terms_sha256': hashlib.sha256(json.dumps(quarterly_sec_extraction.SEARCH_TERMS).encode()).hexdigest(),
    'retrieval_radius_characters': 850,
    'maximum_passages_per_document': 18,
    'maximum_exposures_per_document': 6,
    'maximum_evidence_words': 60,
    'document_relatedness_score': 'exposure_strength multiplied by confidence',
    'firm_mechanism_quarter_aggregation': 'maximum valid document relatedness score',
    'provisional_exposure_threshold': 0.35,
    'material_change_threshold': 0.10,
    'zero_retrieval_interpretation': 'not_identified_in_supplied_text',
    'outcomes_accessed': False,
}
rules_path = MANIFESTS / 'extraction_rules_v1.json'
rules_path.write_text(json.dumps(extraction_rules, indent=2, sort_keys=True) + '\\n')
display(pd.DataFrame([extraction_rules]))
display(pd.DataFrame(ontology['mechanisms'])[['id','definition']])
"""),
md("""
## 3. Retrieve auditable filing passages

Retrieval uses a frozen list of mineral, processing, battery, localization, and
policy phrases. Overlapping windows are merged and ranked using only textual
specificity and business-materiality terms. This stage is deterministic and
does not call OpenAI.

An empty retrieval result means only that this procedure did not identify
evidence; it does not prove that the firm has no exposure.
"""),
code("""
retrieval_path = PROCESSED / 'critical_minerals_retrieval_records.jsonl'
retrieval_audit_path = RESULTS / 'retrieval_audit.parquet'
if retrieval_path.exists() and retrieval_audit_path.exists():
    retrieval_records = load_jsonl(retrieval_path)
    retrieval_audit = pd.read_parquet(retrieval_audit_path)
    print('Loaded checkpointed retrieval.')
else:
    retrieval_records = build_retrieval_records(documents)
    with retrieval_path.open('w', encoding='utf-8') as handle:
        for record in retrieval_records: handle.write(json.dumps(record, ensure_ascii=False) + '\\n')
    retrieval_audit = pd.DataFrame([{
        'document_id': record['document_id'], 'firm_id': record['firm_id'],
        'document_type': record['document_type'], 'available_date': record['available_date'],
        'content_sha256': record['content_sha256'], 'passage_count': len(record['passages']),
        'retrieved_characters': record['retrieved_characters'],
    } for record in retrieval_records])
    retrieval_audit.to_parquet(retrieval_audit_path, index=False)

retrieval_lookup = {str(record['document_id']): record for record in retrieval_records}
summary = retrieval_audit.groupby('document_type', as_index=False).agg(
    documents=('document_id','nunique'), documents_with_passages=('passage_count',lambda x:int((x>0).sum())),
    firms_with_passages=('firm_id',lambda x:retrieval_audit.loc[x.index].query('passage_count > 0').firm_id.nunique()),
    retrieved_characters=('retrieved_characters','sum'))
summary['retrieval_hit_rate'] = summary.documents_with_passages / summary.documents
display(summary)
display(retrieval_audit.retrieved_characters.describe(percentiles=[.5,.75,.9,.95,.99]).to_frame().T)
"""),
md("""
## 4. Audit quarterly retrieval coverage

This audit distinguishes the number of firms in the corpus from the number for
which the frozen retrieval rule found potentially relevant language. The queue
is built over unique filings, so a 10-K reused in four vintages generates only
one API call.
"""),
code("""
quarter_retrieval = (information_panel[['vintage','permco','document_id','document_role']]
    .merge(retrieval_audit[['document_id','passage_count','retrieved_characters']], on='document_id', how='left', validate='many_to_one'))
quarter_summary = quarter_retrieval.groupby('vintage', as_index=False).agg(
    corpus_firms=('permco','nunique'), corpus_documents=('document_id','nunique'),
    retrieved_documents=('passage_count',lambda x:int((x>0).sum())),
    retrieved_firms=('permco',lambda x:quarter_retrieval.loc[x.index].query('passage_count > 0').permco.nunique()),
    retrieved_characters=('retrieved_characters','sum'))
quarter_summary['retrieved_firm_rate'] = quarter_summary.retrieved_firms / quarter_summary.corpus_firms
display(quarter_summary)
quarter_summary.to_parquet(RESULTS / 'quarterly_retrieval_coverage.parquet', index=False)
"""),
md("""
## 5. Freeze the API queue and preview token cost

Only documents containing at least one retrieved passage enter the API queue.
Input tokens are approximated as characters divided by four plus a fixed
allowance for the prompt, ontology, schema, and metadata. Output cost is shown
using a conservative 800-token allowance per response. The estimate is a
planning calculation; actual usage will be recorded from every response.
"""),
code("""
queue_path = PROCESSED / 'critical_minerals_api_queue_frozen.parquet'
if queue_path.exists():
    api_queue = pd.read_parquet(queue_path)
    print('Loaded frozen API queue.')
else:
    api_queue = retrieval_audit.query('passage_count > 0').copy().sort_values(['document_type','document_id'])
    api_queue['approx_input_tokens'] = np.ceil(api_queue.retrieved_characters / 4).astype(int) + 2600
    api_queue.to_parquet(queue_path, index=False)
queue_hash = sha256_path(queue_path)

PRICE_SNAPSHOT = {
    'gpt-5.6-luna': {'input': .20, 'output': 1.20},
    'gpt-5-mini': {'input': .25, 'output': 2.00},
    'gpt-5.4-mini': {'input': .75, 'output': 4.50},
    'gpt-5.6-terra': {'input': 2.00, 'output': 12.00},
    'gpt-5.6-sol': {'input': 4.00, 'output': 20.00},
}
cost_rows = []
input_tokens = int(api_queue.approx_input_tokens.sum())
assumed_output_tokens = len(api_queue) * 800
for model, price in PRICE_SNAPSHOT.items():
    cost_rows.append({'model':model, 'queued_documents':len(api_queue), 'approx_input_tokens':input_tokens,
                      'assumed_output_tokens':assumed_output_tokens,
                      'estimated_usd':input_tokens/1e6*price['input'] + assumed_output_tokens/1e6*price['output']})
cost_preview = pd.DataFrame(cost_rows)
display(cost_preview)
cost_preview.to_parquet(RESULTS / 'api_cost_preview.parquet', index=False)
print('Frozen queue SHA-256:', queue_hash)
"""),
md("""
## 6. Freeze the model and execution switches

The default is GPT-5.6 Terra with low reasoning effort. Override it only before
the first response by setting `OPENAI_QUARTERLY_SEC_MODEL` in the environment.
Once any response exists, changing the model creates an invalid mixed contract.

Start with the ten-document challenge batch. Full extraction remains disabled
until its evidence has been inspected.
"""),
code("""
OPENAI_MODEL = os.getenv('OPENAI_QUARTERLY_SEC_MODEL', 'gpt-5.6-terra').strip()
CHALLENGE_SIZE = 10
API_BATCH_SIZE = 25
RUN_CHALLENGE_BATCH = False
RUN_FULL_BATCH = False

contract = {
    **extraction_rules, 'requested_model': OPENAI_MODEL, 'reasoning_effort': 'low',
    'store': False, 'queue_sha256': queue_hash, 'corpus_manifest_sha256': sha256_path(corpus_manifest_path),
}
contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
contract['contract_sha256'] = contract_hash
contract_path = MANIFESTS / 'critical_minerals_api_contract_v1.json'
if contract_path.exists():
    frozen_contract = json.loads(contract_path.read_text())
    if frozen_contract != contract:
        raise RuntimeError('The frozen API contract differs from current settings. Do not mix responses.')
else:
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\\n')

challenge = pd.concat([
    api_queue.query("document_type == '10-K'").nlargest(CHALLENGE_SIZE//2, 'retrieved_characters'),
    api_queue.query("document_type == '10-Q'").nlargest(CHALLENGE_SIZE-CHALLENGE_SIZE//2, 'retrieved_characters'),
]).drop_duplicates('document_id').head(CHALLENGE_SIZE)
challenge.to_parquet(PROCESSED / 'challenge_queue_frozen.parquet', index=False)
display(pd.DataFrame([{'model':OPENAI_MODEL, 'api_key_present':bool(os.getenv('OPENAI_API_KEY')),
                       'challenge_documents':len(challenge), 'full_queue_documents':len(api_queue),
                       'challenge_enabled':RUN_CHALLENGE_BATCH, 'full_enabled':RUN_FULL_BATCH,
                       'contract_sha256':contract_hash}]))
"""),
md("""
## 7. Run the guarded challenge batch

Set `RUN_CHALLENGE_BATCH=True` in Section 6 and rerun Sections 6–7. Each response
is written immediately, so interruption does not repeat successful calls.
"""),
code("""
existing_ids = {path.stem for path in RESPONSES.glob('*.json')}
remaining_challenge = challenge[~challenge.document_id.astype(str).isin(existing_ids)]
if not RUN_CHALLENGE_BATCH:
    print('Challenge batch disabled. Set RUN_CHALLENGE_BATCH=True in Section 6 when ready.')
elif remaining_challenge.empty:
    print('All challenge documents are already checkpointed.')
else:
    failures = run_batch(remaining_challenge, retrieval_lookup, ontology, schema, prompt, OPENAI_MODEL, RESPONSES)
    failures.to_parquet(RESULTS / 'latest_challenge_failures.parquet', index=False)
    display(failures)
"""),
md("""
## 8. Validate the challenge responses

Every positive exposure must contain an exact quotation found in the locally
stored source document and no longer than 60 words. Review the displayed rows
before enabling the full queue.
"""),
code("""
prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
ontology_hash = hashlib.sha256(json.dumps(ontology, sort_keys=True).encode()).hexdigest()
response_audit, extracted_exposures = validate_checkpointed_responses(
    api_queue, documents, RESPONSES, OPENAI_MODEL, prompt_hash, schema_hash, ontology_hash)
challenge_audit = response_audit[response_audit.document_id.isin(challenge.document_id.astype(str))]
challenge_exposures = extracted_exposures[extracted_exposures.document_id.isin(challenge.document_id.astype(str))] if len(extracted_exposures) else extracted_exposures
display(challenge_audit)
if len(challenge_exposures):
    display(challenge_exposures[['document_id','mechanism','direction','directness','exposure_strength','confidence','evidence_quote','quote_valid']])
print('Challenge response files:', len(challenge_audit), '/', len(challenge))
"""),
md("""
## 9. Run resumable full extraction

Enable this only after the challenge evidence is satisfactory. Each run submits
at most 25 remaining documents and shows progress after every response. Rerun
Sections 6 and 9 until the queue is complete; checkpointed responses are skipped.
"""),
code("""
existing_ids = {path.stem for path in RESPONSES.glob('*.json')}
remaining_queue = api_queue[~api_queue.document_id.astype(str).isin(existing_ids)].head(API_BATCH_SIZE)
status = pd.DataFrame([{'queued':len(api_queue), 'checkpointed':len(set(api_queue.document_id.astype(str)) & existing_ids),
                        'remaining_total':len(api_queue)-len(set(api_queue.document_id.astype(str)) & existing_ids),
                        'next_batch':len(remaining_queue), 'enabled':RUN_FULL_BATCH}])
display(status)
if not RUN_FULL_BATCH:
    print('Full batch disabled. Set RUN_FULL_BATCH=True in Section 6 after challenge review.')
elif remaining_queue.empty:
    print('All queued documents are checkpointed.')
else:
    failures = run_batch(remaining_queue, retrieval_lookup, ontology, schema, prompt, OPENAI_MODEL, RESPONSES)
    failures.to_parquet(RESULTS / 'latest_full_batch_failures.parquet', index=False)
    display(failures)
"""),
md("""
## 10. Consolidate and validate all checkpointed responses

Run this after each batch. Model, prompt, ontology, schema, source-document hash,
logical consistency, evidence length, and exact quotation alignment must pass.
Extraction is complete only when all queued documents have valid responses.
"""),
code("""
response_audit, extracted_exposures = validate_checkpointed_responses(
    api_queue, documents, RESPONSES, OPENAI_MODEL, prompt_hash, schema_hash, ontology_hash)
response_audit.to_parquet(RESULTS / 'response_audit.parquet', index=False)
extracted_exposures.to_parquet(RESULTS / 'document_exposures.parquet', index=False)
completed = set(response_audit.document_id)
validation_status = pd.DataFrame([{
    'queued_documents': len(api_queue), 'response_documents': len(completed),
    'remaining_documents': len(set(api_queue.document_id.astype(str)) - completed),
    'valid_responses': int(response_audit.valid.sum()) if len(response_audit) else 0,
    'invalid_responses': int((~response_audit.valid).sum()) if len(response_audit) else 0,
    'exposure_records': len(extracted_exposures),
    'exact_quote_failures': int((~extracted_exposures.quote_valid).sum()) if len(extracted_exposures) else 0,
}])
display(validation_status)
if len(response_audit) and (~response_audit.valid).any():
    display(response_audit.query('not valid'))
"""),
md("""
## 11. Prepare and confirm the evidence adjudication

The API labels are provisional. Every extracted record is shown with an explicit
proposed decision and rationale. Two records are rejected: generic conflict-
minerals compliance without a sufficiently self-contained thematic quotation,
and wind-policy support without a disclosed critical-mineral connection.

Hubbell's pronoun-only quotation is replaced with a longer exact passage that
names copper and nickel and states the price-and-availability exposure. Review
the full table, then set `CONFIRM_MANUAL_ADJUDICATION=True`. This confirmation is
required before quarterly labels can be constructed.
"""),
code("""
CONFIRM_MANUAL_ADJUDICATION = False

decision_overrides = {
    ('0000046619-23-000163','commodity_input_cost'): (
        'reject', 'Generic conflict-minerals compliance risk; the cited quotation neither names the minerals nor independently establishes material thematic exposure.'),
    ('0001739445-24-000067','energy_power_systems'): (
        'reject', 'Wind-tower policy support is disclosed, but the evidence does not establish a critical-mineral relationship and the quotation exceeds 60 words.'),
}
hubbell_quote = ('Many of our businesses have a dependency on certain basic raw materials needed to produce their products '
                 'including steel, aluminum, brass, copper, bronze, zinc, nickel, plastics, phenols, elastomers and petrochemicals '
                 'as well as purchased electrical and electronic components. Our financial results could be affected by the '
                 'availability and changes in prices of these materials and components.')

evidence_review = extracted_exposures.copy()
evidence_review['review_decision'] = 'accept'
evidence_review['review_note'] = 'The exact filing evidence supports the frozen mechanism under the deliberately broad strategic-minerals scope.'
for key, (decision, note) in decision_overrides.items():
    mask = evidence_review.document_id.eq(key[0]) & evidence_review.mechanism.eq(key[1])
    evidence_review.loc[mask, ['review_decision','review_note']] = [decision, note]
hubbell_mask = evidence_review.document_id.eq('0001628280-24-003792') & evidence_review.mechanism.eq('commodity_input_cost')
evidence_review['adjudicated_evidence_quote'] = evidence_review.evidence_quote
evidence_review.loc[hubbell_mask, 'adjudicated_evidence_quote'] = hubbell_quote
evidence_review['quote_repaired'] = hubbell_mask

source_lookup = {str(item['document_id']): item['text'] for item in documents}
evidence_review['adjudicated_quote_occurrences'] = evidence_review.apply(
    lambda row: source_lookup[str(row.document_id)].count(row.adjudicated_evidence_quote), axis=1)
evidence_review['adjudicated_quote_word_count'] = evidence_review.adjudicated_evidence_quote.str.findall(r'\\b\\w+\\b').str.len()
evidence_review['adjudicated_quote_valid'] = (
    evidence_review.adjudicated_quote_occurrences.ge(1) & evidence_review.adjudicated_quote_word_count.le(60))
evidence_review['decision_valid'] = evidence_review.review_decision.isin(['accept','reject'])
evidence_review.to_parquet(RESULTS / 'evidence_review_proposal.parquet', index=False)

pilot_names = pilot[['permco','firm_name','ticker']].drop_duplicates('permco')
review_display = evidence_review.merge(pilot_names, left_on=evidence_review.firm_id.astype(int), right_on='permco', how='left')
display(review_display[['firm_name','ticker','document_id','mechanism','direction','directness',
                        'exposure_strength','confidence','adjudicated_evidence_quote','review_decision','review_note']])
display(evidence_review.groupby(['review_decision','quote_repaired']).size().rename('records').reset_index())

if CONFIRM_MANUAL_ADJUDICATION:
    if not evidence_review.decision_valid.all(): raise RuntimeError('Every exposure requires an accept/reject decision.')
    if not evidence_review.loc[evidence_review.review_decision.eq('accept'),'adjudicated_quote_valid'].all():
        raise RuntimeError('Every accepted exposure requires a valid exact quotation.')
    evidence_review.to_parquet(RESULTS / 'adjudicated_document_exposures.parquet', index=False)
    adjudication_manifest = {
        'records':len(evidence_review), 'accepted':int(evidence_review.review_decision.eq('accept').sum()),
        'rejected':int(evidence_review.review_decision.eq('reject').sum()),
        'repaired_quotes':int(evidence_review.quote_repaired.sum()), 'confirmed_by_user':True,
        'outcomes_accessed':False,
    }
    (MANIFESTS/'evidence_adjudication_manifest.json').write_text(json.dumps(adjudication_manifest,indent=2,sort_keys=True)+'\\n')
    print('Adjudication confirmed and checkpointed.')
else:
    print('Review the table. Set CONFIRM_MANUAL_ADJUDICATION=True and rerun Section 11 to confirm.')
"""),
md("""
## 12. Construct provisional firm–mechanism–quarter labels

This section runs only after the response queue is complete and the proposed
manual adjudication is explicitly confirmed. For each accepted mechanism,
document relatedness equals strength × confidence. The most recent accepted
evidence available by each cutoff is carried forward until newer accepted
evidence updates it. A later filing's silence does not erase earlier evidence.
Changes are differences from the previous quarter under the frozen 0.10 rule.

This carry-forward rule is a documented pre-outcome amendment. The first draft
used only the latest selected 10-Q, which incorrectly treated a non-repeated
disclosure as evidence that an exposure ended. No return outcome has been opened.

These remain model-generated labels pending the planned human audit.
"""),
code("""
status = validation_status.iloc[0]
adjudicated_path = RESULTS / 'adjudicated_document_exposures.parquet'
if status.remaining_documents:
    raise RuntimeError('Full response coverage is required before quarterly aggregation.')
if not adjudicated_path.exists():
    raise RuntimeError('Confirm the manual adjudication in Section 11 before aggregation.')
valid_exposures = pd.read_parquet(adjudicated_path).query("review_decision == 'accept'").copy()
valid_exposures['evidence_quote'] = valid_exposures['adjudicated_evidence_quote']
valid_exposures['relatedness_score'] = valid_exposures.exposure_strength * valid_exposures.confidence
direction_sign = {'positive':1.0, 'negative':-1.0, 'mixed':0.0, 'unclear':0.0}
valid_exposures['signed_score'] = valid_exposures.relatedness_score * valid_exposures.direction.map(direction_sign)

aggregation_amendment = {
    'version':'quarterly_exposure_aggregation_v1_1',
    'rule':'carry forward the most recent accepted firm-mechanism evidence available by each cutoff',
    'reason':'A later filing that does not repeat an exposure is not evidence that the exposure ceased.',
    'superseded_rule':'use only the latest selected 10-K and latest subsequent 10-Q in each vintage',
    'adopted_before_outcomes':True,
}
(MANIFESTS/'aggregation_amendment_v1_1.json').write_text(json.dumps(aggregation_amendment,indent=2,sort_keys=True)+'\\n')

valid_exposures['firm_id_int'] = pd.to_numeric(valid_exposures.firm_id, errors='raise').astype(int)
valid_exposures['available_date'] = pd.to_datetime(valid_exposures.available_date)
cutoffs = information_panel[['vintage','document_cutoff']].drop_duplicates().copy()
cutoffs['document_cutoff'] = pd.to_datetime(cutoffs.document_cutoff)
mapped = valid_exposures.merge(cutoffs, how='cross')
mapped = mapped[mapped.available_date.le(mapped.document_cutoff)].copy()
mapped['permco'] = mapped['firm_id_int']
mapped = mapped.sort_values(
    ['vintage','permco','mechanism','available_date','relatedness_score','document_id'],
    ascending=[True,True,True,True,True,True])
mechanism_panel = mapped.drop_duplicates(['vintage','permco','mechanism'], keep='last').copy()

all_grid = pd.MultiIndex.from_product(
    [sorted(information_panel.vintage.unique()), sorted(pilot.permco.unique()),
     [item['id'] for item in ontology['mechanisms']]],
    names=['vintage','permco','mechanism']).to_frame(index=False)
keep = ['vintage','permco','mechanism','relatedness_score','signed_score','direction','directness',
        'document_id','document_type','available_date','evidence_quote','reasoning_summary']
mechanism_panel = all_grid.merge(mechanism_panel[keep], on=['vintage','permco','mechanism'], how='left', validate='one_to_one')
mechanism_panel['relatedness_score'] = mechanism_panel.relatedness_score.fillna(0.0)
mechanism_panel['signed_score'] = mechanism_panel.signed_score.fillna(0.0)
mechanism_panel['provisionally_exposed'] = mechanism_panel.relatedness_score.ge(extraction_rules['provisional_exposure_threshold'])
mechanism_panel['vintage_number'] = mechanism_panel.vintage.str[-1].astype(int)
mechanism_panel = mechanism_panel.sort_values(['permco','mechanism','vintage_number'])
mechanism_panel['previous_relatedness_score'] = mechanism_panel.groupby(['permco','mechanism']).relatedness_score.shift()
mechanism_panel['exposure_change'] = mechanism_panel.relatedness_score - mechanism_panel.previous_relatedness_score
mechanism_panel['previous_document_id'] = mechanism_panel.groupby(['permco','mechanism']).document_id.shift()
mechanism_panel['new_evidence_document'] = (
    mechanism_panel.document_id.notna() & mechanism_panel.document_id.ne(mechanism_panel.previous_document_id))
threshold = extraction_rules['material_change_threshold']
mechanism_panel['change_class'] = np.select(
    [mechanism_panel.previous_relatedness_score.isna(),
     mechanism_panel.previous_relatedness_score.eq(0) & mechanism_panel.relatedness_score.ge(extraction_rules['provisional_exposure_threshold']),
     mechanism_panel.new_evidence_document & mechanism_panel.exposure_change.ge(threshold),
     mechanism_panel.new_evidence_document & mechanism_panel.exposure_change.le(-threshold)],
    ['baseline','newly_identified','increased','decreased'], default='stable')
mechanism_panel.to_parquet(PROCESSED / 'provisional_firm_mechanism_quarter_labels.parquet', index=False)

firm_quarter = mechanism_panel.groupby(['vintage','permco'], as_index=False).agg(
    exposure_level=('relatedness_score','max'), signed_exposure=('signed_score',lambda x:x.iloc[np.abs(x).argmax()] if len(x) else 0),
    exposed_mechanisms=('provisionally_exposed','sum'), new_mechanisms=('change_class',lambda x:int((x=='newly_identified').sum())),
    increased_mechanisms=('change_class',lambda x:int((x=='increased').sum())),
    decreased_mechanisms=('change_class',lambda x:int((x=='decreased').sum())))
firm_quarter['provisionally_exposed'] = firm_quarter.exposure_level.ge(extraction_rules['provisional_exposure_threshold'])
firm_quarter.to_parquet(PROCESSED / 'provisional_firm_quarter_exposure_labels.parquet', index=False)
display(firm_quarter.groupby('vintage',as_index=False).agg(
    firms=('permco','nunique'), exposed_firms=('provisionally_exposed','sum'),
    mean_exposure_level=('exposure_level','mean'), new_mechanisms=('new_mechanisms','sum'),
    increased_mechanisms=('increased_mechanisms','sum'), decreased_mechanisms=('decreased_mechanisms','sum')))
"""),
md("""
## 13. Freeze the extraction-stage decision gate

Notebook 11C may load labels only after every queued filing has a response,
every extracted exposure has an explicit accept/reject decision, and every
accepted quotation passes exact-source validation.
"""),
code("""
gates = pd.DataFrame([
    {'gate':'frozen_queue_complete','value':str(int(status.remaining_documents)),'passed':status.remaining_documents==0},
    {'gate':'manual_adjudication_confirmed','value':str(adjudicated_path.exists()),'passed':adjudicated_path.exists()},
    {'gate':'accepted_evidence_quotes_valid','value':str(bool(valid_exposures.adjudicated_quote_valid.all())),
     'passed':bool(valid_exposures.adjudicated_quote_valid.all())},
    {'gate':'quarterly_labels_created','value':str((PROCESSED/'provisional_firm_quarter_exposure_labels.parquet').exists()),
     'passed':(PROCESSED/'provisional_firm_quarter_exposure_labels.parquet').exists()},
    {'gate':'outcomes_remain_unopened','value':'False','passed':True},
])
gates.to_parquet(MANIFESTS / 'extraction_decision_gates.parquet', index=False)
display(gates)
manifest = {
    'stage':'quarterly_sec_exposure_extraction', 'contract_sha256':contract_hash,
    'queue_sha256':queue_hash, 'requested_model':OPENAI_MODEL, 'queued_documents':len(api_queue),
    'response_documents':int(status.response_documents), 'exposure_records':int(status.exposure_records),
    'all_gates_passed':bool(gates.passed.all()),
    'manual_review_complete':adjudicated_path.exists(),
    'outcomes_accessed':False,
}
(MANIFESTS/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n')
assert gates.passed.all(), 'Extraction stage is incomplete; inspect the gate table.'
print('Automated extraction gates passed. Preserve the evidence-review sheet for human adjudication.')
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb['metadata'] = {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python','version':'3.9'}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
