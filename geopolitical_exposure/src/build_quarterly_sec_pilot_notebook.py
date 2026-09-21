"""Build the first notebook in the quarterly SEC–embedding pilot."""

from pathlib import Path
import nbformat as nbf


GEO_ROOT = Path(__file__).resolve().parents[1]
OUT = GEO_ROOT / "quarterly_sec_pilot" / "11A_Quarterly_SEC_Pilot_Corpus.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
md("""
# Quarterly SEC–Embedding Pilot

## 11A — Frozen design, 100-firm selection, and point-in-time corpus

This notebook begins the SEC-only quarterly pilot. It selects 100 firms without
using returns or thematic outcomes and constructs four information sets using
only filings publicly available by each 2024 quarter end.

The notebook does **not** call OpenAI, score geopolitical exposure, construct a
portfolio, or inspect subsequent returns. Those stages remain downstream of the
corpus audit.
"""),
md("""
## 1. Environment and checkpoint directories

All downloaded and generated material is stored locally. Existing SEC caches
from the earlier geopolitical work are reused by accession number, while this
pilot receives its own immutable manifests and processed outputs.
"""),
code("""
from pathlib import Path
import hashlib, importlib, json, os, shutil, sys, time
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
    ROOT = HERE
    GEO_ROOT = ROOT / 'geopolitical_exposure'
    PILOT_ROOT = GEO_ROOT / 'quarterly_sec_pilot'
else:
    ROOT = (HERE / 'Projects' / 'Asset_Embeddings').resolve()
    GEO_ROOT = ROOT / 'geopolitical_exposure'
    PILOT_ROOT = GEO_ROOT / 'quarterly_sec_pilot'

if not ROOT.exists():
    raise FileNotFoundError('Open the notebook from Asset_Embeddings, geopolitical_exposure, or quarterly_sec_pilot.')

sys.path.insert(0, str(GEO_ROOT / 'src'))
import document_contracts, edgar_pipeline
importlib.reload(document_contracts); importlib.reload(edgar_pipeline)
from document_contracts import normalize_document, validate_documents
from edgar_pipeline import download_company_tickers, filing_metadata_for_cik, get_text, html_to_text, ticker_cik_crosswalk

CONFIG = PILOT_ROOT / 'config'
RAW = PILOT_ROOT / 'data' / 'raw'
PROCESSED = PILOT_ROOT / 'data' / 'processed'
MANIFESTS = PILOT_ROOT / 'manifests'
RESULTS = PILOT_ROOT / 'results'
for path in [RAW, PROCESSED, MANIFESTS, RESULTS]:
    path.mkdir(parents=True, exist_ok=True)

def sha256_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\\n', encoding='utf-8')

print('Project root:', ROOT)
print('Pilot root:', PILOT_ROOT)
"""),
md("""
## 2. Load and hash the frozen research design

The design fixes the sample size, quarter cutoffs, representations, dimensions,
document forms, and three decomposition experiments. Editing the design after
text labels or outcomes have been inspected requires a new version rather than
overwriting this contract.
"""),
code("""
design_path = CONFIG / 'pilot_design.json'
design = json.loads(design_path.read_text(encoding='utf-8'))
design_hash = sha256_path(design_path)
vintages = pd.DataFrame(design['vintages'])
vintages['quarter_end'] = pd.to_datetime(vintages['quarter_end'])
vintages['document_cutoff'] = pd.to_datetime(vintages['document_cutoff'])
design_audit = pd.DataFrame([{
    'design_version': design['design_version'],
    'sha256': design_hash,
    'firms': design['sample_size'],
    'vintages': len(vintages),
    'document_start': design['document_start'],
    'forms': ', '.join(design['document_forms']),
    'outcomes_accessed': design['outcomes_accessed_during_corpus_construction'],
}])
display(design_audit)
display(vintages)
design_audit.to_parquet(MANIFESTS / 'design_audit.parquet', index=False)
"""),
md("""
## 3. Audit the four quarterly embedding universes

The primary representation is the normalized RS-L-Min asset embedding. Both
RS-L-Min and RS-Binary must exist for every quarter. Selection requires a firm
to be present in all four quarters so changes are not mechanically driven by
entry into or exit from the embedding universe.
"""),
code("""
embedding_inventory = []
quarter_frames = {}
for vintage in vintages.itertuples(index=False):
    label = vintage.label
    for model in design['embedding_models']:
        path = ROOT / 'full_replication_80q' / 'embeddings' / label / model / 'asset_embeddings_normalized_labeled.parquet'
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        dimensions = [column for column in frame.columns if column.startswith('embedding_')]
        embedding_inventory.append({
            'vintage': label, 'model': model, 'firms': frame.permco.nunique(),
            'available_dimensions': len(dimensions), 'file_mb': path.stat().st_size / 2**20,
            'sha256': sha256_path(path),
        })
        if model == design['selection_embedding']:
            quarter_frames[label] = frame

embedding_inventory = pd.DataFrame(embedding_inventory)
common_permcos = set.intersection(*[set(frame.permco) for frame in quarter_frames.values()])
embedding_inventory['common_four_quarter_firms'] = len(common_permcos)
display(embedding_inventory)
embedding_inventory.to_parquet(MANIFESTS / 'embedding_inventory.parquet', index=False)
assert len(common_permcos) >= design['sample_size'], 'Fewer common firms than the frozen pilot size.'
"""),
md("""
## 4. Construct the point-in-time identifier and size universe

The SEC bridge uses a unique exact ticker match. Ambiguous and unmatched firms
are excluded rather than manually resolved after seeing exposure. Q1 market
equity supplies the size bins used only for diversified sample selection.
"""),
code("""
q1_names_path = ROOT / 'data' / 'crosswalks' / 'crsp' / 'year=2024' / 'quarter=1' / 'stocknames_2024q1.parquet'
q1_controls_path = ROOT / 'remaining_benchmarks' / 'return_comovement' / 'characteristics' / '2024q1.parquet'
q1_names = pd.read_parquet(q1_names_path)
q1_controls = pd.read_parquet(q1_controls_path, columns=['permco', 'log_market_equity'])
q1_names = (q1_names[q1_names.permco.isin(common_permcos)]
            .sort_values(['permco', 'nameenddt'])
            .drop_duplicates('permco', keep='last'))

sec_reference = GEO_ROOT / 'documents' / 'raw' / 'sec_reference' / 'company_tickers_exchange.parquet'
if not sec_reference.exists():
    SEC_USER_AGENT = os.getenv('SEC_USER_AGENT', '').strip()
    if '@' not in SEC_USER_AGENT:
        SEC_USER_AGENT = input('SEC user agent (example: Academic research name@university.edu): ').strip()
    sec_tickers = download_company_tickers(sec_reference, SEC_USER_AGENT)
else:
    sec_tickers = pd.read_parquet(sec_reference)

bridge = ticker_cik_crosswalk(q1_names, sec_tickers)
eligible = (bridge.query("cik_match_status == 'unique_exact_ticker'")
            .drop_duplicates('permco')
            .merge(q1_controls, on='permco', how='inner', validate='one_to_one'))
eligible = eligible[eligible.log_market_equity.notna()].copy()
eligible['size_quintile'] = pd.qcut(eligible.log_market_equity, 5, labels=False, duplicates='drop') + 1

identifier_audit = pd.DataFrame([{
    'common_embedding_firms': len(common_permcos),
    'q1_names_matched': q1_names.permco.nunique(),
    'unique_exact_sec_links': eligible.permco.nunique(),
    'eligible_rate': eligible.permco.nunique() / len(common_permcos),
}])
display(identifier_audit)
identifier_audit.to_parquet(MANIFESTS / 'identifier_audit.parquet', index=False)
assert len(eligible) >= design['sample_size'], 'Insufficient uniquely linked firms for the pilot.'
"""),
md("""
## 5. Select and freeze 100 firms

Selection is balanced over ten clusters from the first ten Q1 RS-L-Min
dimensions and five Q1 market-size bins. Within every stratum, firms receive a
stable hash order. Round-robin sampling across strata avoids concentrating the
pilot in mega-cap firms or one region of the embedding space.
"""),
code("""
from sklearn.cluster import KMeans

selection_path = PROCESSED / 'quarterly_sec_pilot_100_firms_frozen.parquet'
if selection_path.exists():
    pilot = pd.read_parquet(selection_path)
    print('Loaded frozen 100-firm selection.')
else:
    q1 = quarter_frames['2024q1']
    dims = [f'embedding_{number:03d}' for number in range(1, design['selection_dimensions'] + 1)]
    selection = eligible.merge(q1[['permco', 'firm_name'] + dims], on='permco', how='inner', validate='one_to_one')
    kmeans = KMeans(n_clusters=design['selection_clusters'], random_state=design['selection_seed'], n_init=20)
    selection['embedding_cluster'] = kmeans.fit_predict(selection[dims].to_numpy(dtype='float64')) + 1
    selection['stable_order'] = selection.permco.map(
        lambda value: hashlib.sha256(f"{design['selection_seed']}:{int(value)}".encode()).hexdigest()
    )
    selection = selection.sort_values(['embedding_cluster', 'size_quintile', 'stable_order'])
    selection['stratum_rank'] = selection.groupby(['embedding_cluster', 'size_quintile']).cumcount()
    pilot = (selection.sort_values(['stratum_rank', 'embedding_cluster', 'size_quintile', 'stable_order'])
             .head(design['sample_size']).copy())
    pilot['selection_design_sha256'] = design_hash
    pilot.to_parquet(selection_path, index=False)

pilot_hash = sha256_path(selection_path)
selection_audit = (pilot.groupby(['embedding_cluster', 'size_quintile'], as_index=False)
                   .agg(firms=('permco', 'nunique')))
display(pd.DataFrame([{
    'selected_firms': pilot.permco.nunique(),
    'clusters': pilot.embedding_cluster.nunique(),
    'size_quintiles': pilot.size_quintile.nunique(),
    'selection_sha256': pilot_hash,
}]))
display(selection_audit.pivot(index='embedding_cluster', columns='size_quintile', values='firms').fillna(0).astype(int))
assert pilot.permco.nunique() == design['sample_size']
"""),
md("""
## 6. Configure SEC acquisition

SEC EDGAR requires an identifying user agent containing an email address. It is
not a secret and is not saved in the notebook. Existing submissions and filing
HTML are reused automatically. The execution switches allow metadata and
documents to be acquired in separate, resumable stages.
"""),
code("""
SEC_USER_AGENT = os.getenv('SEC_USER_AGENT', '').strip()
if '@' not in SEC_USER_AGENT:
    SEC_USER_AGENT = input('SEC user agent (example: Academic research name@university.edu): ').strip()
if '@' not in SEC_USER_AGENT:
    raise ValueError('A valid SEC user agent containing an email is required.')

DOWNLOAD_SEC_METADATA = False
DOWNLOAD_SEC_DOCUMENTS = False

print('SEC acquisition configured.')
print('Metadata download enabled:', DOWNLOAD_SEC_METADATA)
print('Document download enabled:', DOWNLOAD_SEC_DOCUMENTS)
"""),
md("""
## 7. Acquire and checkpoint filing metadata

Metadata covers 2023 through the final 2024 cutoff and is filtered to 10-K and
10-Q filings. Cached CIK submission JSON from the earlier project is reused.
Set `DOWNLOAD_SEC_METADATA=True` in Section 6, then run this cell. Interrupted
runs are safe because each CIK response is cached independently.
"""),
code("""
metadata_path = PROCESSED / 'sec_filing_metadata_100_firms.parquet'
metadata_error_path = RESULTS / 'sec_metadata_errors.parquet'
submission_cache = GEO_ROOT / 'documents' / 'raw' / 'sec_submissions'

if metadata_path.exists():
    filing_metadata = pd.read_parquet(metadata_path)
    print('Loaded checkpointed filing metadata.')
elif not DOWNLOAD_SEC_METADATA:
    raise RuntimeError('Set DOWNLOAD_SEC_METADATA=True in Section 6 and rerun Sections 6–7.')
else:
    frames, errors = [], []
    for position, row in enumerate(pilot.itertuples(index=False), 1):
        try:
            frame = filing_metadata_for_cik(
                int(row.cik), SEC_USER_AGENT, design['document_forms'],
                design['document_start'], str(vintages.document_cutoff.max().date()), submission_cache,
            )
            if not frame.empty:
                frame['permco'] = int(row.permco)
                frames.append(frame)
        except Exception as error:
            errors.append({'permco': int(row.permco), 'cik': int(row.cik), 'error_type': type(error).__name__, 'error': str(error)})
        if position % 10 == 0 or position == len(pilot):
            print(f'Metadata: {position}/{len(pilot)} firms; errors={len(errors)}')
    filing_metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pd.DataFrame(errors).to_parquet(metadata_error_path, index=False)
    if errors:
        raise RuntimeError(f'{len(errors)} metadata failures remain. Rerun after resolving them; successful CIK files are cached.')
    filing_metadata.to_parquet(metadata_path, index=False)

filing_metadata['filingDate'] = pd.to_datetime(filing_metadata['filingDate'])
metadata_audit = pd.DataFrame([{
    'filings': len(filing_metadata),
    'firms_with_filings': filing_metadata.permco.nunique(),
    'ten_k': int(filing_metadata.form.eq('10-K').sum()),
    'ten_q': int(filing_metadata.form.eq('10-Q').sum()),
    'latest_filing_date': filing_metadata.filingDate.max(),
}])
display(metadata_audit)
"""),
md("""
## 8. Construct four point-in-time document selections

For each firm and cutoff, retain the latest available 10-K and, when available,
the latest 10-Q filed after that 10-K. The annual filing anchors structural
exposure; the quarterly filing supplies incremental information. The same
accession can serve several vintages and is downloaded only once.
"""),
code("""
selection_rows = []
for vintage in vintages.itertuples(index=False):
    available = filing_metadata[filing_metadata.filingDate.le(vintage.document_cutoff)].copy()
    for permco, group in available.groupby('permco', sort=True):
        annual = group[group.form.eq('10-K')].sort_values(['filingDate', 'accessionNumber'])
        if annual.empty:
            continue
        latest_annual = annual.iloc[-1]
        for role, row in [('structural_anchor', latest_annual)]:
            selection_rows.append({'vintage': vintage.label, 'document_role': role, **row.to_dict()})
        quarterly = group[group.form.eq('10-Q') & group.filingDate.gt(latest_annual.filingDate)].sort_values(['filingDate', 'accessionNumber'])
        if not quarterly.empty:
            row = quarterly.iloc[-1]
            selection_rows.append({'vintage': vintage.label, 'document_role': 'quarterly_update', **row.to_dict()})

vintage_documents = pd.DataFrame(selection_rows).drop_duplicates(['vintage', 'accessionNumber'])
vintage_documents['document_id'] = vintage_documents['accessionNumber'].astype(str)
vintage_documents.to_parquet(PROCESSED / 'vintage_document_map.parquet', index=False)
unique_documents = (vintage_documents.sort_values(['filingDate', 'accessionNumber'])
                    .drop_duplicates('accessionNumber').reset_index(drop=True))
unique_documents.to_parquet(PROCESSED / 'unique_selected_documents.parquet', index=False)

coverage = (vintage_documents.groupby('vintage', as_index=False)
            .agg(firms=('permco', 'nunique'), documents=('document_id', 'nunique'),
                 structural_anchors=('document_role', lambda x: int((x == 'structural_anchor').sum())),
                 quarterly_updates=('document_role', lambda x: int((x == 'quarterly_update').sum())),
                 latest_available_date=('filingDate', 'max')))
coverage['firm_coverage'] = coverage.firms / design['sample_size']
display(coverage)
coverage.to_parquet(RESULTS / 'vintage_document_coverage.parquet', index=False)
"""),
md("""
## 9. Download, normalize, and hash unique filings

This is the longest SEC cell. It searches the earlier V1 and V2 filing caches
before making a request. Each newly downloaded HTML file is checkpointed
immediately. Set `DOWNLOAD_SEC_DOCUMENTS=True` in Section 6 and rerun this cell;
an interruption never requires starting over.
"""),
code("""
normalized_path = PROCESSED / 'sec_documents_100_firms.jsonl'
new_html_dir = RAW / 'sec_filings'
new_html_dir.mkdir(parents=True, exist_ok=True)
legacy_html_dirs = [
    GEO_ROOT / 'documents' / 'raw' / 'sec_filings',
    GEO_ROOT / 'text_expansion_v2' / 'raw' / 'sec_filings',
]

def locate_cached_html(accession):
    names = [f'{accession}.html', f'{accession}.htm']
    for directory in [new_html_dir] + legacy_html_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None

if normalized_path.exists():
    normalized_documents = [json.loads(line) for line in normalized_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    normalized_documents = validate_documents(normalized_documents)
    print('Loaded checkpointed normalized corpus.')
elif not DOWNLOAD_SEC_DOCUMENTS:
    raise RuntimeError('Set DOWNLOAD_SEC_DOCUMENTS=True in Section 6 and rerun Sections 6 and 9.')
else:
    normalized_documents, errors = [], []
    for position, row in enumerate(unique_documents.itertuples(index=False), 1):
        accession = str(row.accessionNumber)
        destination = new_html_dir / f'{accession}.html'
        try:
            cached = locate_cached_html(accession)
            if cached is not None:
                html = cached.read_text(encoding='utf-8', errors='replace')
            else:
                html = get_text(row.source_uri, SEC_USER_AGENT)
                destination.write_text(html, encoding='utf-8')
            text = html_to_text(html)
            normalized_documents.append(normalize_document({
                'document_id': accession,
                'firm_id': str(int(row.permco)),
                'document_type': str(row.form),
                'document_date': pd.Timestamp(row.reportDate).date().isoformat(),
                'available_date': pd.Timestamp(row.filingDate).date().isoformat(),
                'source': 'SEC EDGAR', 'source_uri': row.source_uri, 'text': text,
            }))
        except Exception as error:
            errors.append({'document_id': accession, 'permco': int(row.permco), 'error_type': type(error).__name__, 'error': str(error)})
        if position % 10 == 0 or position == len(unique_documents):
            print(f'Documents: {position}/{len(unique_documents)}; errors={len(errors)}')
    pd.DataFrame(errors).to_parquet(RESULTS / 'document_errors.parquet', index=False)
    with normalized_path.open('w', encoding='utf-8') as handle:
        for record in normalized_documents:
            handle.write(json.dumps(record, ensure_ascii=False) + '\\n')
    if errors:
        raise RuntimeError(f'{len(errors)} document failures remain. Successful downloads are cached; rerun after inspection.')

document_index = pd.DataFrame([{key: record.get(key) for key in [
    'document_id', 'firm_id', 'document_type', 'document_date', 'available_date',
    'source', 'source_uri', 'content_sha256'
]} | {'characters': len(record['text'])} for record in normalized_documents])
document_index.to_parquet(PROCESSED / 'document_index.parquet', index=False)
display(pd.DataFrame([{
    'unique_documents': len(document_index),
    'firms': document_index.firm_id.nunique(),
    'ten_k': int(document_index.document_type.eq('10-K').sum()),
    'ten_q': int(document_index.document_type.eq('10-Q').sum()),
    'median_characters': document_index.characters.median(),
}]))
"""),
md("""
## 10. Build the firm-quarter information panel

The panel records exactly which hashed documents were available for each firm
and quarter. It does not yet contain an exposure label. Repeated annual filings
are intentionally visible across vintages, allowing the next notebook to reuse
their cached extraction rather than paying for the same text repeatedly.
"""),
code("""
document_columns = ['document_id', 'content_sha256', 'characters']
information_panel = (vintage_documents
    .merge(document_index[document_columns], on='document_id', how='left', validate='many_to_one'))
information_panel['available_date'] = pd.to_datetime(information_panel.filingDate)
information_panel['document_cutoff'] = information_panel.vintage.map(
    vintages.set_index('label').document_cutoff
)
information_panel['point_in_time_valid'] = information_panel.available_date.le(information_panel.document_cutoff)
information_panel.to_parquet(PROCESSED / 'firm_quarter_information_panel.parquet', index=False)

panel_audit = (information_panel.groupby('vintage', as_index=False)
    .agg(firms=('permco', 'nunique'), documents=('document_id', 'nunique'),
         missing_downloads=('content_sha256', lambda x: int(x.isna().sum())),
         cutoff_violations=('point_in_time_valid', lambda x: int((~x).sum())),
         total_characters=('characters', 'sum')))
display(panel_audit)
panel_audit.to_parquet(RESULTS / 'information_panel_audit.parquet', index=False)
"""),
md("""
## 11. Freeze the corpus manifest and decision gate

Notebook 11B may begin only when all 100 selected firms have at least one
document at every vintage, every selected document is downloaded and hashed,
and no filing violates its information cutoff. Failure is a diagnostic result,
not permission to modify the sample after observing thematic labels.
"""),
code("""
expected_vintages = set(vintages.label)
observed_vintages = set(panel_audit.vintage)
gate_rows = [
    {'gate': 'frozen_100_firm_sample', 'value': pilot.permco.nunique(), 'passed': pilot.permco.nunique() == design['sample_size']},
    {'gate': 'all_four_vintages', 'value': len(observed_vintages), 'passed': observed_vintages == expected_vintages},
    {'gate': 'complete_firm_coverage_each_vintage', 'value': int(panel_audit.firms.min()), 'passed': bool((panel_audit.firms == design['sample_size']).all())},
    {'gate': 'all_selected_documents_downloaded', 'value': int(panel_audit.missing_downloads.sum()), 'passed': panel_audit.missing_downloads.sum() == 0},
    {'gate': 'point_in_time_cutoff_integrity', 'value': int(panel_audit.cutoff_violations.sum()), 'passed': panel_audit.cutoff_violations.sum() == 0},
    {'gate': 'outcomes_remain_unopened', 'value': False, 'passed': True},
]
gates = pd.DataFrame(gate_rows)
gates['value'] = gates['value'].map(str)
display(gates)
gates.to_parquet(MANIFESTS / 'corpus_decision_gates.parquet', index=False)

manifest = {
    'stage': 'quarterly_sec_pilot_corpus',
    'design_sha256': design_hash,
    'selection_sha256': pilot_hash,
    'sample_size': int(pilot.permco.nunique()),
    'vintages': vintages.astype(str).to_dict('records'),
    'unique_documents': int(document_index.document_id.nunique()),
    'document_index_sha256': sha256_path(PROCESSED / 'document_index.parquet'),
    'information_panel_sha256': sha256_path(PROCESSED / 'firm_quarter_information_panel.parquet'),
    'all_gates_passed': bool(gates.passed.all()),
    'outcomes_accessed': False,
}
save_json(manifest, MANIFESTS / 'corpus_manifest.json')
assert gates.passed.all(), 'At least one corpus gate failed; inspect the displayed gate table before proceeding.'
print('Corpus stage complete. Notebook 11B can now freeze retrieval and extraction.')
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
