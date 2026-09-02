# Geopolitical Exposure from Text and Holdings Embeddings

This extension tests whether pre-event corporate text identifies direct geopolitical exposures and whether holdings-based asset embeddings propagate those signals to firms whose exposure is not obvious in their own disclosures.

The first pilot candidate is the Russian invasion of Ukraine on February 24, 2022. Event selection is finalized only after the point-in-time coverage audit; returns are not used to select the event, channels, seeds, thresholds, or embedding specification.

## Sequence

1. `00_Research_Design.ipynb` — freezes hypotheses, temporal rules, comparison models, schemas, and leakage safeguards.
2. `01_Event_and_Data_Audit.ipynb` — audits the 2021Q4 embeddings, CRSP identifiers/returns, and available point-in-time text.
3. `02_Text_Exposure_Pipeline.ipynb` — retrieval-first extraction, manual adjudication, and seed freezing.
4. `02B_Text_Corpus_Expansion.ipynb` — expands the pre-event corpus to a deterministic 300-firm sample and freezes the V2 extraction queue.
5. `03_Embedding_Expansion.ipynb` — expands the original pilot seeds using all-investor and behavioral-group embeddings.
6. `03B_Expanded_Label_Propagation.ipynb` — constructs the audited V2 seed panel, performs leave-one-seed-out validation, and propagates expanded labels through five holdings embeddings.
7. `04_Event_Study_Evaluation.ipynb` — evaluates the original pilot using frozen abnormal-return and placebo tests.
8. `04B_Expanded_Label_Event_Study.ipynb` — evaluates the expanded labels on a common non-seed sample with controls, placebos, and multiple-testing adjustments.
9. `Geopolitical_Exposure_Extension_Summary.ipynb` — static, presentation-style account of the rationale, exact extraction prompt, methodology, results, limitations, and conclusions.

The V1 experiment quarantines event outcomes through Notebook 03. The 02B expansion was designed after aggregate V1 event results were inspected, so any revised same-event V2 analysis is explicitly exploratory and requires confirmation on a separate held-out event.

## Text-corpus contract

Any document source must normalize to the schema in `schemas/document_schema.json`. SEC filings can be used for the proof of concept. Earnings-call transcripts can later be added as `document_type="earnings_call"` without changing downstream code.

## API security

Never paste an API key into chat, source code, notebooks, configuration files, or committed outputs. Export `OPENAI_API_KEY` in the local shell. The extraction pipeline will use the Responses API with a strict JSON schema, `store=False`, deterministic prompts where supported, local response logging, and explicit model/prompt hashes.

Raw documents, processed text, model responses, manifests containing row-level results, and generated numerical outputs are excluded from Git.

See `API_AND_DOCUMENT_SETUP.md` for secure key configuration and the small transcript-test handoff format.
