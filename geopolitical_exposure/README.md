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
10. `05_Text_vs_Embedding_Incremental_Discovery.ipynb` — tests whether leave-self-out embedding scores identify event-sensitive firms whose own reviewed filings produced no direct channel label.
11. `06_Geopolitical_Exposure_Portfolios.ipynb` — converts direct text, seed-excluded embedding expansion, characteristics, and their fixed combination into matched-size long–short portfolios for an explicitly exploratory Ukraine event test.
12. `07_Blinded_Exposure_Retrieval_Benchmark.ipynb` — freezes a return-blind comparison of embedding-retrieved firms against covariate-matched controls and prepares identical pre-event relationship-validation packets.
13. `07B_Blinded_Relationship_Classification.ipynb` — runs checkpointed, structured, evidence-grounded classification on the blinded candidate queue and stops for human review before unblinding.
14. `08_Unblinded_Relationship_Discovery_Evaluation.ipynb` — freezes conservative admissibility while blinded, then compares embedding-selected candidates with matched controls using overall, channel-level, paired inference, retrieval-opportunity diagnostics, matching audits, and evidence-strength sensitivity.
15. `09_Directional_Geopolitical_Embedding.ipynb` — tests an outcome-free beneficiary-versus-adverse centroid direction with leave-one-out validation, permutation inference, conventional baselines, industry residualization, investor-group comparisons, dimension sensitivity, and mixed-firm diagnostics.
16. `10_Matched_Pairwise_Contrastive_Learning.ipynb` — learns a regularized direction from industry-and-characteristic-matched positive-minus-negative pairs, using leave-one-pair-out validation, within-pair sign-flip inference, exact-industry robustness, and no event returns.
17. `quarterly_sec_pilot/11A_Quarterly_SEC_Pilot_Corpus.ipynb` — begins a separate SEC-only, 100-firm quarterly pilot for 2024, freezing the sample and point-in-time 10-K/10-Q information sets before any exposure extraction or outcome evaluation.
18. `quarterly_sec_pilot/11B_Quarterly_SEC_Exposure_Extraction.ipynb` — freezes a detailed critical-minerals ontology and evidence contract, retrieves point-in-time passages, runs guarded structured extraction, and separates provisional exposure levels from quarter-to-quarter changes.
19. `quarterly_sec_pilot/11C_Three_Seed_Specification_Propagation.ipynb` through `11F_2025_2026_Forward_Performance.ipynb` — compare broad-market, sector-enriched, and combined seed designs; evaluate quarterly portfolios; run robustness checks; and examine forward performance.
20. `sp500_critical_minerals/12A_SP500_Point_in_Time_Universe.ipynb` through `14G_2024Q1_Portfolio_Persistence_Through_2026.ipynb` — scale the critical-minerals workflow to point-in-time S&P 500 universes, calibrate and run the evidence-grounded LLM extraction, propagate labels through embeddings, freeze portfolios, test thematic-factor sensitivity, and evaluate the 2024Q1 portfolios through August 2026.

The V1 experiment quarantines event outcomes through Notebook 03. The 02B expansion was designed after aggregate V1 event results were inspected, so any revised same-event V2 analysis is explicitly exploratory and requires confirmation on a separate held-out event.

## Text-corpus contract

Any document source must normalize to the schema in `schemas/document_schema.json`. SEC filings can be used for the proof of concept. Earnings-call transcripts can later be added as `document_type="earnings_call"` without changing downstream code.

## API security

Never paste an API key into chat, source code, notebooks, configuration files, or committed outputs. Export `OPENAI_API_KEY` in the local shell. The extraction pipeline will use the Responses API with a strict JSON schema, `store=False`, deterministic prompts where supported, local response logging, and explicit model/prompt hashes.

Raw documents, processed text, model responses, manifests containing row-level results, and generated numerical outputs are excluded from Git.

The critical-minerals work remains nested here because it is an application of the same text-to-seed and holdings-embedding framework. Its detailed sequence, main results, and interpretation limits are documented in `sp500_critical_minerals/README.md`.
The presentation-ready entry point is `sp500_critical_minerals/Critical_Minerals_Extension_Summary.ipynb`.

See `API_AND_DOCUMENT_SETUP.md` for secure key configuration and the small transcript-test handoff format.
