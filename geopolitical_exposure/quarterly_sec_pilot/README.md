# Quarterly SEC–Embedding Pilot

This work package tests whether a fixed thematic research question produces a
changing firm portfolio as public SEC disclosures and holdings-based asset
embeddings evolve through 2024.

## Frozen pilot sequence

1. `11A_Quarterly_SEC_Pilot_Corpus.ipynb` freezes the design, selects 100 firms,
   and constructs a point-in-time 10-K/10-Q corpus for four 2024 cutoffs.
2. `11B_Quarterly_SEC_Exposure_Extraction.ipynb` retrieves passages, freezes the
   critical-minerals prompt/model/schema, runs checkpointed API extraction, and
   creates separate exposure-level and exposure-change labels.
3. `11B_Sector_Enriched_Supplement.ipynb` preserves the broad-market sample and
   adds a separately labelled 50-firm exploratory industry supplement when the
   original screen provides too few seeds for reliable retrieval evaluation.
4. `11C_Three_Seed_Specification_Propagation.ipynb` compares broad-market,
   sector-enriched, and combined seed specifications while retaining their
   distinct provenance.
5. `11D_Quarterly_Portfolio_Evaluation.ipynb` measures constituent changes,
   turnover, neighbour stability, and publication-lagged forward performance.
6. `11E_Robustness_and_Seed_Design_Comparison.ipynb` evaluates whether the
   conclusions depend on seed breadth, portfolio size, or propagation design.
7. `11F_2025_2026_Forward_Performance.ipynb` extends the frozen portfolios into
   the later return window and audits data-source coverage.

The first notebook must pass its corpus and timing gates before Notebook 11B is
created. This prevents downstream choices from being influenced by outcomes.

## Timing interpretation

Quarter-end embedding vintages are descriptive holdings-period snapshots. They
are not assumed to have been publicly investable at quarter end. Any return
evaluation must use separately audited public-availability and formation dates.

## Local data and security

Raw filings, derived passages, API responses, and row-level outputs remain
local and are checkpointed. Never store API keys in this directory or in a
notebook. SEC requests use `SEC_USER_AGENT`; OpenAI requests later use
`OPENAI_API_KEY` from the environment.

This pilot is retained as the methodological development stage. The larger,
point-in-time S&P 500 validation is in `../sp500_critical_minerals/`.
