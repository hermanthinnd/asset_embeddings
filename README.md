# Asset Embeddings Replication and Extension

This repository contains an end-to-end replication of Gabaix, Koijen, Richmond, and Yogo, *Asset Embeddings* (BFI Working Paper 2025-51), followed by an extension studying whether active and passive mutual-fund holdings produce embeddings with different economic effectiveness.

## Project structure

- `Full_Replication_80Q.ipynb` — constructs the 2005Q1–2024Q4 replication pipeline and the RS and Word2Vec embeddings.
- `Robustness_and_Sanity_Checks.ipynb` — evaluates leakage, split sensitivity, target construction, and embedding integrity.
- `Remaining_Benchmarks.ipynb` — implements return-comovement and managed-portfolio tests.
- `Replication_Summary.ipynb` — readable, results-focused account of the complete replication.
- `Active_Passive_Extension.ipynb` — constructs Active Share and turnover classifications and tests embedding effectiveness across investor types.
- `Active_Passive_Extension_Summary.ipynb` — readable, results-focused account of the extension.
- `Pipeline_Validation_3Q.ipynb` — three-quarter pipeline validation.
- `geopolitical_exposure/` — transcript-ready proof of concept using pre-event corporate text and holdings embeddings to identify direct and latent geopolitical exposures.
- `src/`, `scripts/`, and `tests/` — reusable project code and supporting checks.

## Data availability

The analysis uses licensed WRDS datasets, including Thomson Reuters S12/S34, CRSP, and Compustat-derived inputs. Those source data and all generated row-level holdings, matrices, and embeddings are intentionally excluded from Git because they are large and may be subject to redistribution restrictions.

The repository retains the executable notebooks, methodological documentation, research papers used in the project, and rendered summary figures. A user with the required WRDS subscriptions can reconstruct the local artifacts by running the pipeline notebooks in sequence.

## Recommended reading order

1. `Replication_Summary.ipynb`
2. `Active_Passive_Extension_Summary.ipynb`
3. The underlying pipeline and benchmark notebooks for implementation details

The geopolitical application begins with `geopolitical_exposure/00_Research_Design.ipynb` and `geopolitical_exposure/01_Event_and_Data_Audit.ipynb`. Its text adapter accepts SEC filings now and earnings-call transcripts later without changing the downstream seed, embedding, or event-study interfaces.

## Main findings

Holdings-based embeddings explain economically meaningful variation in relative valuation and contribute information beyond standard firm characteristics. Model performance is task-specific: global recommender systems are strongest for relative valuation, while Word2Vec is particularly effective for masked portfolio prediction.

In the active/passive extension, market-distinctive ownership is more important than trading frequency. Passive-like portfolios are easier to reconstruct, but active portfolios produce more valuation-informative low-dimensional embeddings.
