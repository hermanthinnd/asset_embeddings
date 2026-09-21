# S&P 500 Critical-Minerals Discovery

This extension compares direct SEC-filing exposure extraction, embedding-only
discovery, and a prespecified hybrid signal within the historical S&P 500.

For a presentation-ready account of the complete design, chronology, results,
and limitations, begin with `Critical_Minerals_Extension_Summary.ipynb`.

The experiment is staged to keep information and spending boundaries explicit:

1. `12A_SP500_Point_in_Time_Universe.ipynb` freezes historical constituents,
   identifiers, embedding coverage, portfolio rules, and cost gates. It does
   not download filings, call OpenAI, or access subsequent returns.
2. `12B_SP500_SEC_Corpus.ipynb` resolves remaining SEC identifiers, inventories
   point-in-time filings, downloads each unique filing only once, reuses prior
   caches, and constructs the retrieval-filtered API queue.
3. `12C_Blinded_Prompt_Calibration.ipynb` compares the original extraction
   contract with an analyst-style v2 contract on the same frozen 25 documents.
   It creates a blinded human-review packet and does not load returns or
   embeddings. The full batch is not authorized merely because one prompt
   produces more labels.
4. `12D_SP500_Production_Extraction.ipynb` applies the selected v2 contract
   with v2.1 exact-quotation and intermediary-directness safeguards. It begins
   with a guarded 25-document run and checkpoints bounded repair attempts.
5. `12D2_Quarantined_Response_Resolution.ipynb` resolves the six quarantined
   records using a frozen, non-destructive audit contract and no API calls.
6. `12E_Quarterly_Exposure_Aggregation.ipynb` converts document evidence into
   deterministic point-in-time firm scores and positive/negative seed panels.
7. `12F_Embedding_Propagation_and_Placebos.ipynb` freezes RS-L-Min 10D
   proximity, equal-cardinality discovery portfolios, and 100 matched placebo
   portfolios without opening returns.
8. `12G_Locked_Forward_Performance_Evaluation.ipynb` opens the frozen
   three-month outcomes, compares direct-LLM and embedding-only portfolios,
   and conducts matched-placebo inference.
9. `12H_Long_Short_and_Hybrid_Evaluation.ipynb` completes the pre-specified
   direct, embedding, and hybrid long-short tests and the hybrid long-only test.
10. `12I_Text_Embedding_Consensus_Portfolios.ipynb` tests strict sign agreement
    using leave-one-seed-out embedding corroboration and matched text comparators.
11. `13A_Scenario_Specific_Exposure_Validation.ipynb` decomposes the compound
    theme into five frozen shocks, creates continuous text/embedding/hybrid
    scores, and prepares an outcome-blind independent-validation sample.
12. `13B_Blinded_Scenario_Adjudication.ipynb` records blinded scenario-level
    evidence review. This branch is retained as a validation artifact and is
    not used to redefine the main portfolio results.
13. `14A_Critical_Minerals_Factor_Construction.ipynb` constructs an external,
    investable critical-minerals equity-theme factor plus frozen robustness
    definitions, without opening firm or portfolio outcomes.
14. `14B_Firm_Thematic_Beta_Estimation.ipynb` estimates historical and forward
    firm sensitivities to those factors from daily CRSP returns.
15. `14C_Exposure_Scores_vs_Thematic_Beta.ipynb` tests whether frozen text and
    embedding scores predict forward thematic beta beyond historical beta and
    conventional characteristics using nested out-of-sample evaluation.
16. `14D_Factor_Controlled_Portfolio_Attribution.ipynb` attributes the frozen
    three-month portfolio returns using daily Fama–French, momentum, and
    critical-minerals factors. It reports a 248-day calendar-time test and
    separate vintage stability diagnostics without reselecting portfolios.
17. `14E_Uncapped_Seeded_Embedding_Portfolios.ipynb` implements the thematic
    portfolio-generation application: signed text relationships define the
    seeds, every embedding discovery above the frozen random-pair threshold is
    admitted without a cardinality cap, and long-only and long–short portfolios
    are compared with SPY and the critical-minerals ETF basket.
18. `14F_Capped_Portfolio_Risk_and_Theme_Analysis.ipynb` applies comparable
    Sharpe, information-ratio, volatility-matching, and controlled thematic-beta
    diagnostics to the previously frozen capped portfolios.
19. `14G_2024Q1_Portfolio_Persistence_Through_2026.ipynb` holds the frozen
    2024Q1 portfolios through August 2026. The final monthly specification uses
    CRSP through December 2025 and adjusted-price returns in 2026, explicitly
    excludes BK only when its 2026 return becomes unavailable, and reports
    horizon-specific performance, benchmark-relative statistics, thematic-factor
    persistence, coverage, and rolling risk.

The frozen design is in `config/sp500_experiment_design_v1.json`.

Prompt configuration is versioned in `config/`. Document-level extraction and
quarterly aggregation are separate stages: a later 10-Q updates the 10-K
baseline, and silence does not automatically erase prior evidence.

## Main completed result

The frozen 2024Q1 portfolios are evaluated from June 2024 through August 2026.
The equal-weight critical-minerals ETF benchmark (`COPX`, `LIT`, and `REMX`)
earned 84.84% with 32.44% annualized volatility. The capped hybrid long-only
portfolio earned 69.84% with 19.46% volatility and a 1.10 Sharpe ratio; the
capped embedding-only and uncapped embedding long-only portfolios earned
54.49% and 52.86% with approximately 14% volatility. The corresponding
critical-minerals betas remained positive and statistically significant after
controlling for the market.

The conclusion is deliberately limited: LLM-derived seeds plus holdings
embeddings produced diversified portfolios that preserved meaningful thematic
exposure and reduced risk relative to the concentrated minerals benchmark.
The single main formation vintage does not establish persistent abnormal return.

## Repository boundary

Committed files include executed notebooks, notebook builders, reusable source,
versioned prompts and schemas, design contracts, compact manifests, and rendered
figures. Raw SEC filings, retrieved passages, WRDS extracts, API response
envelopes, blinded review packets, and generated row-level tables are excluded.
They remain in the same local paths so the notebooks can reuse their checkpoints.
