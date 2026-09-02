# Economic Shock Sensitivity

This extension asks whether asset embeddings inferred from institutional holdings encode firms' exposure to recurring, economically interpretable shocks.

The first application studies monetary-policy shocks. Historical high-frequency FOMC surprises will be used to estimate a continuous firm-level rate-sensitivity target. Holdings embeddings dated before each event will then be evaluated on both cross-sectional target recovery and genuinely later FOMC return reactions.

## Notebooks

1. `00_Rate_Shock_Research_Design_and_Data_Audit.ipynb` freezes the design, downloads and inventories the official San Francisco Fed surprise data, audits all reusable local inputs, fixes the timing and sample splits, and estimates the size of the only new WRDS extract.
2. `01_FOMC_Event_Return_Acquisition.ipynb` freezes the statement-window shock, downloads exact CRSP event dates with restartable yearly checkpoints, and constructs lag-value-weighted PERMCO event returns.
3. `02_Rolling_Rate_Sensitivity_Targets.ipynb` estimates quarterly firm rate sensitivities from the preceding five years of FOMC events, applies transparent reliability shrinkage, and audits target coverage and stability.
4. `03_Future_Rate_Sensitivity_Validation.ipynb` tests whether the historical targets predict genuinely later FOMC reactions using event fixed effects, clustered inference, portfolio spreads, and shuffled placebos.
5. `04_Ex_Ante_Embedding_Rate_Sensitivity.ipynb` tests whether pre-event holdings embeddings recover rate sensitivity and predict future FOMC reactions under economic-holdings and public-availability timing conventions.
6. `04B_Incremental_Embedding_Robustness.ipynb` applies paired quarterly inference and shuffled-embedding placebos to the incremental contribution beyond characteristics.
7. `05_Behavioral_Embedding_Rate_Sensitivity.ipynb` compares which investor-behavior subsamples encode rate sensitivity most effectively.
8. `06_Post_Shock_Holdings_Adjustment.ipynb` tests return-adjusted fund rebalancing after surprises, while treating quarter-end holdings as outcomes observable in the following quarter.
9. `Economic_Shock_Sensitivity_Summary.ipynb` is a self-contained, presentation-ready narrative with frozen figures, detailed results, limitations, and notebook-level provenance.

Generated data, manifests, and results are local checkpoints and are excluded from Git.
