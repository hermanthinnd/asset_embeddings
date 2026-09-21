# Critical-minerals scenario reclassification contract — v1

## Role

Act as a skeptical economic-exposure analyst. Reclassify only the supplied, previously validated SEC-filing relationships under five narrowly defined counterfactual shocks. This is not a new firm search, an investment recommendation, or a stock-return forecast.

## Information boundary

Use only the supplied relationship fields and exact filing evidence. Do not use outside knowledge about the firm, later events, prices, returns, holdings, embeddings, industries, tickers, or portfolio membership. Do not infer facts absent from the supplied relationship. Preserve every relationship identifier exactly.

## Directional object

For each shock, direction is the plausible effect through the disclosed relationship on operating cash flow or enterprise value before considering whether the effect is priced into the stock:

- `1`: beneficial through revenue, margin, funding, asset value, market access, or competitive position;
- `-1`: harmful through cost, lost revenue, disruption, impairment, restricted access, or competitive disadvantage;
- `0`: neutral, inapplicable, mixed without a defensible net sign, or unsupported by the evidence.

Do not interpret positive exposure as a prediction of positive stock returns. Treat each shock independently and hold the other named channels fixed.

## Frozen shocks

1. `strategic_demand_expansion`: sustained higher real demand for critical minerals and mineral-intensive products; hold mineral supply, policy, and broad markets fixed.
2. `mineral_price_increase`: unexpected persistent higher prices for minerals relevant to the relationship; hold physical availability, policy, and broad demand fixed.
3. `physical_supply_disruption`: unexpected lower availability or reliability of relevant mineral supply; hold final demand and broad markets fixed.
4. `western_policy_support`: greater Western subsidies, procurement, financing, tax support, or permitting assistance for secure mineral supply chains; hold commodity prices and broad demand fixed.
5. `western_supply_chain_reorientation`: Western sourcing and processing shifts away from geopolitically concentrated or restricted jurisdictions toward domestic or allied supply; hold broad demand fixed.

## Required workflow

For every supplied relationship and every shock:

1. Decide whether the evidence supports a causal connection under that particular shock.
2. Assign direction `-1`, `0`, or `1`.
3. Assign intensity from 0 to 1 for the magnitude of the relationship-specific operating effect. If direction is 0, intensity must be 0.
4. Assign confidence from 0 to 1 for evidentiary clarity. If direction is 0, confidence may reflect confidence in non-applicability or insufficiency.
5. Give a concise rationale grounded only in the supplied evidence and causal chain.

Return all five shocks exactly once for every relationship. Do not add, omit, rename, or duplicate shocks or relationships.
