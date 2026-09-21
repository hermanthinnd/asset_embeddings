# Production safeguards — v2.1 addendum

These rules override any conflicting interpretation of the v2 contract.

1. **Contiguous evidence:** `evidence_quote` must be copied from exactly one supplied passage, character for character, in its original order, with no skipped words or sentences. It must occur as one exact substring of the passage named by `evidence_passage_id` and contain at most 60 whitespace-delimited words. If no qualifying quotation exists, omit the exposure rather than reconstructing a quotation.
2. **Intermediary directness:** when the stated causal route operates through the firm's supplier, customer, distributor, contract manufacturer, or another intermediary, it cannot be `direct`. Use `first_order_indirect` for one visible intermediary and `second_order_indirect` for two. A firm purchasing or consuming a mineral itself may remain `direct`; mere mention of a supplier elsewhere does not change that.
3. **No duplicated mechanisms:** do not split one economic relationship into both input-cost and geographic-supply records unless different passages establish economically distinct effects. Explain any retained separation in `netting_note`.
4. **Self-check before output:** verify identifier preservation, evidence-array consistency, exact quotation containment, confidence-band consistency, and intermediary directness. If a check fails, correct the object before returning it.
