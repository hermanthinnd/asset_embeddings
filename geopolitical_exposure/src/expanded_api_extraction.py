import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from getpass import getpass

import pandas as pd


INSTRUCTIONS = """You are a conservative research-data extractor. Use only the supplied
pre-event document passages. Do not use outside knowledge or infer exposure from the company
name. Return an exposure only when the quoted text directly supports the specified channel and
direction. A material disclosed business relationship to a channel—such as revenue, customers,
operations, assets, production, or facilities—counts as exposure even if the filing predates and
does not name the future event. Evidence must be an exact, shortest self-contained quote of no
more than 1000 characters from the supplied passage. If nothing is supported, set
no_supported_exposure=true and return an empty exposures array. A company name, patent
jurisdiction, generic compliance language, or generic industry membership without a disclosed
business relationship is not by itself evidence of scenario exposure. evidence_start and
evidence_end must be offsets in the original document, computed from the supplied passage start
plus the quote's location in that passage."""


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeze_contract(geo_root, v2_results, manifests, event, selection_hash, queue_hash,
                    api_queue, model="gpt-5.6-luna", batch_size=25):
    schema = json.loads((Path(geo_root) / "schemas/exposure_output_schema.json").read_text())
    prompt_hash = hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    response_dir = Path(v2_results) / "openai_responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    queued_ids = set(api_queue.document_id.astype(str))
    completed_ids = {path.stem for path in response_dir.glob("*.json")} & queued_ids
    remaining = api_queue[~api_queue.document_id.astype(str).isin(completed_ids)].sort_values("document_id")
    contract = {
        "event_id": event["event_id"], "information_cutoff": event["information_cutoff"],
        "selection_sha256": selection_hash, "queue_sha256": queue_hash,
        "prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "model": model,
        "store": False, "batch_size": batch_size, "event_outcomes_loaded": False,
    }
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    (Path(manifests) / "text_expansion_v2_extraction_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    status = pd.DataFrame([{
        "queued_documents": len(api_queue), "checkpointed_documents": len(completed_ids),
        "remaining_documents": len(remaining), "next_batch_documents": min(batch_size, len(remaining)),
        "approx_remaining_input_tokens": int(remaining.approx_input_tokens.sum()),
        "api_key_ready": bool(os.getenv("OPENAI_API_KEY")), "frozen_model": model,
        "contract_sha256": contract["sha256"],
    }])
    return schema, contract, response_dir, remaining, status


def run_batch(remaining, retrieval_records, event, schema, contract, response_dir):
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not os.getenv("OPENAI_API_KEY"):
        api_key = getpass("OpenAI API key (input concealed): ").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        os.environ["OPENAI_API_KEY"] = api_key
    # Prevent one slow network request from holding the notebook indefinitely.
    # The SDK retries transient failures; completed responses remain checkpointed.
    client = OpenAI(timeout=120.0, max_retries=2)
    lookup = {str(record["document_id"]): record for record in retrieval_records}
    failures = []
    batch = remaining.head(contract["batch_size"])
    progress = tqdm(
        batch.itertuples(index=False), total=len(batch), desc="OpenAI extraction",
        unit="document", dynamic_ncols=True,
    )
    for position, row in enumerate(progress, 1):
        document_id = str(row.document_id)
        progress.set_postfix_str(document_id)
        record = lookup[document_id]
        output_path = Path(response_dir) / f"{document_id}.json"
        payload = {
            "firm_id": record["firm_id"], "document_id": document_id,
            "scenario_id": event["event_id"], "channel_definitions": event["channels"],
            "passages": record["passages"],
        }
        try:
            response = client.responses.create(
                model=contract["model"], instructions=INSTRUCTIONS, input=json.dumps(payload),
                store=False, reasoning={"effort": "low"},
                text={"format": {"type": "json_schema", "name": "scenario_exposure",
                                  "strict": True, "schema": schema}},
            )
            audit = {
                "result": json.loads(response.output_text), "response_id": response.id,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "prompt_sha256": contract["prompt_sha256"],
                "schema_sha256": contract["schema_sha256"],
                "queue_sha256": contract["queue_sha256"],
                "source_content_sha256": record["content_sha256"], "store": False,
            }
            output_path.write_text(json.dumps(audit, indent=2))
            progress.set_postfix(saved=position, failures=len(failures), refresh=True)
        except Exception as error:
            failures.append({"document_id": document_id, "firm_id": record["firm_id"],
                             "error_type": type(error).__name__, "error": str(error)})
            tqdm.write(f"ERROR {document_id}: {type(error).__name__}: {error}")
            progress.set_postfix(saved=position - len(failures), failures=len(failures), refresh=True)
    return pd.DataFrame(failures, columns=["document_id", "firm_id", "error_type", "error"])


def validate_responses(api_queue, normalized_documents, retrieval_records, event, contract,
                       response_dir, v2_results):
    document_lookup = {str(record["document_id"]): record for record in normalized_documents}
    retrieval_lookup = {str(record["document_id"]): record for record in retrieval_records}
    channels = {item["channel"] for item in event["channels"]}
    response_rows, exposure_rows = [], []
    for path in sorted(Path(response_dir).glob("*.json")):
        audit = json.loads(path.read_text())
        result = audit.get("result", {})
        document_id = str(result.get("document_id", path.stem))
        retrieval = retrieval_lookup.get(document_id)
        document = document_lookup.get(document_id)
        contract_valid = (
            audit.get("prompt_sha256") == contract["prompt_sha256"]
            and audit.get("schema_sha256") == contract["schema_sha256"]
            and audit.get("queue_sha256") == contract["queue_sha256"]
            and audit.get("model") == contract["model"] and audit.get("store") is False
        )
        identity_valid = bool(retrieval) and path.stem == document_id and str(result.get("firm_id")) == str(retrieval["firm_id"])
        source_hash_valid = bool(retrieval) and audit.get("source_content_sha256") == retrieval["content_sha256"]
        exposures = result.get("exposures", [])
        no_exposure = bool(result.get("no_supported_exposure", False))
        logical_empty_valid = no_exposure == (len(exposures) == 0)
        response_rows.append({
            "document_id": document_id, "firm_id": result.get("firm_id"),
            "no_supported_exposure": no_exposure, "exposure_count": len(exposures),
            "contract_valid": contract_valid, "identity_valid": identity_valid,
            "source_hash_valid": source_hash_valid, "logical_empty_valid": logical_empty_valid,
            "response_file": str(path),
        })
        for number, item in enumerate(exposures):
            evidence = str(item.get("evidence", ""))
            full_text = document["text"] if document else ""
            occurrences = full_text.count(evidence) if evidence else 0
            score = pd.to_numeric(item.get("score"), errors="coerce")
            confidence = pd.to_numeric(item.get("confidence"), errors="coerce")
            exposure_rows.append({
                "document_id": document_id, "firm_id": result.get("firm_id"),
                "exposure_number": number, "channel": item.get("channel"),
                "direction": item.get("direction"), "score": score, "confidence": confidence,
                "evidence": evidence, "evidence_occurrences": occurrences,
                "reasoning_summary": item.get("reasoning_summary"),
                "channel_valid": item.get("channel") in channels,
                "direction_valid": item.get("direction") in {"positive", "negative", "ambiguous"},
                "score_valid": pd.notna(score) and -1 <= float(score) <= 1,
                "confidence_valid": pd.notna(confidence) and 0 <= float(confidence) <= 1,
                "quote_valid": 0 < len(evidence) <= 1000 and occurrences >= 1,
                "response_contract_valid": contract_valid and identity_valid and source_hash_valid and logical_empty_valid,
            })
    response_audit = pd.DataFrame(response_rows)
    columns = ["document_id", "firm_id", "exposure_number", "channel", "direction", "score",
               "confidence", "evidence", "evidence_occurrences", "reasoning_summary", "channel_valid",
               "direction_valid", "score_valid", "confidence_valid", "quote_valid", "response_contract_valid"]
    candidates = pd.DataFrame(exposure_rows, columns=columns)
    response_audit.to_parquet(Path(v2_results) / "expanded_response_audit.parquet", index=False)
    candidates.to_parquet(Path(v2_results) / "expanded_exposure_candidates.parquet", index=False)
    response_failures = int((~response_audit[["contract_valid", "identity_valid", "source_hash_valid", "logical_empty_valid"]].all(axis=1)).sum()) if len(response_audit) else 0
    exposure_failures = int((~candidates[["channel_valid", "direction_valid", "score_valid", "confidence_valid", "quote_valid", "response_contract_valid"]].all(axis=1)).sum()) if len(candidates) else 0
    completed = response_audit.document_id.nunique() if len(response_audit) else 0
    status = pd.DataFrame([{
        "queued_documents": len(api_queue), "validated_responses": completed,
        "remaining_documents": len(api_queue) - completed, "candidate_exposures": len(candidates),
        "response_gate_failures": response_failures, "exposure_gate_failures": exposure_failures,
        "complete_and_valid": completed == len(api_queue) and response_failures == 0 and exposure_failures == 0,
    }])
    return response_audit, candidates, status


def build_evidence_repair_review(candidates, retrieval_records, v2_results):
    """Propose exact source quotations for non-exact model evidence; never auto-accept them."""
    retrieval_lookup = {str(record["document_id"]): record for record in retrieval_records}

    def normalized(value):
        return re.sub(r"\s+", " ", str(value)).strip().lower()

    def source_candidates(record, target):
        options = []
        target_tokens = re.findall(r"[a-z0-9]+", target.lower())
        target_length = len(target_tokens)
        for passage_number, passage in enumerate(record.get("passages", [])):
            text = passage.get("text", "")
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
            for width in (1, 2, 3):
                for start in range(max(0, len(sentences) - width + 1)):
                    quote = " ".join(sentences[start:start + width]).strip()
                    if 20 <= len(quote) <= 1000:
                        options.append((passage_number, quote))
            # Filing text often contains abbreviations such as "U.S." and table formatting
            # that defeat sentence splitting. Align normalized word sequences and propose
            # exact contiguous source spans around the best alignment as a second route.
            token_matches = list(re.finditer(r"\S+", text))
            source_tokens = [re.sub(r"[^a-z0-9]+", "", match.group().lower()) for match in token_matches]
            if target_length and source_tokens:
                match = SequenceMatcher(None, target_tokens, source_tokens, autojunk=False).find_longest_match()
                estimated_start = max(0, min(len(source_tokens) - 1, match.b - match.a))
                for start_shift in range(-8, 9):
                    start = max(0, estimated_start + start_shift)
                    for length_shift in (-12, -6, 0, 6, 12, 18):
                        width = max(4, target_length + length_shift)
                        end = min(len(token_matches), start + width)
                        if end <= start:
                            continue
                        quote = text[token_matches[start].start():token_matches[end - 1].end()].strip()
                        if 20 <= len(quote) <= 1000:
                            options.append((passage_number, quote))
        return options

    rows = []
    for row in candidates.itertuples(index=False):
        base = row._asdict()
        if bool(row.quote_valid):
            base.update({
                "evidence_status": "exact_model_quote", "proposed_exact_evidence": row.evidence,
                "repair_similarity": 1.0, "repair_passage_number": pd.NA,
                "review_decision": "accept_exact", "review_notes": "",
            })
            rows.append(base)
            continue
        record = retrieval_lookup[str(row.document_id)]
        target = normalized(row.evidence)
        ranked = []
        for passage_number, quote in source_candidates(record, row.evidence):
            score = SequenceMatcher(None, target, normalized(quote), autojunk=False).ratio()
            ranked.append((score, passage_number, quote))
        score, passage_number, quote = max(ranked, default=(0.0, pd.NA, ""), key=lambda item: item[0])
        exact_occurrences = sum(p.get("text", "").count(quote) for p in record.get("passages", [])) if quote else 0
        base.update({
            "evidence_status": "manual_review_required", "proposed_exact_evidence": quote,
            "repair_similarity": score, "repair_passage_number": passage_number,
            "proposed_quote_occurrences_in_retrieval": exact_occurrences,
            "review_decision": "pending", "review_notes": "",
        })
        rows.append(base)
    review = pd.DataFrame(rows)
    review.to_parquet(Path(v2_results) / "expanded_evidence_repair_review.parquet", index=False)
    pending = review[review.review_decision.eq("pending")].copy()
    pending.to_csv(Path(v2_results) / "expanded_evidence_manual_review.csv", index=False)
    return review, pending


def finalize_evidence_adjudication(review, retrieval_records, override_specs,
                                   rejected_document_ids, v2_results):
    """Apply explicit review decisions and fail unless every accepted quote is exact."""
    retrieval_lookup = {str(record["document_id"]): record for record in retrieval_records}

    def exact_span(document_id, start_marker, end_marker):
        matches = []
        for passage in retrieval_lookup[document_id].get("passages", []):
            text = passage.get("text", "")
            start = text.find(start_marker)
            if start < 0:
                continue
            end = text.find(end_marker, start)
            if end >= 0:
                matches.append(text[start:end + len(end_marker)].strip())
        if len(matches) != 1:
            raise ValueError(f"Expected one exact override span for {document_id}; found {len(matches)}.")
        return matches[0]

    adjudicated = review.copy()
    final_evidence, decisions, notes = [], [], []
    rejected = {str(value) for value in rejected_document_ids}
    for row in adjudicated.itertuples(index=False):
        document_id = str(row.document_id)
        if row.review_decision == "accept_exact":
            final_evidence.append(row.evidence)
            decisions.append("accept_exact_model_quote")
            notes.append("Model evidence was already an exact source quotation.")
        elif document_id in rejected:
            final_evidence.append("")
            decisions.append("reject_insufficient_specificity")
            notes.append("Generic risk language did not establish a sufficiently concrete event-channel exposure.")
        elif document_id in override_specs:
            start_marker, end_marker = override_specs[document_id]
            final_evidence.append(exact_span(document_id, start_marker, end_marker))
            decisions.append("accept_manual_exact_span")
            notes.append("Classification supported; quotation boundary replaced with an explicit exact source span.")
        else:
            final_evidence.append(row.proposed_exact_evidence)
            decisions.append("accept_algorithmic_exact_repair")
            notes.append("Classification supported and proposed repair is an exact retrieved-passage quotation.")
    adjudicated["final_evidence"] = final_evidence
    adjudicated["final_decision"] = decisions
    adjudicated["adjudication_notes"] = notes
    adjudicated["accepted"] = ~adjudicated.final_decision.str.startswith("reject_")

    def occurrences(row):
        if not row.accepted:
            return 0
        return sum(
            passage.get("text", "").count(row.final_evidence)
            for passage in retrieval_lookup[str(row.document_id)].get("passages", [])
        )

    adjudicated["final_quote_occurrences"] = adjudicated.apply(occurrences, axis=1)
    adjudicated["final_quote_valid"] = (
        (~adjudicated.accepted)
        | ((adjudicated.final_evidence.str.len().between(1, 1000))
           & adjudicated.final_quote_occurrences.gt(0))
    )
    adjudicated["signed_score"] = adjudicated.score.abs()
    adjudicated.loc[adjudicated.direction.eq("negative"), "signed_score"] *= -1
    adjudicated.loc[adjudicated.direction.eq("ambiguous"), "signed_score"] = pd.NA
    if not adjudicated.final_quote_valid.all():
        failed = adjudicated.loc[~adjudicated.final_quote_valid, "document_id"].tolist()
        raise ValueError(f"Final evidence validation failed for: {failed}")
    final_panel = adjudicated[adjudicated.accepted].copy()
    adjudicated.to_parquet(Path(v2_results) / "expanded_evidence_adjudication_audit.parquet", index=False)
    final_panel.to_parquet(Path(v2_results) / "expanded_exposure_panel_final.parquet", index=False)
    return adjudicated, final_panel
