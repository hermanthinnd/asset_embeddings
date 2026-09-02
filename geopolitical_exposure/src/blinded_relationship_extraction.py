from __future__ import annotations

import hashlib
import json
import os
from getpass import getpass
from pathlib import Path

import pandas as pd


MODEL = "gpt-5.6-luna"
INSTRUCTIONS = """You are a conservative, blinded research-data adjudicator. Use only the
supplied pre-event SEC filing passages. Do not use outside knowledge, post-event information,
company-name associations, stock returns, or assumptions based only on industry. The candidate
source is intentionally hidden. Determine whether the passages directly support an economically
meaningful relationship to the specified channel and reference seed businesses. A valid indirect
relationship may be a supplier, customer, competitor, substitute, complement, commodity producer
or consumer, geographic operation, financial exposure, or another explicitly disclosed mechanism.
High-level similarity, generic risk boilerplate, and common investor clientele are not sufficient
unless the filing directly documents a channel-relevant relationship. Unsupported is a valid and
important result. For a supported or ambiguous relationship, evidence_quote must be the shortest
exact self-contained quotation from one supplied passage, no more than 1000 characters, and
evidence_passage_rank must identify that passage. For unsupported results, use relationship_type
'unsupported', direction 'none', channel_relevance 'none', an empty quote, and null passage rank."""

SCHEMA = {
    "type": "object",
    "properties": {
        "blind_id": {"type": "string"},
        "supported_relationship": {"type": "boolean"},
        "relationship_type": {"type": "string", "enum": [
            "peer_or_competitor", "supplier", "customer", "substitute", "complement",
            "geographic_operations", "commodity_producer", "commodity_consumer",
            "financial_exposure", "common_investor_clientele_only", "other", "unsupported", "ambiguous"]},
        "relationship_direction": {"type": "string", "enum": ["positive", "negative", "mixed", "none", "ambiguous"]},
        "channel_relevance": {"type": "string", "enum": ["direct", "indirect", "none", "ambiguous"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_quote": {"type": "string", "maxLength": 1000},
        "evidence_passage_rank": {"type": ["integer", "null"]},
        "reasoning_summary": {"type": "string"},
    },
    "required": ["blind_id", "supported_relationship", "relationship_type", "relationship_direction",
                 "channel_relevance", "confidence", "evidence_quote", "evidence_passage_rank", "reasoning_summary"],
    "additionalProperties": False,
}


def load_packets(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeze_contract(public: Path, packets: list[dict], batch_size: int = 10) -> tuple[dict, pd.DataFrame]:
    queue_contract = json.loads((public / "retrieval_contract.json").read_text())
    prompt_hash = hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()
    schema_hash = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()
    response_dir = public.parent / "api_responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    queued = {packet["blind_id"] for packet in packets}
    completed = {path.stem for path in response_dir.glob("GEO-*.json")} & queued
    contract = {
        "retrieval_contract_sha256": queue_contract["sha256"], "model": MODEL,
        "prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "batch_size": batch_size,
        "store": False, "candidate_origin_blinded": True, "event_outcomes_loaded": False,
    }
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    (public / "relationship_extraction_contract.json").write_text(json.dumps(contract, indent=2))
    remaining = [packet for packet in packets if packet["blind_id"] not in completed]
    status = pd.DataFrame([{"queued": len(packets), "checkpointed": len(completed), "remaining": len(remaining),
                            "next_batch": min(batch_size, len(remaining)),
                            "approx_remaining_input_tokens": sum(sum(len(p["text"]) for p in packet["passages"]) for packet in remaining) // 4,
                            "api_key_ready": bool(os.getenv("OPENAI_API_KEY")), "model": MODEL,
                            "contract_sha256": contract["sha256"]}])
    return contract, status


def run_batch(public: Path, packets: list[dict], contract: dict) -> pd.DataFrame:
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not os.getenv("OPENAI_API_KEY"):
        key = getpass("OpenAI API key (input concealed): ").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required.")
        os.environ["OPENAI_API_KEY"] = key
    response_dir = public.parent / "api_responses"
    completed = {path.stem for path in response_dir.glob("GEO-*.json")}
    remaining = [packet for packet in packets if packet["blind_id"] not in completed][:contract["batch_size"]]
    client = OpenAI(timeout=120.0, max_retries=2)
    failures = []
    for packet in tqdm(remaining, desc="Blinded relationship extraction", unit="candidate", dynamic_ncols=True):
        payload = {"blind_id": packet["blind_id"], "firm_id": packet["firm_id"],
                   "firm_name": packet.get("firm_name"), "channel": packet["channel"],
                   "reference_seed_names": packet["reference_seed_names"], "information_cutoff": "2022-02-23",
                   "passages": packet["passages"]}
        try:
            response = client.responses.create(
                model=contract["model"], instructions=INSTRUCTIONS, input=json.dumps(payload),
                store=False, reasoning={"effort": "low"},
                text={"format": {"type": "json_schema", "name": "blinded_relationship",
                                   "strict": True, "schema": SCHEMA}},
            )
            saved = {"result": json.loads(response.output_text), "response_id": response.id,
                     "model": response.model, "usage": response.usage.model_dump() if response.usage else None,
                     "contract_sha256": contract["sha256"], "prompt_sha256": contract["prompt_sha256"],
                     "schema_sha256": contract["schema_sha256"], "source_content_sha256": packet["content_sha256"],
                     "store": False}
            (response_dir / f"{packet['blind_id']}.json").write_text(json.dumps(saved, indent=2))
        except Exception as error:
            failures.append({"blind_id": packet["blind_id"], "error_type": type(error).__name__, "error": str(error)})
    return pd.DataFrame(failures, columns=["blind_id", "error_type", "error"])


def validate_responses(public: Path, packets: list[dict], contract: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {packet["blind_id"]: packet for packet in packets}
    rows = []
    for path in sorted((public.parent / "api_responses").glob("GEO-*.json")):
        saved = json.loads(path.read_text())
        result = saved.get("result", {})
        blind_id = result.get("blind_id", path.stem)
        packet = lookup.get(blind_id)
        passages = packet.get("passages", []) if packet else []
        quote = str(result.get("evidence_quote", ""))
        rank = result.get("evidence_passage_rank")
        rank_valid = rank is None or (isinstance(rank, int) and 1 <= rank <= len(passages))
        exact_occurrences = sum(item["text"].count(quote) for item in passages) if quote else 0
        supported = bool(result.get("supported_relationship", False))
        logical_valid = ((supported and bool(quote) and rank is not None) or
                         (not supported and not quote and rank is None and result.get("relationship_type") == "unsupported"))
        # Batch size is an execution-control parameter, not part of the scientific
        # extraction specification.  Keep exact contract equality for the audit,
        # while validating responses against the invariant model/prompt/schema.
        batch_contract_exact = saved.get("contract_sha256") == contract["sha256"]
        protocol_valid = (saved.get("model") == contract["model"] and
                          saved.get("prompt_sha256") == contract["prompt_sha256"] and
                          saved.get("schema_sha256") == contract["schema_sha256"] and
                          saved.get("store") is False)
        rows.append({**result, "identity_valid": path.stem == blind_id and packet is not None,
                     "batch_contract_exact": batch_contract_exact,
                     "protocol_valid": protocol_valid,
                     "source_hash_valid": bool(packet) and saved.get("source_content_sha256") == packet.get("content_sha256"),
                     "passage_rank_valid": rank_valid, "exact_quote_occurrences": exact_occurrences,
                     "exact_quote_valid": (exact_occurrences >= 1) if supported else not quote,
                     "logical_valid": logical_valid})
    audit = pd.DataFrame(rows)
    gates = ["identity_valid", "protocol_valid", "source_hash_valid", "passage_rank_valid", "exact_quote_valid", "logical_valid"]
    if len(audit):
        audit["machine_valid"] = audit[gates].all(axis=1)
        review = audit.copy()
        review["human_decision"] = "pending" 
        review["human_notes"] = ""
    else:
        review = audit.copy()
    audit.to_parquet(public / "relationship_response_audit.parquet", index=False)
    review.to_csv(public / "relationship_human_review.csv", index=False)
    return audit, review
