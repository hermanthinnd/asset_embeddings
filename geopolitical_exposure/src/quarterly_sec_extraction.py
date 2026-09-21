"""Retrieval and checkpointed structured extraction for the quarterly SEC pilot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


SEARCH_TERMS = [
    "critical mineral", "critical minerals", "rare earth", "rare-earth",
    "lithium", "cobalt", "nickel", "graphite", "manganese", "neodymium",
    "praseodymium", "dysprosium", "terbium", "gallium", "germanium", "indium",
    "tellurium", "antimony", "tungsten", "tantalum", "niobium", "vanadium",
    "platinum group", "uranium", "battery material", "battery materials",
    "cathode", "anode", "permanent magnet", "mineral processing",
    "mineral extraction", "mineral refining", "mineral recycling",
    "mine development", "mining operation", "mining operations",
    "smelting", "ore body", "ore bodies", "separation facility",
    "Defense Production Act", "Inflation Reduction Act", "domestic content",
    "supply chain independence", "supply chain localization", "export control",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_jsonl(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def word_count(value: str) -> int:
    return len(re.findall(r"\b\w+\b", str(value)))


def retrieve_passages(text: str, radius: int = 850, max_passages: int = 18) -> List[dict]:
    """Retrieve merged, scored windows around frozen theme terms."""
    lower = text.lower()
    raw = []
    for term in SEARCH_TERMS:
        for match in re.finditer(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lower):
            raw.append((max(0, match.start() - radius), min(len(text), match.end() + radius), term))
    if not raw:
        return []
    raw.sort()
    merged = []
    for start, end, term in raw:
        if merged and start <= merged[-1][1] + 150:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].add(term)
        else:
            merged.append([start, end, {term}])
    ranked = []
    for start, end, terms in merged:
        passage = text[start:end]
        specificity = len(terms)
        materiality_hits = len(re.findall(
            r"\b(revenue|sales|cost|supply|supplier|customer|facility|mine|project|production|"
            r"manufactur|processing|refin|recycl|grant|credit|loan|procurement|shortage|price)\w*\b",
            passage.lower(),
        ))
        ranked.append({
            "start": start, "end": end, "matched_terms": sorted(terms),
            "retrieval_score": specificity * 3 + min(materiality_hits, 12), "text": passage,
        })
    ranked.sort(key=lambda row: (-row["retrieval_score"], row["start"]))
    selected = sorted(ranked[:max_passages], key=lambda row: row["start"])
    for number, row in enumerate(selected, 1):
        row["passage_id"] = f"P{number:02d}"
    return selected


def build_retrieval_records(documents: Iterable[dict]) -> List[dict]:
    records = []
    for document in documents:
        passages = retrieve_passages(document["text"])
        records.append({
            "document_id": document["document_id"], "firm_id": document["firm_id"],
            "document_type": document["document_type"], "available_date": document["available_date"],
            "content_sha256": document["content_sha256"], "passages": passages,
            "retrieved_characters": sum(len(item["text"]) for item in passages),
        })
    return records


def run_batch(remaining: pd.DataFrame, retrieval_lookup: Dict[str, dict], ontology: dict,
              schema: dict, prompt: str, model: str, response_dir: Path) -> pd.DataFrame:
    import os
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required; set it without printing its value.")
    from openai import OpenAI
    client = OpenAI()
    response_dir = Path(response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for position, row in enumerate(remaining.itertuples(index=False), 1):
        document_id = str(row.document_id)
        record = retrieval_lookup[document_id]
        payload = {
            "firm_id": record["firm_id"], "document_id": document_id,
            "document_type": record["document_type"], "available_date": record["available_date"],
            "theme": ontology, "passages": record["passages"],
        }
        try:
            response = client.responses.create(
                model=model, instructions=prompt, input=json.dumps(payload, ensure_ascii=False),
                store=False, reasoning={"effort": "low"},
                text={"format": {"type": "json_schema", "name": "critical_minerals_exposure",
                                  "strict": True, "schema": schema}},
            )
            result = json.loads(response.output_text)
            audit = {
                "result": result, "response_id": response.id, "model": response.model,
                "requested_model": model,
                "usage": response.usage.model_dump() if response.usage else None,
                "source_content_sha256": record["content_sha256"],
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "schema_sha256": sha256_bytes(json.dumps(schema, sort_keys=True).encode()),
                "ontology_sha256": sha256_bytes(json.dumps(ontology, sort_keys=True).encode()),
                "store": False,
            }
            (response_dir / f"{document_id}.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception as error:
            failures.append({"document_id": document_id, "error_type": type(error).__name__, "error": str(error)})
        print(f"API progress: {position}/{len(remaining)}; document={document_id}; failures={len(failures)}")
    return pd.DataFrame(failures)


def validate_checkpointed_responses(queue: pd.DataFrame, documents: Iterable[dict], response_dir: Path,
                                    model: str, prompt_hash: str, schema_hash: str,
                                    ontology_hash: str):
    documents = {str(item["document_id"]): item for item in documents}
    queued = set(queue.document_id.astype(str))
    response_paths = sorted(Path(response_dir).glob("*.json"))
    audit_rows, exposure_rows = [], []
    for path in response_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("result", {})
        document_id = str(result.get("document_id", path.stem))
        source = documents.get(document_id)
        errors = []
        if document_id not in queued: errors.append("not_in_queue")
        if source is None: errors.append("missing_source")
        if record.get("requested_model", record.get("model")) != model: errors.append("model_mismatch")
        if record.get("prompt_sha256") != prompt_hash: errors.append("prompt_mismatch")
        if record.get("schema_sha256") != schema_hash: errors.append("schema_mismatch")
        if record.get("ontology_sha256") != ontology_hash: errors.append("ontology_mismatch")
        if source and record.get("source_content_sha256") != source["content_sha256"]: errors.append("source_hash_mismatch")
        exposures = result.get("exposures", [])
        if bool(result.get("has_exposure_evidence")) != bool(exposures): errors.append("logical_inconsistency")
        quote_failures = 0
        for number, exposure in enumerate(exposures):
            quote = str(exposure.get("evidence_quote", ""))
            occurrences = source["text"].count(quote) if source and quote else 0
            quote_valid = occurrences >= 1 and word_count(quote) <= 60
            if not quote_valid: quote_failures += 1
            exposure_rows.append({
                "document_id": document_id, "firm_id": result.get("firm_id"),
                "document_type": result.get("document_type"), "available_date": result.get("available_date"),
                "exposure_number": number + 1, **exposure,
                "quote_occurrences": occurrences, "quote_word_count": word_count(quote),
                "quote_valid": quote_valid,
            })
        if quote_failures: errors.append("invalid_evidence_quote")
        audit_rows.append({
            "document_id": document_id, "exposures": len(exposures),
            "quote_failures": quote_failures, "valid": not errors,
            "errors": ",".join(errors),
        })
    return pd.DataFrame(audit_rows), pd.DataFrame(exposure_rows)
