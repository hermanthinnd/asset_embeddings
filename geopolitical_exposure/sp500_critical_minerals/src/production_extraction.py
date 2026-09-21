import hashlib
import json
import re
import time
from pathlib import Path


INTERMEDIARY = re.compile(r"\b(supplier|suppliers|customer|customers|distributor|contract manufacturer|channel partner)s?\b", re.I)
DIRECT_ACTIVITY = re.compile(r"\b(we|our|company|firm)\b.{0,100}\b(extract|mine|process|refin|recycl|purchase|purchases|procure|consume|uses?|own|operat)", re.I | re.S)


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_payload(record):
    return {
        "firm_id": str(record["firm_id"]),
        "document_id": str(record["document_id"]),
        "document_type": record["document_type"],
        "available_date": str(record["available_date"]),
        "theme": {
            "name": "critical_minerals_secure_supply",
            "shock": "Sustained increase in strategic demand and policy support with Western supply-chain security and reorientation",
        },
        "passages": record["passages"],
    }


def validate_result(result, record):
    issues = []
    expected = {k: str(record[k]) for k in ["firm_id", "document_id", "document_type", "available_date"]}
    for key, value in expected.items():
        if str(result.get(key)) != value:
            issues.append(f"identifier_mismatch:{key}")
    exposures = result.get("exposures", [])
    if bool(result.get("has_exposure_evidence")) != bool(exposures):
        issues.append("evidence_array_inconsistency")
    passage_map = {p["passage_id"]: p["text"] for p in record["passages"]}
    for i, exposure in enumerate(exposures):
        prefix = f"exposure_{i+1}"
        quote = exposure.get("evidence_quote", "")
        pid = exposure.get("evidence_passage_id", "")
        if not quote or quote not in passage_map.get(pid, ""):
            issues.append(f"{prefix}:noncontiguous_or_wrong_passage_quote")
        if len(quote.split()) > 60:
            issues.append(f"{prefix}:quote_over_60_words")
        confidence = float(exposure.get("confidence", -1))
        expected_band = "low" if confidence < .60 else "medium" if confidence < .85 else "high"
        if exposure.get("confidence_band") != expected_band:
            issues.append(f"{prefix}:confidence_band_mismatch")
        combined = " ".join([quote, exposure.get("causal_chain", "")])
        if exposure.get("directness") == "direct" and INTERMEDIARY.search(combined) and not DIRECT_ACTIVITY.search(combined):
            issues.append(f"{prefix}:possible_intermediary_marked_direct")
    return sorted(set(issues))


def call_and_validate(client, model, prompt, schema, record, max_attempts=3):
    payload = build_payload(record)
    repair = ""
    attempts = []
    for attempt in range(1, max_attempts + 1):
        response = client.responses.create(
            model=model,
            instructions=prompt + repair,
            input=json.dumps(payload),
            store=False,
            reasoning={"effort": "low"},
            text={"format": {"type": "json_schema", "name": "critical_minerals_v2_1", "strict": True, "schema": schema}},
        )
        result = json.loads(response.output_text)
        issues = validate_result(result, record)
        attempts.append({"attempt": attempt, "response_id": response.id, "model": response.model, "usage": response.usage.model_dump() if response.usage else None, "issues": issues})
        if not issues:
            return result, attempts, True
        repair = "\n\nYour prior object failed deterministic validation for: " + "; ".join(issues) + ". Re-read the supplied passages and return a corrected complete object. Do not preserve an exposure that cannot pass the rules."
        time.sleep(min(2 ** attempt, 8))
    return result, attempts, False


def save_envelope(path, result, attempts, valid, record, prompt_hash, schema_hash):
    envelope = {
        "result": result,
        "attempts": attempts,
        "valid": valid,
        "source_content_sha256": record["content_sha256"],
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "store": False,
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(envelope, indent=2))
    tmp.replace(path)
