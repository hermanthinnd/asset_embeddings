"""Point-in-time document normalization helpers.

This module is deliberately independent of any LLM provider. SEC filings and
earnings-call transcripts must pass the same contract before retrieval or
exposure extraction begins.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_FIELDS = {
    "document_id",
    "firm_id",
    "document_type",
    "document_date",
    "available_date",
    "source",
    "text",
}

ALLOWED_DOCUMENT_TYPES = {
    "10-K",
    "10-Q",
    "8-K",
    "earnings_call",
    "company_description",
    "other",
}


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO date: {value!r}") from exc


def normalize_document(record: Dict[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Missing document fields: {sorted(missing)}")

    normalized = dict(record)
    normalized["document_id"] = str(normalized["document_id"]).strip()
    normalized["firm_id"] = str(normalized["firm_id"]).strip()
    normalized["document_type"] = str(normalized["document_type"]).strip()
    normalized["source"] = str(normalized["source"]).strip()
    normalized["text"] = str(normalized["text"])

    if normalized["document_type"] not in ALLOWED_DOCUMENT_TYPES:
        raise ValueError(f"Unsupported document_type: {normalized['document_type']}")
    if not normalized["document_id"] or not normalized["firm_id"]:
        raise ValueError("document_id and firm_id cannot be empty")
    if not normalized["text"].strip():
        raise ValueError("text cannot be empty")

    document_date = _parse_iso_date(normalized["document_date"], "document_date")
    available_date = _parse_iso_date(normalized["available_date"], "available_date")
    if available_date < document_date:
        raise ValueError("available_date cannot precede document_date")

    supplied_hash = normalized.get("content_sha256")
    computed_hash = content_sha256(normalized["text"])
    if supplied_hash and supplied_hash != computed_hash:
        raise ValueError("content_sha256 does not match text")
    normalized["content_sha256"] = computed_hash
    normalized.setdefault("section", None)
    normalized.setdefault("source_uri", None)
    return normalized


def validate_documents(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [normalize_document(record) for record in records]
    ids = [record["document_id"] for record in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("document_id values must be unique")
    return normalized


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return validate_documents(records)
