"""Reproducible SEC EDGAR acquisition helpers for the geopolitical pilot.

Network functions require an identifying SEC_USER_AGENT and deliberately
throttle requests. They never call an LLM and never inspect event returns.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"


def sec_headers(user_agent: str) -> Dict[str, str]:
    """Return compliant request headers after validating identification."""
    value = str(user_agent or "").strip()
    if "@" not in value or len(value) < 8:
        raise ValueError(
            "Set SEC_USER_AGENT to a descriptive value containing your email, "
            "for example 'Academic research your_email@university.edu'."
        )
    return {"User-Agent": value, "Accept-Encoding": "gzip, deflate"}


def get_json(url: str, user_agent: str, pause_seconds: float = 0.15) -> dict:
    response = requests.get(url, headers=sec_headers(user_agent), timeout=60)
    response.raise_for_status()
    time.sleep(max(0.0, pause_seconds))
    return response.json()


def get_text(url: str, user_agent: str, pause_seconds: float = 0.15) -> str:
    response = requests.get(url, headers=sec_headers(user_agent), timeout=90)
    response.raise_for_status()
    time.sleep(max(0.0, pause_seconds))
    return response.text


def download_company_tickers(cache_path: Path, user_agent: str, refresh: bool = False) -> pd.DataFrame:
    """Download the SEC ticker/CIK association file or load its local cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    payload = get_json(SEC_TICKERS_URL, user_agent)
    frame = pd.DataFrame(payload["data"], columns=payload["fields"])
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def ticker_cik_crosswalk(crsp_names: pd.DataFrame, sec_tickers: pd.DataFrame) -> pd.DataFrame:
    """Create a conservative exact-ticker bridge and flag every ambiguity."""
    crsp = crsp_names.copy()
    crsp["ticker_clean"] = crsp["ticker"].astype("string").str.upper().str.strip()
    sec = sec_tickers.rename(columns={"ticker": "ticker_clean", "name": "sec_name"}).copy()
    keep = [column for column in ["cik", "ticker_clean", "sec_name", "exchange"] if column in sec]
    merged = crsp.merge(sec[keep], on="ticker_clean", how="left")
    counts = merged.groupby("permco", dropna=False)["cik"].nunique(dropna=True)
    merged["cik_matches_for_permco"] = merged["permco"].map(counts).fillna(0).astype(int)
    merged["cik_match_status"] = "unmatched"
    merged.loc[merged["cik_matches_for_permco"].eq(1), "cik_match_status"] = "unique_exact_ticker"
    merged.loc[merged["cik_matches_for_permco"].gt(1), "cik_match_status"] = "ambiguous_exact_ticker"
    return merged


def filing_metadata_for_cik(
    cik: int,
    user_agent: str,
    forms: Iterable[str],
    start_date: str,
    cutoff_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Return qualifying filing metadata, caching the raw submissions JSON."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"CIK{int(cik):010d}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = get_json(SEC_SUBMISSIONS_URL.format(cik=int(cik)), user_agent)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    filing_frames = [pd.DataFrame(payload.get("filings", {}).get("recent", {}))]
    start, cutoff = pd.Timestamp(start_date), pd.Timestamp(cutoff_date)
    for descriptor in payload.get("filings", {}).get("files", []):
        file_start = pd.to_datetime(descriptor.get("filingFrom"), errors="coerce")
        file_end = pd.to_datetime(descriptor.get("filingTo"), errors="coerce")
        if pd.isna(file_start) or pd.isna(file_end) or file_end < start or file_start > cutoff:
            continue
        name = descriptor["name"]
        supplemental_path = cache_dir / name
        if supplemental_path.exists():
            supplemental = json.loads(supplemental_path.read_text(encoding="utf-8"))
        else:
            supplemental = get_json(SEC_SUBMISSIONS_FILE_URL.format(name=name), user_agent)
            supplemental_path.write_text(json.dumps(supplemental), encoding="utf-8")
        filing_frames.append(pd.DataFrame(supplemental))
    recent = pd.concat([frame for frame in filing_frames if not frame.empty], ignore_index=True)
    if recent.empty:
        return recent
    recent["cik"] = int(cik)
    recent["company_name"] = payload.get("name")
    recent["filingDate"] = pd.to_datetime(recent["filingDate"], errors="coerce")
    recent["reportDate"] = pd.to_datetime(recent["reportDate"], errors="coerce")
    wanted = {str(form).upper() for form in forms}
    recent = recent[
        recent["form"].astype(str).str.upper().isin(wanted)
        & recent["filingDate"].between(start, cutoff)
    ].copy()
    if recent.empty:
        return recent
    recent["accession_clean"] = recent["accessionNumber"].str.replace("-", "", regex=False)
    recent["source_uri"] = recent.apply(
        lambda row: SEC_ARCHIVE_URL.format(
            cik=int(row["cik"]), accession=row["accession_clean"], primary_document=row["primaryDocument"]
        ),
        axis=1,
    )
    return recent.sort_values(["filingDate", "accessionNumber"])


def html_to_text(html: str) -> str:
    """Convert filing HTML to readable text while removing scripts and styles."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Install beautifulsoup4 in the notebook kernel.") from exc
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def keyword_passages(
    text: str,
    terms: Iterable[str],
    radius: int = 900,
    max_passages: int = 30,
) -> List[dict]:
    """Retrieve auditable windows around geopolitical terms before LLM extraction."""
    lower = text.lower()
    hits = []
    for term in terms:
        escaped = re.escape(str(term).lower())
        pattern = rf"(?<!\w){escaped}(?!\w)"
        for match in re.finditer(pattern, lower):
            hits.append((match.start(), match.end(), str(term)))
    hits.sort()
    passages: List[dict] = []
    last_end = -1
    for start, end, term in hits:
        left, right = max(0, start - radius), min(len(text), end + radius)
        if left <= last_end:
            continue
        passages.append({"term": term, "start": left, "end": right, "text": text[left:right]})
        last_end = right
        if len(passages) >= max_passages:
            break
    return passages
