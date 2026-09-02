from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_checkpoint(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if not path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        status = "downloaded"
    return {
        "file": path.name,
        "status": status,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "source_url": url,
    }


def quarter_labels(start="2005Q1", end="2024Q4") -> list[str]:
    return [str(period).lower() for period in pd.period_range(start, end, freq="Q")]


def parquet_audit(path: Path, label: str, expected_key="permco") -> dict:
    record = {"input": label, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    frame = pd.read_parquet(path)
    embedding_columns = [c for c in frame if c.startswith("embedding_")]
    record.update(
        rows=len(frame),
        columns=len(frame.columns),
        embedding_dimensions=len(embedding_columns),
        key_present=expected_key in frame,
        unique_keys=frame[expected_key].nunique() if expected_key in frame else np.nan,
        duplicate_keys=int(frame.duplicated(expected_key).sum()) if expected_key in frame else np.nan,
        bytes=path.stat().st_size,
    )
    return record


def inventory_excel(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    workbook = pd.ExcelFile(path)
    frames, rows = {}, []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        frames[sheet] = frame
        rows.append({
            "container": path.name,
            "member": sheet,
            "rows": len(frame),
            "columns": len(frame.columns),
            "first_columns": ", ".join(map(str, frame.columns[:12])),
        })
    return pd.DataFrame(rows), frames


def inventory_zip(path: Path, extract_directory: Path) -> pd.DataFrame:
    extract_directory.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = extract_directory / Path(info.filename).name
            if not target.exists():
                target.write_bytes(archive.read(info.filename))
            records.append({
                "container": path.name,
                "member": info.filename,
                "bytes": info.file_size,
                "local_path": str(target),
            })
    return pd.DataFrame(records)


def inspect_tabular_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif suffix in {".rds", ".rdata"}:
            return {"path": str(path), "format": suffix, "readable_by_pandas": False}
        else:
            return {"path": str(path), "format": suffix, "readable_by_pandas": False}
        date_candidates = [c for c in frame if re.search(r"date|event|fomc", str(c), re.I)]
        numeric = frame.select_dtypes(include="number")
        return {
            "path": str(path),
            "format": suffix,
            "readable_by_pandas": True,
            "rows": len(frame),
            "columns": len(frame.columns),
            "date_candidates": ", ".join(map(str, date_candidates[:10])),
            "numeric_columns": ", ".join(map(str, numeric.columns[:20])),
            "first_columns": ", ".join(map(str, frame.columns[:20])),
        }
    except Exception as error:
        return {"path": str(path), "format": suffix, "readable_by_pandas": False, "error": repr(error)}


def permanent_split(permco: pd.Series, seed: str = "rate-shock-v1") -> pd.Series:
    def assign(value):
        token = hashlib.sha256(f"{seed}:{int(value)}".encode()).hexdigest()
        u = int(token[:12], 16) / float(16**12)
        return "train" if u < 0.60 else "validation" if u < 0.80 else "test"
    return permco.map(assign)
