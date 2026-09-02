from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


ADJUDICATION_POLICY = {
    "name": "conservative_exact_evidence_v1",
    "positive_rule": "supported_relationship AND machine_valid",
    "invalid_response_treatment": "unsupported",
    "uses_candidate_origin": False,
    "uses_event_returns": False,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_blinded_outcomes(public: Path) -> tuple[pd.DataFrame, dict]:
    audit_path = public / "relationship_response_audit.parquet"
    audit = pd.read_parquet(audit_path)
    required = {"blind_id", "supported_relationship", "machine_valid"}
    missing = required - set(audit.columns)
    if missing:
        raise KeyError(f"Rerun Sections 1 and 4 of Notebook 07B; missing audit columns: {sorted(missing)}")
    if len(audit) != 100 or audit["blind_id"].duplicated().any():
        raise ValueError("Expected exactly 100 unique blinded responses.")
    frozen = audit.copy()
    frozen["adjudicated_supported"] = (
        frozen["supported_relationship"].fillna(False).astype(bool)
        & frozen["machine_valid"].fillna(False).astype(bool)
    )
    keep = ["blind_id", "relationship_type", "relationship_direction", "channel_relevance",
            "confidence", "evidence_quote", "reasoning_summary", "machine_valid",
            "adjudicated_supported"]
    frozen = frozen[keep].sort_values("blind_id").reset_index(drop=True)
    output_path = public / "adjudicated_blinded_outcomes.parquet"
    frozen.to_parquet(output_path, index=False)
    manifest = {
        "policy": ADJUDICATION_POLICY,
        "input_audit_sha256": sha256_file(audit_path),
        "output_sha256": sha256_file(output_path),
        "responses": len(frozen),
        "adjudicated_supported": int(frozen["adjudicated_supported"].sum()),
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (public / "adjudication_manifest.json").write_text(json.dumps(manifest, indent=2))
    return frozen, manifest


def unblind(public: Path, restricted_truth: Path) -> pd.DataFrame:
    frozen = pd.read_parquet(public / "adjudicated_blinded_outcomes.parquet")
    truth = pd.read_parquet(restricted_truth / "candidate_source_mapping.parquet")
    panel = frozen.merge(truth, on="blind_id", how="left", validate="one_to_one")
    if panel["hidden_source"].isna().any():
        raise ValueError("Some frozen outcomes did not map to restricted truth.")
    return panel


def source_summary(panel: pd.DataFrame) -> pd.DataFrame:
    out = (panel.groupby("hidden_source", as_index=False)
           .agg(candidates=("blind_id", "size"),
                supported=("adjudicated_supported", "sum"),
                mean_confidence=("confidence", "mean")))
    out["support_rate"] = out["supported"] / out["candidates"]
    return out


def channel_summary(panel: pd.DataFrame) -> pd.DataFrame:
    out = (panel.groupby(["channel", "hidden_source"], as_index=False)
           .agg(candidates=("blind_id", "size"), supported=("adjudicated_supported", "sum")))
    out["support_rate"] = out["supported"] / out["candidates"]
    return out


def retrieval_diagnostics(public: Path, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    packets = []
    with (public / "blinded_evidence_packets.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            packet = json.loads(line)
            passages = packet["passages"]
            packets.append({
                "blind_id": packet["blind_id"],
                "document_available": bool(packet["document_available"]),
                "passage_count": len(passages),
                "retrieved_characters": sum(len(item["text"]) for item in passages),
                "top_retrieval_score": max(item["retrieval_score"] for item in passages),
                "mean_retrieval_score": np.mean([item["retrieval_score"] for item in passages]),
            })
    detail = pd.DataFrame(packets).merge(
        panel[["blind_id", "hidden_source"]], on="blind_id", how="left", validate="one_to_one"
    )
    summary = (detail.groupby("hidden_source", as_index=False)
               .agg(candidates=("blind_id", "size"),
                    documents_available=("document_available", "sum"),
                    mean_passages=("passage_count", "mean"),
                    mean_characters=("retrieved_characters", "mean"),
                    mean_top_retrieval_score=("top_retrieval_score", "mean"),
                    mean_retrieval_score=("mean_retrieval_score", "mean")))
    return detail, summary


def matching_diagnostics(restricted_truth: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(restricted_truth / "embedding_control_pairs.parquet")
    overall = pd.DataFrame([{
        "pairs": len(pairs), "exact_sic2_pairs": int(pairs["same_sic2"].sum()),
        "exact_sic2_rate": float(pairs["same_sic2"].mean()),
        "median_match_distance": float(pairs["match_distance"].median()),
        "maximum_match_distance": float(pairs["match_distance"].max()),
    }])
    by_channel = (pairs.groupby("pair_channel", as_index=False)
                  .agg(pairs=("pair_rank", "size"), exact_sic2_rate=("same_sic2", "mean"),
                       median_match_distance=("match_distance", "median"),
                       maximum_match_distance=("match_distance", "max")))
    return overall, by_channel


def inferential_tests(panel: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(panel["hidden_source"], panel["adjudicated_supported"])
    for value in [False, True]:
        if value not in table.columns:
            table[value] = 0
    table = table.reindex(["embedding", "matched_control"], fill_value=0)
    # Boolean column labels must be selected with reindex; ``table[[True, False]]``
    # can be interpreted as a row mask by pandas.
    contingency = table.reindex(columns=[True, False], fill_value=0).to_numpy()
    odds, fisher_p_greater = fisher_exact(contingency, alternative="greater")
    _, fisher_p_two_sided = fisher_exact(contingency, alternative="two-sided")

    pairs = panel.pivot(index=["channel", "retrieval_rank"], columns="hidden_source",
                        values="adjudicated_supported")
    if pairs.isna().any().any() or len(pairs) != 50:
        raise ValueError("Expected 50 complete channel-rank matched pairs.")
    emb_only = int(((pairs["embedding"] == 1) & (pairs["matched_control"] == 0)).sum())
    control_only = int(((pairs["embedding"] == 0) & (pairs["matched_control"] == 1)).sum())
    discordant = emb_only + control_only
    paired_p_greater = binomtest(emb_only, discordant, 0.5, alternative="greater").pvalue if discordant else 1.0
    paired_p_two_sided = binomtest(emb_only, discordant, 0.5, alternative="two-sided").pvalue if discordant else 1.0
    return pd.DataFrame([
        {"test": "Fisher exact", "alternative": "embedding > control",
         "embedding_only": np.nan, "control_only": np.nan,
         "odds_ratio": odds, "p_value": fisher_p_greater},
        {"test": "Fisher exact", "alternative": "two-sided",
         "embedding_only": np.nan, "control_only": np.nan,
         "odds_ratio": odds, "p_value": fisher_p_two_sided},
        {"test": "exact paired sign", "alternative": "embedding > control",
         "embedding_only": emb_only, "control_only": control_only,
         "odds_ratio": np.nan, "p_value": paired_p_greater},
        {"test": "exact paired sign", "alternative": "two-sided",
         "embedding_only": emb_only, "control_only": control_only,
         "odds_ratio": np.nan, "p_value": paired_p_two_sided},
    ])
