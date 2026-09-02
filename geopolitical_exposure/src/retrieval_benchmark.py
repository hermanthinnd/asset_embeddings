from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CHANNEL_TERMS = {
    "defense_demand": "defense military government contract aerospace security weapons demand supplier customer",
    "oil_gas_production": "oil gas energy petroleum production reserves drilling pipeline commodity producer consumer",
    "russia_ukraine_operations_revenue": "Russia Russian Ukraine Ukrainian operations revenue assets customers facilities exposure",
    "sanctions_sensitive_supply_chain": "sanctions export controls restricted parties supply chain supplier customer sourcing compliance",
    "shipping_logistics": "shipping logistics freight transportation port ocean air cargo delivery route disruption",
}
MATCH_COLUMNS = ["beta_60m_paper", "log_market_equity", "log_book_to_market", "at_gr1_paper", "gp_at_paper", "momentum_12_1"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_embedding_candidates(root: Path, k: int = 10) -> pd.DataFrame:
    expansion = root / "embedding_expansion_v2"
    scores = pd.read_parquet(expansion / "embedding_propagated_scores.parquet")
    scores = scores.loc[scores["representation"].eq("all_s12")].copy()
    scores["permco"] = pd.to_numeric(scores["permco"], errors="coerce").astype("Int64")
    seeds = pd.read_parquet(expansion / "directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds["firm_id"], errors="coerce").astype("Int64")
    seed_ids = set(seeds["firm_id"].dropna())
    documents = load_jsonl(root / "text_expansion_v2/processed/expanded_sec_documents.jsonl")
    documented = {int(record["firm_id"]) for record in documents}
    scores = scores.loc[~scores["permco"].isin(seed_ids) & scores["permco"].isin(documented)].copy()
    direction = seeds.groupby("channel")["signed_score"].mean().map(np.sign).to_dict()
    selected = []
    for channel, group in scores.groupby("channel"):
        sign = int(direction[channel])
        chosen = group.nlargest(k, "propagated_score") if sign > 0 else group.nsmallest(k, "propagated_score")
        chosen = chosen.assign(channel_direction=sign, retrieval_rank=range(1, len(chosen) + 1))
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def match_controls(root: Path, candidates: pd.DataFrame, seed: int = 20220224) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(root / "text_expansion_v2/processed/expanded_300_firms_frozen.parquet")
    controls = pd.read_parquet(root.parent / "remaining_benchmarks/return_comovement/characteristics/2021q4.parquet")
    documents = load_jsonl(root / "text_expansion_v2/processed/expanded_sec_documents.jsonl")
    documented = {int(record["firm_id"]) for record in documents}
    universe = corpus[["permco", "comnam", "ticker", "sic2"]].drop_duplicates("permco").merge(controls, on="permco", how="inner")
    universe = universe.loc[universe["permco"].isin(documented)].copy()
    seeds = pd.read_parquet(root / "embedding_expansion_v2/directional_firm_channel_seeds.parquet")
    excluded = set(seeds["firm_id"].astype(int)) | set(candidates["permco"].astype(int))
    pool = universe.loc[~universe["permco"].isin(excluded)].dropna(subset=MATCH_COLUMNS).copy()
    candidate_features = candidates[["permco", "firm_name", "channel", "retrieval_rank"]].merge(universe, on="permco", how="inner")
    combined = pd.concat([candidate_features[MATCH_COLUMNS], pool[MATCH_COLUMNS]])
    means, scales = combined.mean(), combined.std(ddof=0).replace(0, 1)
    pool_z = (pool[MATCH_COLUMNS] - means) / scales
    used_by_channel = {channel: set() for channel in candidates["channel"].unique()}
    matches = []
    rng = np.random.default_rng(seed)
    order = candidate_features.sample(frac=1, random_state=seed)
    for row in order.itertuples(index=False):
        used = used_by_channel[row.channel]
        available = pool.loc[~pool["permco"].isin(used)].copy()
        target = (pd.Series({column: getattr(row, column) for column in MATCH_COLUMNS}) - means) / scales
        all_distance = np.square(pool_z.loc[available.index] - target).sum(axis=1)
        exact_industry = available.loc[available["sic2"].astype(str).eq(str(row.sic2))]
        exact_is_close = len(exact_industry) and all_distance.loc[exact_industry.index].min() <= 12.0
        if exact_is_close:
            available = exact_industry
        else:
            candidate_division = str(row.sic2)[:1]
            same_division = available.loc[available["sic2"].astype(str).str[:1].eq(candidate_division)]
            if len(same_division):
                available = same_division
        az = pool_z.loc[available.index]
        distance = np.square(az - target).sum(axis=1)
        jitter = pd.Series(rng.uniform(0, 1e-10, len(distance)), index=distance.index)
        index = (distance + jitter).idxmin()
        control = available.loc[index]
        used.add(int(control.permco))
        matches.append({"pair_channel": row.channel, "pair_rank": row.retrieval_rank,
                        "embedding_permco": row.permco, "embedding_name": row.firm_name,
                        "control_permco": control.permco, "control_name": control.comnam,
                        "embedding_sic2": row.sic2, "control_sic2": control.sic2,
                        "same_sic2": bool(str(row.sic2) == str(control.sic2)),
                        "match_distance": float(distance.loc[index])})
    matches = pd.DataFrame(matches).sort_values(["pair_channel", "pair_rank"])
    return matches, universe


def seed_contributors(root: Path, candidates: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    path = root.parent / "active_passive_extension/benchmarks/all_s12_control/2021q4/rs_l_min/asset_embeddings_10d.parquet"
    embedded = pd.read_parquet(path)
    columns = [column for column in embedded if column.startswith("embedding_")]
    matrix = embedded[columns].to_numpy(float)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    lookup = {int(value): index for index, value in enumerate(embedded["permco"])}
    seeds = pd.read_parquet(root / "embedding_expansion_v2/directional_firm_channel_seeds.parquet")
    seeds["firm_id"] = pd.to_numeric(seeds["firm_id"], errors="coerce").astype("Int64")
    names = embedded[["permco", "firm_name"]].rename(columns={"permco": "firm_id", "firm_name": "seed_name"})
    seeds = seeds.merge(names, on="firm_id", how="left")
    rows = []
    for candidate in candidates.itertuples():
        eligible = seeds.loc[seeds["channel"].eq(candidate.channel) & seeds["firm_id"].map(lambda x: pd.notna(x) and int(x) in lookup)].copy()
        eligible["cosine_similarity"] = [float(matrix[lookup[int(candidate.permco)]] @ matrix[lookup[int(seed)]]) for seed in eligible["firm_id"]]
        for rank, item in enumerate(eligible.nlargest(n, "cosine_similarity").itertuples(), 1):
            rows.append({"channel": candidate.channel, "retrieval_rank": candidate.retrieval_rank,
                         "candidate_permco": candidate.permco, "contributor_rank": rank,
                         "seed_permco": item.firm_id, "seed_name": item.seed_name,
                         "seed_signed_score": item.signed_score, "cosine_similarity": item.cosine_similarity})
    return pd.DataFrame(rows)


def freeze_blinded_queue(root: Path, k: int = 10, seed: int = 20220224) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = select_embedding_candidates(root, k)
    matches, universe = match_controls(root, candidates, seed)
    contributors = seed_contributors(root, candidates)
    references = (contributors.sort_values("contributor_rank").groupby(["channel", "retrieval_rank"])
                  .agg(reference_seed_names=("seed_name", lambda x: " | ".join(x)),
                       reference_seed_permcos=("seed_permco", lambda x: " | ".join(x.astype(str))))
                  .reset_index())
    embedding = matches[["pair_channel", "pair_rank", "embedding_permco", "embedding_name"]].rename(
        columns={"pair_channel": "channel", "pair_rank": "retrieval_rank", "embedding_permco": "permco", "embedding_name": "firm_name"}
    ).assign(hidden_source="embedding")
    control = matches[["pair_channel", "pair_rank", "control_permco", "control_name"]].rename(
        columns={"pair_channel": "channel", "pair_rank": "retrieval_rank", "control_permco": "permco", "control_name": "firm_name"}
    ).assign(hidden_source="matched_control")
    hidden = pd.concat([embedding, control], ignore_index=True).merge(references, on=["channel", "retrieval_rank"], how="left")
    rng = np.random.default_rng(seed)
    hidden = hidden.iloc[rng.permutation(len(hidden))].reset_index(drop=True)
    hidden.insert(0, "blind_id", [f"GEO-{number:03d}" for number in range(1, len(hidden) + 1)])
    public = hidden.drop(columns="hidden_source").copy()
    public = public.merge(universe[["permco", "ticker", "sic2"]], on="permco", how="left")
    hidden = hidden.merge(universe[["permco", "ticker", "sic2"]], on="permco", how="left")
    return public, hidden, matches


def _chunks(text: str, size: int = 1600, overlap: int = 250) -> list[tuple[int, int, str]]:
    clean = re.sub(r"\s+", " ", text).strip()
    return [(start, min(len(clean), start + size), clean[start:start + size])
            for start in range(0, len(clean), size - overlap) if len(clean[start:start + size]) >= 200]


def build_evidence_packets(root: Path, public_queue: pd.DataFrame, passages: int = 8) -> list[dict]:
    documents = load_jsonl(root / "text_expansion_v2/processed/expanded_sec_documents.jsonl")
    by_firm: dict[int, list[dict]] = {}
    for document in documents:
        by_firm.setdefault(int(document["firm_id"]), []).append(document)
    packets = []
    for row in public_queue.itertuples(index=False):
        available = sorted(by_firm.get(int(row.permco), []), key=lambda x: x["available_date"], reverse=True)
        if not available:
            packets.append({"blind_id": row.blind_id, "firm_id": int(row.permco), "channel": row.channel,
                            "reference_seed_names": row.reference_seed_names, "passages": [], "document_available": False})
            continue
        document = available[0]
        chunks = _chunks(document["text"])
        seed_terms = re.sub(r"[^A-Za-z0-9 ]+", " ", row.reference_seed_names)
        query = CHANNEL_TERMS[row.channel] + " " + seed_terms + " supplier customer competitor peer relationship exposure"
        texts = [item[2] for item in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts + [query])
        scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        chosen = np.argsort(scores)[::-1][:passages]
        selected = [{"rank": rank, "start": chunks[index][0], "end": chunks[index][1],
                     "retrieval_score": float(scores[index]), "text": chunks[index][2]}
                    for rank, index in enumerate(chosen, 1)]
        packets.append({"blind_id": row.blind_id, "firm_id": int(row.permco), "firm_name": row.firm_name,
                        "ticker": row.ticker, "channel": row.channel,
                        "reference_seed_names": row.reference_seed_names,
                        "document_id": document["document_id"], "available_date": document["available_date"],
                        "content_sha256": document["content_sha256"], "document_available": True,
                        "passages": selected})
    return packets


def queue_hash(queue: pd.DataFrame) -> str:
    return hashlib.sha256(queue.sort_values("blind_id").to_json(orient="records").encode()).hexdigest()
