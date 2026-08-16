"""Paper-faithful quarterly CBOW asset embeddings for Asset Embeddings (2025)."""
from pathlib import Path
import time

import numpy as np
import pandas as pd


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Run the Word2Vec dependency cell (`%pip install torch`) and restart the kernel.") from exc
    return torch


def choose_device():
    torch = require_torch()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def portfolio_sequences(holdings_path, investor_mapping_path, firm_mapping_path):
    holdings = pd.read_parquet(holdings_path)
    investors = pd.read_parquet(investor_mapping_path)
    firms = pd.read_parquet(firm_mapping_path).sort_values("firm_index").reset_index(drop=True)
    indexed = (
        holdings.merge(investors[["investor_key", "investor_index"]], on="investor_key", validate="many_to_one")
        .merge(firms[["permco", "firm_index"]], on="permco", validate="many_to_one")
        .sort_values(["investor_index", "portfolio_weight", "permco"], ascending=[True, False, True])
    )
    sequences = [g["firm_index"].to_numpy(dtype="int64") for _, g in indexed.groupby("investor_index", sort=True)]
    return sequences, firms


def iter_cbow_batches(sequences, radius, batch_size, shuffle_seed=None, validation_fraction=0.0, validation=False):
    """Yield padded contexts and targets; portfolios stay ordered by decreasing weights."""
    rng = np.random.default_rng(shuffle_seed)
    order = np.arange(len(sequences))
    if shuffle_seed is not None:
        rng.shuffle(order)
    contexts, targets = [], []
    for sequence_index in order:
        sequence = sequences[sequence_index]
        for position, target in enumerate(sequence):
            marker = ((sequence_index * 1_000_003 + position * 97) % 10_000) / 10_000
            is_validation = marker < validation_fraction
            if is_validation != validation:
                continue
            left, right = max(0, position - radius), min(len(sequence), position + radius + 1)
            context = np.concatenate([sequence[left:position], sequence[position + 1:right]])
            if context.size == 0:
                continue
            contexts.append(context)
            targets.append(target)
            if len(targets) == batch_size:
                yield _pad_batch(contexts, targets)
                contexts, targets = [], []
    if targets:
        yield _pad_batch(contexts, targets)


def _pad_batch(contexts, targets):
    width = max(map(len, contexts))
    values = np.full((len(contexts), width), -1, dtype="int64")
    mask = np.zeros((len(contexts), width), dtype="bool")
    for row, context in enumerate(contexts):
        values[row, :len(context)] = context
        mask[row, :len(context)] = True
    return values, mask, np.asarray(targets, dtype="int64")


def fit_cbow(sequences, vocabulary_size, dimension, radius, epochs, batch_size, learning_rate,
             validation_fraction, seed, patience=2, device=None):
    torch = require_torch()
    torch.manual_seed(seed)
    device = device or choose_device()
    embedding = torch.nn.Embedding(vocabulary_size, dimension, device=device)
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=1 / np.sqrt(dimension))
    optimizer = torch.optim.AdamW(embedding.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_loss, best_weights, stale, history = np.inf, None, 0, []
    for epoch in range(epochs):
        embedding.train()
        train_loss, train_n = 0.0, 0
        batches = iter_cbow_batches(sequences, radius, batch_size, seed + epoch, validation_fraction, False)
        for context_np, mask_np, target_np in batches:
            context = torch.as_tensor(context_np, device=device)
            mask = torch.as_tensor(mask_np, device=device)
            target = torch.as_tensor(target_np, device=device)
            safe_context = context.clamp_min(0)
            vectors = embedding(safe_context) * mask.unsqueeze(-1)
            mean_context = vectors.sum(1) / mask.sum(1, keepdim=True)
            logits = mean_context @ embedding.weight.T
            row_ids = torch.arange(len(target), device=device).unsqueeze(1).expand_as(safe_context)
            logits[row_ids[mask], safe_context[mask]] = float("-inf")
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target_np)
            train_n += len(target_np)
        validation_loss, validation_n = evaluate_cbow(
            embedding, sequences, radius, batch_size, validation_fraction, device
        )
        history.append({"epoch": epoch + 1, "training_loss": train_loss / train_n,
                        "validation_loss": validation_loss, "training_examples": train_n,
                        "validation_examples": validation_n})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_weights = embedding.weight.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    embedding.weight.data.copy_(best_weights.to(device))
    return embedding.weight.detach().cpu().numpy(), pd.DataFrame(history), str(device)


def evaluate_cbow(embedding, sequences, radius, batch_size, validation_fraction, device):
    torch = require_torch()
    embedding.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for context_np, mask_np, target_np in iter_cbow_batches(
            sequences, radius, batch_size, None, validation_fraction, True
        ):
            context = torch.as_tensor(context_np, device=device)
            mask = torch.as_tensor(mask_np, device=device)
            target = torch.as_tensor(target_np, device=device)
            safe_context = context.clamp_min(0)
            vectors = embedding(safe_context) * mask.unsqueeze(-1)
            mean_context = vectors.sum(1) / mask.sum(1, keepdim=True)
            logits = mean_context @ embedding.weight.T
            row_ids = torch.arange(len(target), device=device).unsqueeze(1).expand_as(safe_context)
            logits[row_ids[mask], safe_context[mask]] = float("-inf")
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            total += loss.item()
            count += len(target_np)
    return total / count, count


def normalize_embeddings(values):
    covariance = values.T @ values / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = eigenvectors @ np.diag(1 / np.sqrt(np.maximum(eigenvalues, 1e-10))) @ eigenvectors.T
    return (values @ whitening).astype("float32")


def train_quarter(sequences, firms, dimension, radius, output_directory, config, force=False):
    output_directory = Path(output_directory)
    embedding_path = output_directory / "asset_embeddings_labeled.parquet"
    summary_path = output_directory / "fit_summary.parquet"
    if embedding_path.exists() and summary_path.exists() and not force:
        return pd.read_parquet(embedding_path), pd.read_parquet(summary_path)
    start = time.perf_counter()
    values, history, device = fit_cbow(
        sequences, len(firms), dimension, radius, config["epochs"], config["batch_size"],
        config["learning_rate"], config["validation_fraction"], config["seed"] + dimension
    )
    values = normalize_embeddings(values)
    labeled = firms[["permco", "firm_name", "firm_index"]].copy()
    for index in range(dimension):
        labeled[f"embedding_{index + 1:03d}"] = values[:, index]
    best = history.loc[history["validation_loss"].idxmin()]
    summary = pd.DataFrame({"dimension": [dimension], "context_radius": [radius],
                            "epochs_completed": [len(history)], "best_epoch": [int(best["epoch"])],
                            "best_validation_loss": [best["validation_loss"]],
                            "device": [device], "fit_seconds": [time.perf_counter() - start],
                            "firms": [len(firms)], "portfolios": [len(sequences)],
                            "model_version": [1]})
    output_directory.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(embedding_path, index=False)
    summary.to_parquet(summary_path, index=False)
    history.to_parquet(output_directory / "training_history.parquet", index=False)
    return labeled, summary
