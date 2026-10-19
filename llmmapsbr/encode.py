"""Encode each unique item card once with the frozen Qwen3 embedder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import read_json, write_json, new_directory, data_fingerprint, fingerprint


def encode(data_dir, output, *, device="cuda", batch_size=16,
           model_name="Qwen/Qwen3-Embedding-4B"):
    from sentence_transformers import SentenceTransformer

    data_dir = Path(data_dir)
    if Path(output).exists():
        raise FileExistsError(f"Output already exists: {output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cards = read_json(data_dir / "item_cards.json")
    if not cards or any(not str(text).strip() for text in cards.values()):
        raise ValueError("Catalog item cards must be nonempty.")
    item_ids = list(cards)
    # Keep the released model's own pooling and tokenizer configuration.
    # No query instructions, adapters, or prompt wrappers are added.
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.max_seq_length = 1024
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    texts = list(cards.values())
    unique = list(dict.fromkeys(texts))
    inverse = {text: i for i, text in enumerate(unique)}
    with torch.inference_mode():
        encoded = model.encode(unique, batch_size=batch_size, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(encoded, dtype=np.float32)[[inverse[text] for text in texts]]
    if embeddings.shape != (len(item_ids), 2560) or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite 2560-dimensional Qwen3-Embedding-4B vectors.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Language representations must have unit L2 norm.")
    output = new_directory(output)
    np.save(output / "item_embeddings.npy", embeddings, allow_pickle=False)
    write_json(output / "item_ids.json", item_ids)
    write_json(output / "encoder_config.json", {
        "model_name": model_name, "max_length": 1024, "normalize": True,
        "batch_size": batch_size, "input_dim": 2560,
        "model_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "data_fingerprint": data_fingerprint(data_dir),
        "embedding_fingerprint": fingerprint([output / "item_ids.json", output / "item_embeddings.npy"]),
    })
