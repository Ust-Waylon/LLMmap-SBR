"""Full-catalog training, validation checkpoint selection, and last-click evaluation."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .data import load_split
from .io import read_json, write_json, new_directory, data_fingerprint, fingerprint
from .model import LLMTransform

logger = logging.getLogger(__name__)


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if set(cfg) != {"seed", "model", "train"}:
        raise ValueError("Config must contain exactly seed, model, and train sections.")
    if set(cfg["train"]) != {"lr", "weight_decay", "batch_size", "epochs", "patience", "max_grad_norm"}:
        raise ValueError("train must contain exactly lr, weight_decay, batch_size, epochs, patience, and max_grad_norm.")
    if int(cfg["train"]["epochs"]) < 1 or int(cfg["train"]["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    return cfg


def load_catalog(data_dir, language_dir):
    data_dir, language_dir = Path(data_dir), Path(language_dir)
    cards = read_json(data_dir / "item_cards.json")
    item_ids = read_json(language_dir / "item_ids.json")
    if item_ids != list(cards):
        raise ValueError("Language representation row order does not match item_cards.json. Re-encode this catalog.")
    metadata = read_json(language_dir / "encoder_config.json")
    if metadata["data_fingerprint"] != data_fingerprint(data_dir):
        raise ValueError("Data changed after encoding. Use matching data and language representations.")
    checksum = fingerprint([language_dir / "item_ids.json", language_dir / "language_representations.npy"])
    if metadata["language_fingerprint"] != checksum:
        raise ValueError("Language representation fingerprint mismatch.")
    language_reps = np.load(language_dir / "language_representations.npy", allow_pickle=False)
    if language_reps.shape != (len(item_ids), 2560) or not np.isfinite(language_reps).all():
        raise ValueError("Expected a finite [catalog size, 2560] language representation matrix.")
    if not np.allclose(np.linalg.norm(language_reps, axis=1), 1.0, atol=1e-4):
        raise ValueError("Expected L2-normalized language representations.")
    return item_ids, language_reps, metadata


def metrics(top_items, targets):
    if len(targets) == 0:
        raise ValueError("Cannot evaluate an empty split.")
    matches = top_items == targets[:, None]
    hits = matches.any(axis=1)
    rank = matches.argmax(axis=1) + 1
    return {"HR@20": float(hits.mean()), "MRR@20": float(np.where(hits, 1.0 / rank, 0.0).mean()),
            "n_examples": int(len(targets))}


@torch.no_grad()
def evaluate(model, language_reps, sequences, targets, *, device, batch_size=512):
    model.eval()
    language = torch.as_tensor(language_reps, dtype=torch.float32, device=device)
    item_embeddings = model.transform_items(language)
    top = []
    for start in range(0, len(targets), batch_size):
        history = torch.as_tensor(sequences[start:start + batch_size], device=device)
        scores = model(language, history, item_embeddings=item_embeddings)
        top.append(scores.topk(min(20, item_embeddings.shape[0]), dim=1).indices.cpu().numpy())
    return metrics(np.concatenate(top), targets)


def train(config_path, data_dir, language_dir, output, *, device="cuda"):
    cfg = load_config(config_path)
    if Path(output).exists():
        raise FileExistsError(f"Output already exists: {output}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    item_ids, language_reps, encoder_meta = load_catalog(data_dir, language_dir)
    model = LLMTransform(**cfg["model"]).to(device)
    tcfg = cfg["train"]
    train_seq, train_targets = load_split(data_dir, "train", item_ids, cfg["model"]["max_length"])
    valid_seq, valid_targets = load_split(data_dir, "valid", item_ids, cfg["model"]["max_length"])
    # Test data are not used to select a checkpoint.
    loader = DataLoader(TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_targets)),
                        batch_size=tcfg["batch_size"], shuffle=True, drop_last=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    language = torch.as_tensor(language_reps, dtype=torch.float32, device=device)
    output = new_directory(output)
    write_json(output / "config.json", cfg)
    eval_kwargs = {"device": device, "batch_size": tcfg["batch_size"]}
    best_valid = evaluate(model, language_reps, valid_seq, valid_targets, **eval_kwargs)
    best_epoch, bad_epochs = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "valid": best_valid}]
    logger.info("initial valid HR@20=%.4f MRR@20=%.4f", best_valid["HR@20"], best_valid["MRR@20"])
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)
            scores = model(language, seq)
            loss = F.cross_entropy(scores, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        valid = evaluate(model, language_reps, valid_seq, valid_targets, **eval_kwargs)
        history.append({"epoch": epoch, "train_ce": total_loss / len(train_targets), "valid": valid})
        logger.info("epoch %d CE=%.4f valid HR@20=%.4f MRR@20=%.4f", epoch,
                    history[-1]["train_ce"], valid["HR@20"], valid["MRR@20"])
        if valid["HR@20"] > best_valid["HR@20"] + 1e-6:
            best_valid, best_epoch, bad_epochs = valid, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        write_json(output / "train_log.json", history)
        if bad_epochs >= tcfg["patience"]:
            break
    torch.save({"state_dict": best_state, "config": cfg, "item_ids": item_ids,
                "data_fingerprint": encoder_meta["data_fingerprint"],
                "language_fingerprint": encoder_meta["language_fingerprint"],
                "best_epoch": best_epoch}, output / "model.pt")
    write_json(output / "summary.json", {"best_epoch": best_epoch, "best_valid": best_valid,
               "epochs_ran": len(history) - 1, "n_parameters": sum(p.numel() for p in model.parameters())})
    return output


def evaluate_checkpoint(checkpoint, data_dir, language_dir, *, split="test", device="cuda"):
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation uses last-click valid or test only.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    item_ids, language_reps, metadata = load_catalog(data_dir, language_dir)
    if (saved["item_ids"] != item_ids or saved["data_fingerprint"] != metadata["data_fingerprint"]
            or saved["language_fingerprint"] != metadata["language_fingerprint"]):
        raise ValueError("Checkpoint, data, and frozen language representations do not match.")
    cfg = saved["config"]
    model = LLMTransform(**cfg["model"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    seq, targets = load_split(data_dir, split, item_ids, cfg["model"]["max_length"])
    return {"split": split, **evaluate(model, language_reps, seq, targets, device=device,
            batch_size=cfg["train"]["batch_size"])}
