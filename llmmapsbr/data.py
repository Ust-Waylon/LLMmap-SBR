"""Item cards, prefix training examples, and chronological model inputs."""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np

from .io import iter_jsonl, split_path


def clean_text(value, max_length=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        text = "; ".join(filter(None, (clean_text(x) for x in value)))
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if max_length and len(text) > max_length:
        text = text[:max_length - 1].rstrip() + "…"
    return text


def item_card(row):
    categories = row.get("categories")
    if isinstance(categories, str) and categories.strip().startswith("["):
        categories = json.loads(categories)
    if isinstance(categories, list):
        flat = []
        for entry in categories:
            if isinstance(entry, list):
                flat.extend(str(x) for x in entry if x is not None)
            elif entry is not None:
                flat.append(str(entry))
        categories = " > ".join(flat)
    else:
        categories = clean_text(categories)
    leaf = categories.split(" > ")[-1].strip() if categories else None
    fields = [
        ("Title", clean_text(row.get("title"), 160)),
        ("Brand", clean_text(row.get("store", row.get("brand")))),
        ("Category", leaf or clean_text(row.get("main_category"))),
        ("Price", clean_text(row.get("price"))),
    ]
    text = " | ".join(f"{label}: {value}" for label, value in fields if value)
    return text if len(text) <= 400 else text[:399].rstrip() + "…"


def examples(sessions, *, training, max_length=20):
    for session in sessions:
        items = session.get("item_id_list")
        if items is None:
            items = session["history"] + [session["target"]]
        items = [str(item) for item in items]
        if len(items) < 2:
            raise ValueError("Each session must contain a history and a target.")
        ends = range(1, len(items)) if training else [len(items) - 1]
        for end in ends:
            yield {"history": items[max(0, end - max_length):end], "target": items[end]}


def pack_histories(histories, id_to_row, max_length=20):
    """Left-pad with -1. Inputs depend only on history, including repeated clicks."""
    packed = np.full((len(histories), max_length), -1, dtype=np.int64)
    for i, history in enumerate(histories):
        history = list(history)[-max_length:]
        if not history:
            raise ValueError("Empty model history.")
        indices = [id_to_row[str(item)] for item in history]
        packed[i, -len(indices):] = indices
    return packed


def load_split(data_dir, split, item_ids, max_length=20):
    path = split_path(data_dir, split)
    # Count first, then fill arrays without retaining millions of Python records.
    count = sum(1 for _ in iter_jsonl(path))
    if not count:
        raise ValueError(f"Empty {split} split.")
    id_to_row = {str(item): i for i, item in enumerate(item_ids)}
    seq = np.full((count, max_length), -1, dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    for i, row in enumerate(iter_jsonl(path)):
        history = row["history"][-max_length:]
        if not history:
            raise ValueError("Empty model history.")
        seq[i, -len(history):] = [id_to_row[str(item)] for item in history]
        targets[i] = id_to_row[str(row["target"])]
    return seq, targets


def prepare(interactions_path, metadata_path, output, category):
    from ._amazon import load_interactions, load_metadata, join_interactions_with_meta, meta_for_parquet
    from ._sessions import assign_sbr_sessions, iterative_sbr_filter, build_session_sequences, chronological_session_split
    from .io import new_directory, write_json, write_jsonl, data_fingerprint

    if Path(output).exists():
        raise FileExistsError(f"Output already exists: {output}")
    interactions = load_interactions(Path(interactions_path))
    meta = load_metadata(Path(metadata_path), set(interactions["item_id"]), require_title=True)
    interactions, meta, _ = join_interactions_with_meta(interactions, meta)
    sessions = assign_sbr_sessions(interactions, days_per_session=1, only_first_day=True)
    sessions, _ = iterative_sbr_filter(sessions, min_session_len=2, min_item_freq=5)
    sequences = build_session_sequences(sessions, max_seq_len=20, truncate_from="head")
    splits = chronological_session_split(sequences)
    catalog = set(item for row in splits["train"] for item in row["item_id_list"])
    # Match the string-valued metadata representation used to build the item cards.
    meta = meta_for_parquet(meta[meta["item_id"].isin(catalog)])
    cards = {str(r["item_id"]): item_card(r) for r in meta.to_dict(orient="records")}
    if set(cards) != catalog or any(not value for value in cards.values()):
        raise ValueError("Every training catalog item must have a nonempty item card.")
    output = new_directory(output)
    write_json(output / "item_cards.json", cards)
    counts = {}
    for split in ("train", "valid", "test"):
        rows = list(examples(splits[split], training=split == "train"))
        if not rows:
            raise ValueError(f"Preprocessing produced an empty {split} split.")
        write_jsonl(output / f"{split}.jsonl", rows)
        counts[split] = {"sessions": len(splits[split]), "examples": len(rows)}
    write_json(output / "data_config.json", {
        "category": category, "n_items": len(cards), "splits": counts,
        "first_day_only": True, "calendar_timezone": "UTC",
        "min_session_length": 2, "min_item_frequency": 5,
        "remove_consecutive_duplicates": True, "max_session_length": 20,
        "truncate_from": "head", "split_ratios": [0.8, 0.1, 0.1],
        "train_prefix_augmentation": True, "valid_test_last_click_only": True,
        "fingerprint": data_fingerprint(output),
    })
