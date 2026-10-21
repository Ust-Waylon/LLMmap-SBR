"""Portable artifact IO and content fingerprints."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path):
    return list(iter_jsonl(path))


def split_path(data_dir, split):
    if split not in {"train", "valid", "test"}:
        raise ValueError("Unknown data split.")
    paths = [Path(data_dir) / f"{split}{suffix}" for suffix in (".jsonl", ".jsonl.gz")]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one plain or gzip file for the {split} split.")
    return found[0]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def data_fingerprint(data_dir):
    base = Path(data_dir)
    return fingerprint([base / "item_cards.json"] + [split_path(base, s) for s in ("train", "valid", "test")])


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Choose a new output directory.")
    path.mkdir(parents=True)
    return path
