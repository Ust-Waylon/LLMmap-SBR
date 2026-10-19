"""Amazon Reviews 2023 local input loaders."""
from __future__ import annotations
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from tqdm import tqdm
logger = logging.getLogger(__name__)
META_KEEP_FIELDS = ("parent_asin", "title", "store", "categories", "main_category", "price")

def load_interactions(csv_path: Path) -> pd.DataFrame:
    """Load 5-core rating CSV → normalized columns user_id, item_id, rating, timestamp."""
    logger.info("Loading interactions from %s", csv_path)
    df = pd.read_csv(csv_path)
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}; got {list(df.columns)}")

    df = df.rename(columns={"parent_asin": "item_id"})
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    n = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    logger.info("Loaded interactions: rows=%d users=%d items=%d", n, n_users, n_items)
    return df[["user_id", "item_id", "rating", "timestamp"]]


def _normalize_title(title: Any) -> str | None:
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return None
    s = str(title).strip()
    return s if s else None


def iter_meta_records(meta_file: Path) -> Iterable[dict[str, Any]]:
    """Yield raw meta dicts from jsonl or jsonl.gz."""
    path = Path(meta_file)
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_metadata(
    meta_file: Path,
    item_ids: set[str] | None = None,
    require_title: bool = True,
) -> pd.DataFrame:
    """
    Load metadata jsonl.gz into a DataFrame keyed by item_id (= parent_asin).

    If item_ids is given, only those parent_asins are kept (much faster / smaller).
    Drops items without usable title when require_title=True.
    """
    logger.info("Loading metadata from %s", meta_file)
    rows: list[dict[str, Any]] = []
    n_seen = 0
    n_kept = 0
    for rec in tqdm(iter_meta_records(meta_file), desc="meta", unit="rec"):
        n_seen += 1
        pid = rec.get("parent_asin") or rec.get("asin")
        if pid is None:
            continue
        pid = str(pid)
        if item_ids is not None and pid not in item_ids:
            continue
        title = _normalize_title(rec.get("title"))
        if require_title and not title:
            continue
        row = {"item_id": pid, "title": title}
        for k in META_KEEP_FIELDS:
            if k in ("parent_asin", "title"):
                continue
            row[k] = rec.get(k)
        rows.append(row)
        n_kept += 1

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(
            f"No usable metadata loaded from {meta_file} "
            f"(seen={n_seen}, keep_filter_items={item_ids is not None})"
        )
    # Prefer first occurrence if duplicates
    meta = meta.drop_duplicates(subset=["item_id"], keep="first")
    logger.info(
        "Metadata loaded: kept=%d unique items (scanned=%d lines)",
        len(meta),
        n_seen,
    )
    return meta


def meta_for_parquet(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metadata columns so Arrow/parquet can serialize them.

    Lists/dicts and mixed-type fields (e.g. price as float or '—') become JSON strings.
    """
    out = meta.copy()
    for col in out.columns:
        if col in ("item_id", "title", "store", "main_category"):
            out[col] = out[col].astype("string")
            continue
        if col in ("average_rating", "rating_number"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        # features, description, categories, details, price, …
        def _cell(v: Any) -> str | None:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, (list, dict, tuple)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)

        out[col] = out[col].map(_cell).astype("string")
    return out


def join_interactions_with_meta(
    interactions: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Inner-join interactions with items that have usable metadata.

    Returns (filtered_interactions, kept_meta, join_stats).
    """
    inter_items = set(interactions["item_id"].unique())
    meta_items = set(meta["item_id"].unique())
    missing_meta = inter_items - meta_items
    covered = inter_items & meta_items

    before_n = len(interactions)
    before_items = interactions["item_id"].nunique()
    filtered = interactions[interactions["item_id"].isin(covered)].copy()
    kept_meta = meta[meta["item_id"].isin(covered)].copy()

    after_n = len(filtered)
    stats = {
        "join_key": "parent_asin -> item_id",
        "items_in_interactions": before_items,
        "items_with_meta": len(covered),
        "items_missing_meta_dropped": len(missing_meta),
        "interactions_before_meta_join": before_n,
        "interactions_after_meta_join": after_n,
        "interactions_dropped_missing_meta": before_n - after_n,
        "meta_coverage_items": float(len(covered) / before_items) if before_items else 0.0,
    }
    if stats["meta_coverage_items"] < 0.05:
        raise RuntimeError(
            f"Metadata coverage near zero ({stats['meta_coverage_items']:.2%}). "
            "Check join key / meta file category match."
        )
    logger.info(
        "Meta join: items %d→%d (missing=%d), interactions %d→%d",
        before_items,
        len(covered),
        len(missing_meta),
        before_n,
        after_n,
    )
    return filtered, kept_meta, stats
