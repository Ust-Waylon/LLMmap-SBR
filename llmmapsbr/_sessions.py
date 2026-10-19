"""First-day sessions and the paper's data filtering and split protocol."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _count_stats(
    df: pd.DataFrame,
    name: str,
    session_col: str = "session_id",
) -> dict[str, Any]:
    if df.empty:
        return {
            "step": name,
            "n_interactions": 0,
            "n_users": 0,
            "n_sessions": 0,
            "n_items": 0,
            "avg_session_len": 0.0,
            "median_session_len": 0.0,
        }
    lens = df.groupby(session_col).size()
    return {
        "step": name,
        "n_interactions": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_sessions": int(df[session_col].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "avg_session_len": float(lens.mean()),
        "median_session_len": float(lens.median()),
    }


def assign_sbr_sessions(
    df: pd.DataFrame,
    days_per_session: int = 1,
    only_first_day: bool = True,
) -> pd.DataFrame:
    """
    Form true SBR sessions from raw interactions.

    Calendar-day sessionization:
      - floor timestamps to calendar days
      - day_diff = days since each user's first interaction day
      - if only_first_day: keep only day_diff == 0 (first active day per user)
      - session_group = day_diff // days_per_session
      - session_id = unique (user_id, session_group)

    With only_first_day=True and days_per_session=1, each user contributes at most
    one session: all interactions on their first active day.
    """
    out = df.copy()
    out = out.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    ts = out["timestamp"].astype("int64")
    # Amazon 2023 uses ms; Amazon Reviews 2023 timestamps are in milliseconds
    if ts.mean() > 1e11:
        dt = pd.to_datetime(ts, unit="ms")
    else:
        dt = pd.to_datetime(ts, unit="s")
    out["date"] = dt.dt.floor("D")

    user_min_date = out.groupby("user_id")["date"].transform("min")
    out["day_diff"] = (out["date"] - user_min_date).dt.days.astype(int)

    if only_first_day:
        before = len(out)
        out = out[out["day_diff"] == 0].copy()
        logger.info(
            "only_first_day: kept first active day only — interactions %d → %d (%.1f%%)",
            before,
            len(out),
            100.0 * len(out) / max(before, 1),
        )

    out["session_group"] = (out["day_diff"] // int(days_per_session)).astype(int)
    out["session_id"] = out.groupby(["user_id", "session_group"], sort=False).ngroup()

    logger.info(
        "Assigned sessions: users=%d sessions=%d days_per_session=%d only_first_day=%s",
        out["user_id"].nunique(),
        out["session_id"].nunique(),
        days_per_session,
        only_first_day,
    )
    return out.reset_index(drop=True)


def remove_consecutive_duplicates(
    df: pd.DataFrame,
    group_col: str = "session_id",
) -> pd.DataFrame:
    """Drop consecutive same item_id within each session (chronological)."""
    df = df.sort_values([group_col, "timestamp"], kind="mergesort").reset_index(drop=True)
    same_group = df[group_col] == df[group_col].shift(1)
    same_item = df["item_id"] == df["item_id"].shift(1)
    keep = ~(same_group & same_item)
    out = df.loc[keep].reset_index(drop=True)
    logger.info("Consecutive dedup (by %s): %d → %d", group_col, len(df), len(out))
    return out


def filter_short_sessions(
    df: pd.DataFrame,
    min_session_len: int,
    session_col: str = "session_id",
) -> pd.DataFrame:
    lens = df.groupby(session_col).size()
    keep = lens[lens >= min_session_len].index
    out = df[df[session_col].isin(keep)].reset_index(drop=True)
    logger.info(
        "Filter short sessions (<%d): sessions %d→%d, inter %d→%d",
        min_session_len,
        df[session_col].nunique(),
        out[session_col].nunique(),
        len(df),
        len(out),
    )
    return out


def filter_infrequent_items(df: pd.DataFrame, min_item_freq: int) -> pd.DataFrame:
    counts = df["item_id"].value_counts()
    keep_items = counts[counts >= min_item_freq].index
    out = df[df["item_id"].isin(keep_items)].reset_index(drop=True)
    logger.info(
        "Filter rare items (<%d): items %d→%d, inter %d→%d",
        min_item_freq,
        df["item_id"].nunique(),
        out["item_id"].nunique(),
        len(df),
        len(out),
    )
    return out


def iterative_sbr_filter(
    df: pd.DataFrame,
    min_session_len: int = 2,
    min_item_freq: int = 5,
    remove_consecutive_dups: bool = True,
    max_iters: int = 20,
    session_col: str = "session_id",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Iteratively filter rare items and short *sessions* until stable.

    Requires `session_id` already assigned (true SBR sessions).
    The protocol uses minimum session length 2 and item frequency 5.
    """
    if session_col not in df.columns:
        raise ValueError(f"DataFrame must have '{session_col}' from assign_sbr_sessions")

    funnel: list[dict[str, Any]] = [_count_stats(df, "input", session_col)]
    out = df.sort_values([session_col, "timestamp"], kind="mergesort").reset_index(drop=True)

    if remove_consecutive_dups:
        out = remove_consecutive_duplicates(out, group_col=session_col)
        funnel.append(_count_stats(out, "after_consecutive_dedup", session_col))

    for it in range(1, max_iters + 1):
        before = (len(out), out[session_col].nunique(), out["item_id"].nunique())
        out = filter_infrequent_items(out, min_item_freq)
        out = filter_short_sessions(out, min_session_len, session_col=session_col)
        after = (len(out), out[session_col].nunique(), out["item_id"].nunique())
        funnel.append(_count_stats(out, f"iter_{it}", session_col))
        if after == before:
            logger.info("SBR filter converged at iteration %d", it)
            break
    else:
        logger.warning("SBR filter hit max_iters=%d without full convergence", max_iters)

    if out.empty:
        raise RuntimeError(
            "SBR filter removed all data; loosen min_session_len / min_item_freq"
        )
    funnel.append(_count_stats(out, "final", session_col))
    return out.reset_index(drop=True), funnel


def build_session_sequences(
    df: pd.DataFrame,
    max_seq_len: int | None = 20,
    truncate_from: str = "head",
) -> pd.DataFrame:
    """
    One chronological sequence per session_id.

    The protocol keeps seq[:20] (earliest items in the session).
    Use truncate_from='head' to match; 'tail' keeps most recent.
    """
    if "session_id" not in df.columns:
        raise ValueError("session_id required — run assign_sbr_sessions first")

    df = df.sort_values(["session_id", "timestamp"], kind="mergesort")
    rows = []
    for session_id, g in df.groupby("session_id", sort=False):
        items = g["item_id"].tolist()
        ts = g["timestamp"].tolist()
        user_id = g["user_id"].iloc[0]
        if max_seq_len is not None and len(items) > max_seq_len:
            if truncate_from == "tail":
                items = items[-max_seq_len:]
                ts = ts[-max_seq_len:]
            else:
                items = items[:max_seq_len]
                ts = ts[:max_seq_len]
        rows.append(
            {
                "user_id": user_id,
                "session_id": int(session_id) if not isinstance(session_id, str) else session_id,
                "item_id_list": items,
                "timestamp_list": ts,
                "session_time": int(ts[-1]) if ts else 0,  # max time — for chrono split
                "seq_len": len(items),
            }
        )
    seq = pd.DataFrame(rows)
    logger.info(
        "Built session sequences: n=%d avg_len=%.2f median_len=%.1f (truncate=%s max=%s)",
        len(seq),
        seq["seq_len"].mean(),
        seq["seq_len"].median(),
        truncate_from,
        max_seq_len,
    )
    return seq


def chronological_session_split(
    sequences: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    min_session_len: int = 2,
    filter_to_train_items: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Chronological split over sessions.

    Default: 8:1:1 via ``train_ratio`` / ``valid_ratio`` of all sessions.

    Sessions sorted by session_time (max timestamp in session).
    Within each session used for next-item eval:
      history = items[:-1], target = items[-1]
    Train store: full session item_id_list (no prefix expand offline).

    If filter_to_train_items: valid/test drop items not seen in train and
    re-filter sessions that become too short (after chronological splitting).
    """
    seq = sequences.copy()
    if "session_time" not in seq.columns:
        # fallback: last timestamp in list
        seq["session_time"] = seq["timestamp_list"].apply(
            lambda xs: int(xs[-1]) if isinstance(xs, list) and xs else 0
        )
    seq = seq.sort_values("session_time", kind="mergesort").reset_index(drop=True)

    n = len(seq)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = seq.iloc[:train_end].copy()
    valid_df = seq.iloc[train_end:valid_end].copy()
    test_df = seq.iloc[valid_end:].copy()

    train_items: set[str] | None = None
    if filter_to_train_items:
        train_items = set()
        for items in train_df["item_id_list"]:
            train_items.update(items)

        def _filter_seq(items: list[str]) -> list[str]:
            assert train_items is not None
            return [x for x in items if x in train_items]

        valid_df = valid_df.copy()
        test_df = test_df.copy()
        valid_df["item_id_list"] = valid_df["item_id_list"].apply(_filter_seq)
        test_df["item_id_list"] = test_df["item_id_list"].apply(_filter_seq)
        valid_df = valid_df[valid_df["item_id_list"].apply(len) >= min_session_len]
        test_df = test_df[test_df["item_id_list"].apply(len) >= min_session_len]
        # also ensure train sessions still long enough (should already)
        train_df = train_df[train_df["item_id_list"].apply(len) >= min_session_len]

    def _train_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for rec in df.itertuples(index=False):
            items = list(rec.item_id_list)
            if len(items) < min_session_len:
                continue
            rows.append(
                {
                    "user_id": rec.user_id,
                    "session_id": rec.session_id,
                    "item_id_list": items,
                    "seq_len": len(items),
                    "session_time": int(getattr(rec, "session_time", 0)),
                }
            )
        return rows

    def _eval_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Leave-last-out within session for next-item evaluation."""
        rows = []
        for rec in df.itertuples(index=False):
            items = list(rec.item_id_list)
            if len(items) < min_session_len:
                continue
            # need at least 2 items: history + target
            if len(items) < 2:
                continue
            rows.append(
                {
                    "user_id": rec.user_id,
                    "session_id": rec.session_id,
                    "history": items[:-1],
                    "target": items[-1],
                    "session_time": int(getattr(rec, "session_time", 0)),
                }
            )
        return rows

    train_rows = _train_rows(train_df)
    valid_rows = _eval_rows(valid_df)
    test_rows = _eval_rows(test_df)

    split_desc = f"{train_ratio:g}/{valid_ratio:g}/{1.0 - train_ratio - valid_ratio:g}"
    logger.info(
        "Chronological session split %s: train=%d valid=%d test=%d "
        "(filter_to_train_items=%s, train_items=%s)",
        split_desc,
        len(train_rows),
        len(valid_rows),
        len(test_rows),
        filter_to_train_items,
        len(train_items) if train_items is not None else "n/a",
    )
    return {
        "train": train_rows,
        "valid": valid_rows,
        "test": test_rows,
        "train_item_set": train_items,
    }
