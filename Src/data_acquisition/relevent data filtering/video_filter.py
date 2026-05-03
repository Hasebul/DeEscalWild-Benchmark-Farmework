"""
video_filter.py — Two-stage filtering pipeline for LLM-annotated video transcripts.

Pipeline
--------
Stage 1 – Context filter  : reject rows where any exclusion flag == 1.
Stage 2 – Score filter     : reject rows that fall below minimum group scores.

Each rejected row receives a human-readable ``rejection_reason`` column that
explains exactly which rule(s) it failed so the decision is fully auditable.
"""

from __future__ import annotations

import pandas as pd

import filter_config as cfg
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def run_filter_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply both filtering stages to *df*.

    Parameters
    ----------
    df:
        DataFrame containing the merged transcript + feature columns.

    Returns
    -------
    accepted : pd.DataFrame
        Rows that passed every filter rule.
    rejected : pd.DataFrame
        Rows that failed at least one rule, with a ``rejection_reason`` column.
    """
    df = df.copy()
    df["rejection_reason"] = ""          # will stay empty for accepted rows

    df = _stage1_context_filter(df)
    df = _stage2_score_filter(df)

    accepted = df[df["rejection_reason"] == ""].drop(columns=["rejection_reason"])
    rejected = df[df["rejection_reason"] != ""]

    log.info(
        "Filter complete — accepted: %d  |  rejected: %d  |  total: %d",
        len(accepted), len(rejected), len(df),
    )
    return accepted, rejected


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def _stage1_context_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reject any row where at least one context-exclusion flag is set to 1.

    Already-rejected rows (rejection_reason != "") are skipped so that
    reasons from multiple stages can accumulate cleanly.
    """
    available_flags = [f for f in cfg.CONTEXT_EXCLUSION_FLAGS if f in df.columns]
    missing_flags   = set(cfg.CONTEXT_EXCLUSION_FLAGS) - set(available_flags)

    if missing_flags:
        log.warning(
            "Stage 1 – the following context flags are absent from the data and "
            "will be skipped: %s", missing_flags
        )

    if not available_flags:
        log.info("Stage 1 – no context flag columns found; skipping.")
        return df

    for idx, row in df.iterrows():
        if row["rejection_reason"]:          # already flagged
            continue

        triggered = [flag for flag in available_flags if _is_set(row, flag)]
        if triggered:
            df.at[idx, "rejection_reason"] = (
                "[Stage 1 – context filter] flags triggered: "
                + ", ".join(triggered)
            )

    stage1_rejected = (df["rejection_reason"].str.startswith("[Stage 1")).sum()
    log.info("Stage 1 complete — %d row(s) rejected by context filters.", stage1_rejected)
    return df


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def _stage2_score_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reject any row (not already rejected) that falls below a required group score.

    If ``MAX_ESCALATION_SCORE`` is set in config, rows whose escalation score
    exceeds that cap are also rejected.
    """
    available_thresholds = {
        col: minimum
        for col, minimum in cfg.SCORE_THRESHOLDS.items()
        if col in df.columns
    }
    missing_cols = set(cfg.SCORE_THRESHOLDS) - set(available_thresholds)

    if missing_cols:
        log.warning(
            "Stage 2 – the following score columns are absent and will be skipped: %s",
            missing_cols,
        )

    for idx, row in df.iterrows():
        if row["rejection_reason"]:          # already rejected in Stage 1
            continue

        failures: list[str] = []

        # Minimum score checks
        for col, minimum in available_thresholds.items():
            actual = _safe_int(row.get(col))
            if actual < minimum:
                failures.append(f"{col}={actual} (min {minimum})")

        # Optional escalation cap
        esc_col = "escalation_indicators_score"
        if cfg.MAX_ESCALATION_SCORE is not None and esc_col in df.columns:
            actual = _safe_int(row.get(esc_col))
            if actual > cfg.MAX_ESCALATION_SCORE:
                failures.append(
                    f"{esc_col}={actual} (max {cfg.MAX_ESCALATION_SCORE})"
                )

        if failures:
            df.at[idx, "rejection_reason"] = (
                "[Stage 2 – score filter] below threshold: "
                + "; ".join(failures)
            )

    stage2_rejected = (df["rejection_reason"].str.startswith("[Stage 2")).sum()
    log.info("Stage 2 complete — %d row(s) rejected by score filters.", stage2_rejected)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_set(row: pd.Series, column: str) -> bool:
    """Return True when *column* exists in *row* and its value equals 1."""
    val = row.get(column, 0)
    try:
        return int(val) == 1
    except (ValueError, TypeError):
        return False


def _safe_int(value) -> int:
    """Convert *value* to int safely; return 0 on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
