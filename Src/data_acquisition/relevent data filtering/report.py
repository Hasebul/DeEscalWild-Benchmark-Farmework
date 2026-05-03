"""
report.py — Generates a human-readable filtering summary to stdout and optionally to disk.
"""

from __future__ import annotations

import os

import pandas as pd

import filter_config as cfg
from logger import get_logger

log = get_logger(__name__)

_DIVIDER = "=" * 60


def print_summary(accepted: pd.DataFrame, rejected: pd.DataFrame) -> None:
    """Print a concise filtering report to stdout."""
    total = len(accepted) + len(rejected)

    lines = [
        "",
        _DIVIDER,
        "  FILTERING PIPELINE — SUMMARY REPORT",
        _DIVIDER,
        f"  Total rows processed : {total}",
        f"  ✅ Accepted          : {len(accepted)}  ({_pct(len(accepted), total)}%)",
        f"  ❌ Rejected          : {len(rejected)}  ({_pct(len(rejected), total)}%)",
        "",
        "── Stage breakdown ──────────────────────────────────",
    ]

    # Count rejections per stage
    stage1 = rejected["rejection_reason"].str.startswith("[Stage 1").sum()
    stage2 = rejected["rejection_reason"].str.startswith("[Stage 2").sum()
    lines += [
        f"  Stage 1 (context flags)  : {stage1} rejected",
        f"  Stage 2 (score thresholds): {stage2} rejected",
        "",
        "── Top rejection reasons ────────────────────────────",
    ]

    # Most common individual rejection reasons
    if not rejected.empty:
        reason_counts = rejected["rejection_reason"].value_counts().head(10)
        for reason, count in reason_counts.items():
            lines.append(f"  ({count:>4}x)  {reason}")
    else:
        lines.append("  — none —")

    lines += ["", _DIVIDER, ""]
    print("\n".join(lines))


def save_summary_txt(accepted: pd.DataFrame, rejected: pd.DataFrame) -> None:
    """Write the same summary to a text file in the output directory."""
    import io, sys

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "filter_summary.txt")

    # Capture stdout temporarily
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    print_summary(accepted, rejected)
    sys.stdout = old_stdout

    with open(path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    log.info("Summary report saved → %s", path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0"
    return f"{100 * part / total:.1f}"
