"""
main.py — Entry point for the two-stage video transcript filtering pipeline.

Pipeline
--------
1. Load   — read the merged transcript + features CSV.
2. Filter — Stage 1 (context flags) then Stage 2 (score thresholds).
3. Save   — write accepted.csv, rejected.csv, and a summary report.

Usage
-----
    python main.py                          # uses paths from filter_config.py
    python main.py --input path/to/file.csv # override input path
    python main.py --dry-run               # print summary only, write no files
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import filter_config as cfg
from logger        import get_logger
from report        import print_summary, save_summary_txt
from video_filter  import run_filter_pipeline

log = get_logger(__name__)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Read the merged CSV; exit cleanly if the file is missing or unreadable."""
    if not os.path.exists(path):
        log.error("Input file not found: %s", path)
        sys.exit(1)

    try:
        # cp1252 matches the encoding used in the feature-extraction pipeline
        df = pd.read_csv(path, encoding="cp1252")
        log.info("Loaded %d rows from '%s'", len(df), path)
        return df
    except Exception as exc:
        log.error("Failed to read input file: %s", exc)
        sys.exit(1)


def save_outputs(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    output_dir: str = cfg.OUTPUT_DIR,
) -> None:
    """Write accepted and rejected DataFrames to separate CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    accepted_path = os.path.join(output_dir, cfg.ACCEPTED_FILENAME)
    rejected_path = os.path.join(output_dir, cfg.REJECTED_FILENAME)

    accepted.to_csv(accepted_path, index=False, encoding="utf-8")
    rejected.to_csv(rejected_path, index=False, encoding="utf-8")

    log.info("Accepted rows saved → %s", accepted_path)
    log.info("Rejected rows saved → %s", rejected_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage context + score filter for LLM-annotated video transcripts."
    )
    parser.add_argument(
        "--input",
        default=cfg.INPUT_CSV,
        metavar="CSV_PATH",
        help=f"Path to the merged transcript+features CSV (default: {cfg.INPUT_CSV}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary report without writing any output files.",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # 1. Load
    df = load_data(args.input)

    # 2. Filter
    accepted, rejected = run_filter_pipeline(df)

    # 3. Report
    print_summary(accepted, rejected)

    # 4. Save (unless --dry-run)
    if args.dry_run:
        log.info("Dry-run mode — no files written.")
    else:
        save_outputs(accepted, rejected)
        save_summary_txt(accepted, rejected)

    log.info("Done.")


if __name__ == "__main__":
    main()
