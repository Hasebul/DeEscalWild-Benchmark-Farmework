"""
main.py — Entry point for the YouTube de-escalation scraper pipeline.

Pipeline stages
---------------
1. FETCH    — For each channel in the input CSV, fetch all video metadata.
2. MERGE    — Combine per-channel CSV files into one unified video list.
3. TRANSCRIBE — Fetch transcripts for every video in the merged list.

Usage
-----
    python main.py                        # run all three stages
    python main.py --stage fetch          # only stage 1
    python main.py --stage merge          # only stage 2
    python main.py --stage transcribe     # only stage 3
    python main.py --stage transcribe --resume 142   # resume from row 142
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import config
from channel_fetcher  import fetch_channel_videos
from csv_merger       import merge_channel_csvs
from logger           import get_logger
from transcript_fetcher import fetch_transcripts_for_csv

log = get_logger(__name__)


# ── Stage functions ───────────────────────────────────────────────────────────

def stage_fetch(playlist_csv: str = config.PLAYLIST_CSV) -> None:
    """Stage 1 – fetch video metadata for every channel in *playlist_csv*."""
    log.info("═══ STAGE 1: FETCH CHANNEL VIDEOS ═══")

    try:
        df = pd.read_csv(playlist_csv)
    except FileNotFoundError:
        log.error("Playlist CSV not found: %s", playlist_csv)
        sys.exit(1)

    # Rows 1 … n-2 per the original script (skip header & last sentinel row)
    for row_idx in range(1, len(df) - 1):
        raw_link = df.loc[row_idx, "link"]

        # Guard against malformed rows
        if "@" not in raw_link:
            log.warning("Row %d – no '@' in link '%s'; skipping.", row_idx, raw_link)
            continue

        channel_handle = raw_link.split("@")[1]
        log.info("── Channel %d / %d : @%s", row_idx, len(df) - 2, channel_handle)
        fetch_channel_videos(channel_handle)

    log.info("Stage 1 complete.\n")


def stage_merge() -> str:
    """Stage 2 – merge all per-channel CSVs into one file."""
    log.info("═══ STAGE 2: MERGE CSV FILES ═══")
    output_path = merge_channel_csvs()
    log.info("Stage 2 complete.\n")
    return output_path


def stage_transcribe(start_index: int = 0) -> None:
    """Stage 3 – enrich the merged CSV with transcripts."""
    log.info("═══ STAGE 3: FETCH TRANSCRIPTS ═══")

    input_csv  = os.path.join(config.MERGED_DIR, config.MERGED_FILENAME)
    output_csv = os.path.join(config.MERGED_DIR, config.TRANSCRIPT_FILENAME)

    if not os.path.exists(input_csv):
        log.error(
            "Merged CSV not found at '%s'. Run the merge stage first.", input_csv
        )
        sys.exit(1)

    fetch_transcripts_for_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        start_index=start_index,
    )
    log.info("Stage 3 complete.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube channel scraper – fetch metadata and transcripts."
    )
    parser.add_argument(
        "--stage",
        choices=["fetch", "merge", "transcribe", "all"],
        default="all",
        help="Which pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="ROW",
        help="Row index to resume transcript fetching from (stage 'transcribe' only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.stage in ("fetch", "all"):
        stage_fetch()

    if args.stage in ("merge", "all"):
        stage_merge()

    if args.stage in ("transcribe", "all"):
        stage_transcribe(start_index=args.resume)

    log.info("Pipeline finished.")


if __name__ == "__main__":
    main()
