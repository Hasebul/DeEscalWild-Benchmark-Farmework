"""
csv_merger.py — Merges all per-channel CSVs into a single video-list CSV.

Responsibilities
----------------
* Scan the video-list directory for ``*.csv`` files.
* Tag each row with its source channel name (derived from the filename).
* Concatenate all frames, assign a global integer ``id``, and save.
"""

from __future__ import annotations

import os

import pandas as pd

import config
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def merge_channel_csvs(
    input_dir: str = config.VIDEO_LIST_DIR,
    output_dir: str = config.MERGED_DIR,
    output_filename: str = config.MERGED_FILENAME,
) -> str:
    """
    Merge every ``*.csv`` in *input_dir* into one file at *output_dir/output_filename*.

    Each row receives a ``channel_name`` column populated from the filename prefix
    (the part before the first underscore).  A global ``id`` column (0-based) is
    prepended to the final dataframe.

    Parameters
    ----------
    input_dir:       Directory containing per-channel CSV files.
    output_dir:      Directory where the merged file will be written.
    output_filename: Name of the merged output file.

    Returns
    -------
    str
        Absolute path to the merged CSV.
    """
    csv_files = _list_csv_files(input_dir)
    if not csv_files:
        log.warning("No CSV files found in '%s'. Nothing to merge.", input_dir)
        return ""

    log.info("Found %d CSV file(s) to merge.", len(csv_files))

    frames = _load_and_tag_frames(csv_files, input_dir)
    if not frames:
        log.error("No data was successfully read. Aborting merge.")
        return ""

    merged_df  = _concatenate_frames(frames)
    output_path = _save_merged(merged_df, output_dir, output_filename)
    return output_path


# ── Private helpers ───────────────────────────────────────────────────────────

def _list_csv_files(directory: str) -> list[str]:
    """Return a sorted list of ``*.csv`` filenames inside *directory*."""
    try:
        return sorted(f for f in os.listdir(directory) if f.endswith(".csv"))
    except FileNotFoundError:
        log.error("Input directory not found: '%s'", directory)
        return []


def _load_and_tag_frames(filenames: list[str], directory: str) -> list[pd.DataFrame]:
    """Read each CSV, add a ``channel_name`` column, and return the list of frames."""
    frames: list[pd.DataFrame] = []

    for filename in filenames:
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                log.warning("'%s' is empty – skipped.", filename)
                continue

            df["channel_name"] = filename.split("_")[0]   # e.g. "GoogleDevelopers"
            frames.append(df)
            log.info("  ✓ Read '%s'  (%d rows)", filename, len(df))

        except pd.errors.EmptyDataError:
            log.warning("'%s' is empty – skipped.", filename)
        except Exception as exc:
            log.error("Error reading '%s': %s – skipped.", filename, exc)

    return frames


def _concatenate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames and prepend a global ``id`` column."""
    merged = pd.concat(frames, ignore_index=True)

    # Build a clean global integer id and move it to the first column
    merged.insert(0, "id", merged.index)

    log.info("Merged %d rows across %d file(s).", len(merged), len(frames))
    return merged


def _save_merged(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Write *df* to *output_dir/filename* and return the full path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        df.to_csv(output_path, index=False)
        log.info("Merged CSV saved → %s", output_path)
    except Exception as exc:
        log.error("Failed to save merged CSV: %s", exc)

    return output_path
