"""
main.py — Entry point for the YouTube Speaker Diarization Pipeline.

Pipeline stages (per video)
----------------------------
1. SKIP CHECK  — skip already-processed or permanently-ignored videos.
2. DURATION    — skip videos exceeding the configured max duration.
3. DOWNLOAD    — fetch audio from YouTube as MP3.
4. TRANSCRIBE  — send audio to Gemini; receive structured diarization.
5. EXPORT      — save TXT + CSV outputs and append to the run log.
6. CLEANUP     — delete the temp MP3 file.
7. STATUS      — mark the video as processed in the JSON status file.

Usage
-----
    python main.py                     # process all unprocessed videos
    python main.py --model gemini-2.5-pro
    python main.py --dry-run           # validate manifest, print plan, do nothing
    python main.py --video-id abc123   # process a single video by ID
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
from tqdm import tqdm

import config
import status_tracker
from auth        import build_client
from downloader  import delete_audio, download_audio
from exporter    import append_to_log, save_csv, save_txt
from logger      import get_logger
from transcriber import transcribe_audio

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube speaker diarization pipeline using Gemini."
    )
    parser.add_argument(
        "--model",
        default=config.Model.DEFAULT,
        choices=[
            config.Model.GEMINI_2_0_FLASH,
            config.Model.GEMINI_2_5_FLASH,
            config.Model.GEMINI_2_5_PRO,
        ],
        help="Gemini model to use for transcription.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and print a processing plan without running anything.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        metavar="ID",
        help="Process a single video by its ID (must exist in the manifest).",
    )
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_manifest() -> pd.DataFrame:
    """Read the video manifest CSV; exit cleanly on failure."""
    try:
        df = pd.read_csv(config.MANIFEST_CSV)
        df["id"] = df["id"].astype(str)
        log.info("Manifest loaded — %d videos total.", len(df))
        return df
    except FileNotFoundError:
        log.error("Manifest not found: %s", config.MANIFEST_CSV)
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to read manifest: %s", exc)
        sys.exit(1)


def filter_manifest(df: pd.DataFrame, single_id: str | None) -> pd.DataFrame:
    """Return only the rows that still need processing."""
    if single_id:
        subset = df[df["id"] == single_id]
        if subset.empty:
            log.error("Video ID '%s' not found in manifest.", single_id)
            sys.exit(1)
        return subset

    skip = status_tracker.ids_to_skip()
    remaining = df[~df["id"].isin(skip)]
    log.info(
        "Status: %s  |  Remaining: %d",
        status_tracker.summary(),
        len(remaining),
    )
    return remaining


# ── Per-video processing ───────────────────────────────────────────────────────

def process_video(row: pd.Series, model: str, client) -> None:
    """Run the full pipeline for one video row."""
    video_id = row["id"]
    url       = row["URL"]
    duration  = float(row.get("minutes", 0))

    # ── Duration guard ─────────────────────────────────────────────────────
    if duration > config.MAX_VIDEO_DURATION_MIN:
        log.warning(
            "Skipping ID=%s — duration %.0f min exceeds limit of %d min.",
            video_id, duration, config.MAX_VIDEO_DURATION_MIN,
        )
        append_to_log(video_id, {}, 0.0, model)
        status_tracker.mark_ignored(video_id)
        return

    audio_path: str | None = None

    try:
        # ── Download ───────────────────────────────────────────────────────
        audio_path = download_audio(url, video_id)
        if not audio_path:
            log.error("Download failed for ID=%s — skipping.", video_id)
            return

        # ── Transcribe ─────────────────────────────────────────────────────
        start_time = time.time()
        transcription, token_usage = transcribe_audio(audio_path, client, model)
        elapsed = time.time() - start_time

        log.info(
            "ID=%s  tokens(in=%d out=%d think=%d)  time=%.1fs",
            video_id,
            token_usage.get("input_token",  0),
            token_usage.get("output_token", 0),
            token_usage.get("thought_token", 0),
            elapsed,
        )

        # ── Export ─────────────────────────────────────────────────────────
        save_txt(transcription, video_id)
        save_csv(transcription, video_id)
        append_to_log(video_id, token_usage, elapsed, model)

        # ── Mark done ──────────────────────────────────────────────────────
        status_tracker.mark_processed(video_id)

    finally:
        # Always clean up the temp file, even on error
        delete_audio(audio_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    df        = load_manifest()
    remaining = filter_manifest(df, args.video_id)

    if remaining.empty:
        log.info("Nothing to process — all videos are already done.")
        return

    if args.dry_run:
        log.info("DRY RUN — would process %d video(s):", len(remaining))
        for _, row in remaining.iterrows():
            log.info("  ID=%-20s  URL=%s", row["id"], row["URL"])
        return

    # Authenticate once and reuse the client for the entire run
    client = build_client()

    log.info("Starting pipeline — %d video(s) to process.", len(remaining))

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Diarizing"):
        video_id = row["id"]
        try:
            process_video(row, model=args.model, client=client)

        except KeyboardInterrupt:
            log.warning("Interrupted by user at ID=%s. Exiting.", video_id)
            sys.exit(0)

        except Exception as exc:
            log.error("Unhandled error for ID=%s: %s — skipping.", video_id, exc)
            continue

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()