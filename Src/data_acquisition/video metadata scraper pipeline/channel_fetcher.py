"""
channel_fetcher.py — Fetches video metadata for every video on a YouTube channel.

Responsibilities
----------------
* Accept a channel handle/URL and return a list of video-detail dicts.
* Gracefully skip restricted, deleted, or members-only videos.
* Write results to a per-channel CSV file.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

import yt_dlp
from yt_dlp.utils import DownloadError

import config
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_channel_videos(channel_handle: str) -> str:
    """
    Fetch metadata for every public video on *channel_handle* and write
    the results to ``config.VIDEO_LIST_DIR/<channel_handle>_metadata_and_transcripts.csv``.

    Parameters
    ----------
    channel_handle:
        Raw YouTube handle (without the leading ``@``), e.g. ``"GoogleDevelopers"``.

    Returns
    -------
    str
        Path to the output CSV file.
    """
    channel_url = f"https://www.youtube.com/@{channel_handle}/videos"
    output_file = os.path.join(
        config.VIDEO_LIST_DIR, f"{channel_handle}_metadata_and_transcripts.csv"
    )

    os.makedirs(config.VIDEO_LIST_DIR, exist_ok=True)

    log.info("Fetching channel metadata for @%s …", channel_handle)
    channel_info = _extract_channel_info(channel_url)
    if not channel_info:
        log.error("Aborting – could not retrieve info for @%s.", channel_handle)
        return output_file

    channel_title  = channel_info.get("title", "Unknown Channel")
    video_entries  = [e for e in channel_info.get("entries", []) if e]

    log.info("Channel : %s", channel_title)
    log.info("Videos  : %d (after filtering failed entries)", len(video_entries))

    _write_videos_to_csv(video_entries, output_file)
    log.info("Saved metadata → %s\n", output_file)
    return output_file


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_channel_info(channel_url: str) -> Optional[dict]:
    """Run yt-dlp against *channel_url* and return the info dict, or ``None`` on failure."""
    try:
        with yt_dlp.YoutubeDL(config.YDL_OPTS) as ydl:
            return ydl.extract_info(channel_url, download=False)
    except Exception as exc:
        log.error("Critical error fetching channel info: %s", exc)
        return None


def _fetch_video_details(video_url: str) -> Optional[dict]:
    """Return full metadata dict for a single *video_url*, or ``None`` on failure."""
    try:
        with yt_dlp.YoutubeDL(config.YDL_OPTS) as ydl:
            return ydl.extract_info(video_url, download=False)
    except DownloadError as exc:
        log.debug("Download error (restricted?) for %s: %s", video_url, exc)
        return None
    except Exception as exc:
        log.debug("Unexpected error fetching %s: %s", video_url, exc)
        return None


def _build_video_row(idx: int, details: dict) -> dict:
    """Map raw yt-dlp *details* to a CSV row dict."""
    description = (details.get("description") or "").replace("\n", " ")
    return {
        "Index":          idx,
        "Title":          details.get("title", "Untitled"),
        "URL":            details.get("webpage_url", ""),
        "Uploader":       details.get("uploader", "Unknown"),
        "Views":          details.get("view_count", "N/A"),
        "Upload Date":    details.get("upload_date", "N/A"),
        "Duration (sec)": details.get("duration", 0),
        "Description":    description[: config.DESCRIPTION_LIMIT] + "…",
        "Transcript":     "",           # filled in a separate pipeline stage
    }


def _build_restricted_row(idx: int, video_id: str, video_url: str) -> dict:
    """Return a placeholder row for a video we cannot access."""
    return {
        "Index":          idx,
        "Title":          f"[RESTRICTED] ID: {video_id}",
        "URL":            video_url,
        "Uploader":       "N/A",
        "Views":          "N/A",
        "Upload Date":    "N/A",
        "Duration (sec)": "N/A",
        "Description":    "Content is members-only, age-restricted, or private.",
        "Transcript":     "[Cannot access restricted content]",
    }


def _write_videos_to_csv(video_entries: list[dict], output_file: str) -> None:
    """Iterate over *video_entries*, fetch full details, and write rows to *output_file*."""
    from tqdm import tqdm  # local import so the module is usable without tqdm

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.CSV_FIELDNAMES)
        writer.writeheader()

        for idx, entry in tqdm(enumerate(video_entries, start=1), total=len(video_entries)):
            video_id  = entry.get("id", "")
            video_url = entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"

            details = _fetch_video_details(video_url)

            if not details or not details.get("title"):
                # video is restricted / deleted – write placeholder
                writer.writerow(_build_restricted_row(idx, video_id, video_url))
            else:
                writer.writerow(_build_video_row(idx, details))

            time.sleep(config.REQUEST_DELAY_SEC)
