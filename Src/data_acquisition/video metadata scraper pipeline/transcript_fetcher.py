"""
transcript_fetcher.py — Fetches YouTube transcripts for every row in the merged CSV.

Responsibilities
----------------
* Iterate over every video URL in the merged video-list CSV.
* Attempt English transcript → fall back to auto-generated → record failure reason.
* Checkpoint after every video so progress survives interruptions.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeRequestFailed,
)

import config
from logger import get_logger

log = get_logger(__name__)

# Module-level singleton so callers don't recreate it on every call.
_api = YouTubeTranscriptApi()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_transcripts_for_csv(
    input_csv: str,
    output_csv: str,
    start_index: int = 0,
) -> None:
    """
    Read *input_csv*, populate the ``Transcript`` column, and save to *output_csv*.

    The function checkpoints after every video so you can resume from
    *start_index* if the process is interrupted.

    Parameters
    ----------
    input_csv:
        Path to the merged video-list CSV (must contain a ``URL`` column).
    output_csv:
        Destination path for the enriched CSV.
    start_index:
        Row index (0-based) to resume from.  Useful after a rate-limit crash.
    """
    df = pd.read_csv(input_csv)

    log.info("Starting transcript fetch from row %d / %d …", start_index, len(df))

    for i in range(start_index, len(df)):
        video_url = df.at[i, "URL"]
        video_id  = _extract_video_id(video_url)

        if not video_id:
            log.warning("Row %d – cannot parse video ID from URL: %s", i, video_url)
            df.at[i, "Transcript"] = "[Invalid URL – could not extract video ID]"
            continue

        transcript_text = _get_transcript(video_id, row=i)
        df.at[i, "Transcript"] = transcript_text

        # Checkpoint after every row
        df.to_csv(output_csv, index=False)
        time.sleep(config.REQUEST_DELAY_SEC)

    log.info("Transcript fetch complete → %s", output_csv)


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_video_id(url: str) -> Optional[str]:
    """Return the YouTube video ID from a watch URL, or ``None`` if not found."""
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return None


def _get_transcript(video_id: str, row: int = -1) -> str:
    """
    Try to fetch an English transcript for *video_id*.

    Falls back to an auto-generated transcript if a manual one is unavailable.
    Returns a human-readable error string on failure (never raises).
    """
    # 1️⃣  Manual English transcript
    try:
        transcript = _api.fetch(video_id=video_id, languages=config.TRANSCRIPT_LANGUAGES)
        return " ".join(s.text for s in transcript.snippets)
    except TranscriptsDisabled:
        log.debug("Row %d (%s) – transcripts disabled.", row, video_id)
        return "[Transcript disabled by uploader]"
    except NoTranscriptFound:
        pass    # fall through to auto-generated attempt below
    except YouTubeRequestFailed as exc:
        log.warning("Row %d (%s) – request failed: %s", row, video_id, exc)
        return f"[Rate limit or request failed: {exc}]"
    except Exception as exc:
        log.warning("Row %d (%s) – unexpected error: %s", row, video_id, exc)
        return f"[Error fetching transcript: {exc}]"

    # 2️⃣  Auto-generated transcript fallback
    try:
        transcript = _api.fetch(video_id=video_id, languages=config.TRANSCRIPT_LANGUAGES_AUTO)
        log.debug("Row %d (%s) – using auto-generated transcript.", row, video_id)
        return " ".join(s.text for s in transcript.snippets)
    except Exception:
        return "[No transcript available – manual or automatic]"
