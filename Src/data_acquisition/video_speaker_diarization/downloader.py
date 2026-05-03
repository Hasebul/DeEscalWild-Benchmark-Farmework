"""
downloader.py — Downloads YouTube audio as MP3 using yt-dlp.

Responsibilities
----------------
* Accept a YouTube URL and a file ID.
* Download best-quality audio and convert to MP3 via FFmpeg.
* Return the local path to the downloaded file, or None on failure.
* Clean up temp files on request.
"""

from __future__ import annotations

import os

import yt_dlp

import config
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def download_audio(youtube_url: str, file_id: str) -> str | None:
    """
    Download audio from *youtube_url* and save it as ``{file_id}.mp3``
    inside ``config.TEMP_DIR``.

    Parameters
    ----------
    youtube_url:
        Full YouTube watch URL.
    file_id:
        Unique identifier used as the output filename stem.

    Returns
    -------
    str | None
        Absolute path to the downloaded MP3, or ``None`` on failure.
    """
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    output_template = os.path.join(config.TEMP_DIR, f"{file_id}.%(ext)s")
    expected_path   = os.path.join(config.TEMP_DIR, f"{file_id}.{config.AUDIO_FORMAT}")

    ydl_opts = {
        "format":   "bestaudio/best",
        "outtmpl":  output_template,
        "quiet":    True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key":              "FFmpegExtractAudio",
                "preferredcodec":   config.AUDIO_FORMAT,
                "preferredquality": config.AUDIO_QUALITY,
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            log.info("Downloading audio for ID=%s  url=%s", file_id, youtube_url)
            ydl.download([youtube_url])
        return expected_path

    except Exception as exc:
        log.error("Download failed for ID=%s: %s", file_id, exc)
        return None


def delete_audio(file_path: str | None) -> None:
    """Remove *file_path* from disk if it exists."""
    if not file_path:
        return
    try:
        os.remove(file_path)
        log.debug("Deleted temp file: %s", file_path)
    except OSError as exc:
        log.warning("Could not delete %s: %s", file_path, exc)