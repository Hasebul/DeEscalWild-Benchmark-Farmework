"""
config.py — Centralized configuration for the YouTube scraper pipeline.
"""

# ── Input / Output ────────────────────────────────────────────────────────────
PLAYLIST_CSV        = "Input/Source of de-escalation youtube video playlist.csv"
VIDEO_LIST_DIR      = "output-video-list/"
MERGED_DIR          = "merge-output/"
MERGED_FILENAME     = "video-list.csv"
TRANSCRIPT_FILENAME = "video-list-with-transcripts.csv"

# ── Transcript languages (in priority order) ──────────────────────────────────
TRANSCRIPT_LANGUAGES        = ["en"]
TRANSCRIPT_LANGUAGES_AUTO   = ["en-auto"]

# ── Scraping behaviour ────────────────────────────────────────────────────────
REQUEST_DELAY_SEC   = 1        # seconds to sleep between requests
DESCRIPTION_LIMIT   = 500      # characters to keep from video description

# ── yt-dlp options ────────────────────────────────────────────────────────────
YDL_OPTS: dict = {
    "quiet":        True,
    "extract_flat": False,
    "ignoreerrors": True,
}

# ── CSV column order ──────────────────────────────────────────────────────────
CSV_FIELDNAMES = [
    "Index",
    "Title",
    "URL",
    "Uploader",
    "Views",
    "Upload Date",
    "Duration (sec)",
    "Description",
    "Transcript",
]
