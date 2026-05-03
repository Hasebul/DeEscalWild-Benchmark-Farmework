from __future__ import annotations
 
import os
from datetime import timedelta
from dataclasses import dataclass, field
 
 
# ── API / Authentication ───────────────────────────────────────────────────────
# Set these in your shell or .env file before running:
#   export GOOGLE_API_KEY="your_key_here"
#   export GOOGLE_CLOUD_PROJECT="your_project"   # if using Vertex AI
#   export GOOGLE_CLOUD_LOCATION="us-central1"   # if using Vertex AI
#   export GOOGLE_GENAI_USE_VERTEXAI="false"      # "true" to switch to Vertex AI
 
GOOGLE_API_KEY            = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CLOUD_PROJECT      = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION     = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
 
 
# ── Model ─────────────────────────────────────────────────────────────────────
class Model:
    GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO   = "gemini-2.5-pro"
    DEFAULT          = GEMINI_2_5_FLASH
 
 
# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(ROOT_DIR, "data")
TEMP_DIR      = os.path.join(ROOT_DIR, "temp_downloads")
OUTPUT_DIR    = os.path.join(ROOT_DIR, "output")
TXT_DIR       = os.path.join(OUTPUT_DIR, "txt")
CSV_DIR       = os.path.join(OUTPUT_DIR, "csv")
LOG_FILE      = os.path.join(OUTPUT_DIR, "log.csv")
STATUS_FILE   = os.path.join(OUTPUT_DIR, "process_status.json")
 
# Input manifest — must have columns: id, URL, minutes
MANIFEST_CSV  = os.path.join(DATA_DIR, "youtube_video_manifest.csv")
 
 
# ── Audio download ────────────────────────────────────────────────────────────
AUDIO_FORMAT          = "mp3"
AUDIO_QUALITY         = "192"           # kbps
MAX_VIDEO_DURATION_MIN = 45             # videos longer than this are skipped
 
 
# ── Transcription ─────────────────────────────────────────────────────────────
TRANSCRIPT_TEMPERATURE   = 0.0
TRANSCRIPT_TOP_P         = 0.95
TRANSCRIPT_SEED          = 42
THINKING_BUDGET_FLASH    = 0            # disable thinking for Flash (faster)
THINKING_BUDGET_PRO      = 128
 
# Marker used when speaker attribute cannot be determined
UNKNOWN_SPEAKER_MARKER = "?"
 
 
# ── Retry policy ──────────────────────────────────────────────────────────────
RETRY_MAX_ATTEMPTS    = 7
RETRY_WAIT_START_SEC  = 10
RETRY_WAIT_INCREMENT  = 1
 
 
# ── YouTube constants ─────────────────────────────────────────────────────────
YOUTUBE_URL_PREFIX         = "https://www.youtube.com/watch?v="
CLOUD_STORAGE_URI_PREFIX   = "gs://"