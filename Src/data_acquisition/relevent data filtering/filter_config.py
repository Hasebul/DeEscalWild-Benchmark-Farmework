"""
filter_config.py — Centralized thresholds and rules for the data filtering pipeline.

Modify values here to tune filtering behaviour without touching any logic.
"""

# ── Input / Output ────────────────────────────────────────────────────────────
INPUT_CSV          = "./merge-output/all_videos_with_features.csv"   # transcript + features joined
OUTPUT_DIR         = "./filter-output/"
ACCEPTED_FILENAME  = "accepted_videos.csv"
REJECTED_FILENAME  = "rejected_videos.csv"

# ── Stage 1: Context Filter flags ─────────────────────────────────────────────
# Any row where ONE OR MORE of these feature columns equals 1 is immediately rejected.
CONTEXT_EXCLUSION_FLAGS: list[str] = [
    "no_police_presence",
    "no_conversation_detected",
    "advertisement_content_detected",
    "training_range_context",
    "non_relevant_crime_only_context",
]

# ── Stage 2: Score-based thresholds ───────────────────────────────────────────
# A row must satisfy ALL of the conditions below to be accepted.
# Format:  column_name -> minimum required value  (inclusive)
SCORE_THRESHOLDS: dict[str, int] = {
    "police_presence_signals_score":   2,   # at least 2/6 police signals present
    "interaction_type_signals_score":  2,   # at least 2/6 interaction signals
    "deescalation_indicators_score":   1,   # at least 1 de-escalation indicator
}

# Optional: maximum escalation score allowed (set to None to disable)
MAX_ESCALATION_SCORE: int | None = None     # e.g. set to 5 to exclude very violent content
