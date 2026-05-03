"""
status_tracker.py — Persists processing state to a JSON file for crash-safe resume.

JSON schema
-----------
{
    "processed": ["id1", "id2", ...],   // successfully completed
    "ignore":    ["id3", ...]           // permanently skipped (too long, etc.)
}
"""

from __future__ import annotations

import json
import os

import config
from logger import get_logger

log = get_logger(__name__)

# ── Internal type alias ───────────────────────────────────────────────────────
_StatusDict = dict[str, list[str]]

_EMPTY: _StatusDict = {"processed": [], "ignore": []}


# ── Public API ────────────────────────────────────────────────────────────────

def load() -> _StatusDict:
    """Return the current status dict, or an empty default if the file is missing."""
    if not os.path.exists(config.STATUS_FILE):
        return {"processed": [], "ignore": []}

    try:
        with open(config.STATUS_FILE, "r", encoding="utf-8") as f:
            data: _StatusDict = json.load(f)
        data.setdefault("processed", [])
        data.setdefault("ignore", [])
        return data
    except Exception as exc:
        log.warning("Could not read status file — starting fresh. Error: %s", exc)
        return {"processed": [], "ignore": []}


def mark_processed(video_id: str) -> None:
    """Add *video_id* to the 'processed' list (idempotent)."""
    _add_to_list("processed", video_id)


def mark_ignored(video_id: str) -> None:
    """Add *video_id* to the 'ignore' list (idempotent)."""
    _add_to_list("ignore", video_id)


def ids_to_skip() -> set[str]:
    """Return the union of all processed and ignored IDs."""
    data = load()
    return set(data["processed"]) | set(data["ignore"])


def summary() -> str:
    data = load()
    return (
        f"processed={len(data['processed'])}  "
        f"ignored={len(data['ignore'])}"
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _add_to_list(key: str, video_id: str) -> None:
    data = load()
    if video_id not in data[key]:
        data[key].append(video_id)
        _save(data)


def _save(data: _StatusDict) -> None:
    os.makedirs(os.path.dirname(config.STATUS_FILE), exist_ok=True)
    try:
        with open(config.STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as exc:
        log.error("Failed to save status file: %s", exc)