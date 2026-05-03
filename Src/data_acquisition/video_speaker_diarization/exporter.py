"""
exporter.py — Saves transcription results to disk in multiple formats.

Supported formats
-----------------
* TXT  — human-readable dialogue log with timestamps and speaker labels.
* CSV  — structured rows (start_time, speaker_label, transcript) for analysis.
* Log  — per-video processing log (id, tokens, execution time, model name).
"""

from __future__ import annotations

import csv
import os

import config
from logger import get_logger
from models import VideoTranscription

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def save_txt(transcription: VideoTranscription, file_id: str) -> str:
    """
    Write a human-readable dialogue log to ``config.TXT_DIR/{file_id}.txt``.

    Each line has the format:
        [MM:SS] Speaker Label: Transcript text

    Returns the path to the written file.
    """
    os.makedirs(config.TXT_DIR, exist_ok=True)
    output_path = os.path.join(config.TXT_DIR, f"{file_id}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- Detailed Conversation Log ---\n\n")

        for segment in transcription.task1_transcripts:
            label = transcription.label_for_voice(segment.voice)
            f.write(f"[{segment.start}] {label}: {segment.text}\n")

    log.info("TXT saved → %s", output_path)
    return output_path


def save_csv(transcription: VideoTranscription, file_id: str) -> str:
    """
    Write structured transcript rows to ``config.CSV_DIR/{file_id}.csv``.

    Columns: start_time, speaker_label, transcript

    Returns the path to the written file.
    """
    os.makedirs(config.CSV_DIR, exist_ok=True)
    output_path = os.path.join(config.CSV_DIR, f"{file_id}.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start_time", "speaker_label", "transcript"])

        for segment in transcription.task1_transcripts:
            label = transcription.label_for_voice(segment.voice)
            writer.writerow([segment.start, label, segment.text])

    log.info("CSV saved → %s", output_path)
    return output_path


def append_to_log(
    file_id: str,
    token_usage: dict[str, int],
    execution_time_sec: float,
    model_name: str,
) -> None:
    """
    Append one row to the run log CSV at ``config.LOG_FILE``.

    Columns: id, input_token, output_token, thought_token,
             execution_time_sec, model_name

    The header is written only if the file is new.
    """
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    file_is_new = not os.path.exists(config.LOG_FILE)

    with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if file_is_new:
            writer.writerow(
                ["id", "input_token", "output_token", "thought_token",
                 "execution_time_sec", "model_name"]
            )

        writer.writerow([
            file_id,
            token_usage.get("input_token",  0),
            token_usage.get("output_token", 0),
            token_usage.get("thought_token", 0),
            f"{execution_time_sec:.2f}",
            model_name,
        ])