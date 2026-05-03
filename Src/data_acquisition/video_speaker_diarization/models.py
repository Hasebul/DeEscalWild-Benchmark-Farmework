"""
models.py — Pydantic schemas for structured Gemini responses.

These define exactly what the LLM must return and provide type-safe
access to transcription results throughout the pipeline.
"""

from __future__ import annotations

import pydantic

import config


class Transcript(pydantic.BaseModel):
    """A single timed speech segment attributed to one voice."""
    start: str          # timecode string, e.g. "01:23" or "1:02:45"
    text:  str          # verbatim spoken text
    voice: int          # numeric voice ID (1, 2, 3 …)


class Speaker(pydantic.BaseModel):
    """Metadata about one identified speaker."""
    voice:        int   # matches Transcript.voice
    name:         str   # real name or UNKNOWN_SPEAKER_MARKER
    company:      str
    position:     str
    role_in_video: str


class VideoTranscription(pydantic.BaseModel):
    """Top-level response container returned by the LLM."""
    task1_transcripts: list[Transcript] = pydantic.Field(default_factory=list)
    task2_speakers:    list[Speaker]    = pydantic.Field(default_factory=list)

    # ── Convenience accessors ──────────────────────────────────────────────

    def speaker_map(self) -> dict[int, Speaker]:
        """Return a dict mapping voice ID → Speaker for fast lookup."""
        return {s.voice: s for s in self.task2_speakers}

    def label_for_voice(self, voice: int) -> str:
        """
        Return the best available display label for *voice*.

        Priority: name → position → role_in_video → "[Voice N]"
        """
        speaker = self.speaker_map().get(voice)
        if speaker is None:
            return f"[Voice {voice}]"

        not_found = config.UNKNOWN_SPEAKER_MARKER

        if speaker.name and speaker.name != not_found:
            return speaker.name
        if speaker.position and speaker.position != not_found:
            return f"[{speaker.position}]"
        if speaker.role_in_video and speaker.role_in_video != not_found:
            return f"[{speaker.role_in_video}]"
        return f"[Voice {voice}]"