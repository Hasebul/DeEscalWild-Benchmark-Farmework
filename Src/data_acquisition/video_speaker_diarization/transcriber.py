"""
transcriber.py — Sends audio to Gemini and returns a structured VideoTranscription.

Responsibilities
----------------
* Build the API request (audio part + prompt + config).
* Call client.models.generate_content with retry logic.
* Parse and validate the structured JSON response.
* Return token usage metadata alongside the transcription.
"""

from __future__ import annotations

import mimetypes
import pathlib
import re
from datetime import timedelta
from typing import Union

import tenacity
from google import genai
from google.genai.types import (
    FileData,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    MediaResolution,
    Part,
    ThinkingConfig,
    VideoMetadata,
)

import config
from logger import get_logger
from models import VideoTranscription

log = get_logger(__name__)


# ── Prompts ────────────────────────────────────────────────────────────────────

_AUDIO_PROMPT = """
**Task 1 - Transcripts**
- Listen carefully to the audio file.
- Identify each unique voice using a numeric `voice` ID (1, 2, 3, …).
- Transcribe the audio verbatim with voice diarization.
- Include the `start` timecode ({timecode_spec}) for each speech segment.

**Task 2 - Speakers**
- For each `voice` ID from Task 1, extract information about the corresponding speaker.
- Use audio context (introductions, addressing by name).
- If a piece of information cannot be determined, use "{unknown}" as the value.
- Note: `role_in_video` should be interpreted as `role_in_conversation`.
""".strip()


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    client: genai.Client,
    model: str = config.Model.DEFAULT,
) -> tuple[VideoTranscription, dict[str, int]]:
    """
    Transcribe the audio file at *audio_path* using Gemini.

    Parameters
    ----------
    audio_path:
        Local path to an MP3 (or other audio) file.
    client:
        Authenticated Gemini client (from ``auth.build_client()``).
    model:
        Gemini model ID string from ``config.Model``.

    Returns
    -------
    transcription : VideoTranscription
        Structured transcript + speaker diarization result.
    token_usage : dict[str, int]
        Keys: "input_token", "output_token", "thought_token".
    """
    audio_part = _build_audio_part(audio_path)
    if audio_part is None:
        log.error("Cannot transcribe — failed to load audio from %s", audio_path)
        return VideoTranscription(), {}

    prompt      = _build_prompt(model, audio_path)
    gen_config  = _build_generation_config(model)
    contents    = [audio_part, prompt]

    log.info("Sending audio to Gemini model=%s  file=%s", model, audio_path)
    response = _call_with_retry(client, model, contents, gen_config)

    token_usage    = _extract_token_usage(response)
    transcription  = _parse_response(response)
    return transcription, token_usage


# ── Audio part construction ────────────────────────────────────────────────────

def _build_audio_part(audio_path: str) -> Part | None:
    """Build a Gemini Part from a local audio file (inline bytes)."""
    if not pathlib.Path(audio_path).exists():
        log.error("Audio file not found: %s", audio_path)
        return None

    try:
        path      = pathlib.Path(audio_path)
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "audio/mpeg"
        file_bytes = path.read_bytes()
        return Part(inline_data={"mime_type": mime_type, "data": file_bytes})

    except Exception as exc:
        log.error("Error reading audio file %s: %s", audio_path, exc)
        return None


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(model: str, audio_path: str) -> str:
    timecode_spec = _timecode_spec_for(model, audio_path)
    return _AUDIO_PROMPT.format(
        timecode_spec=timecode_spec,
        unknown=config.UNKNOWN_SPEAKER_MARKER,
    )


def _timecode_spec_for(model: str, audio_path: str) -> str:
    """Use H:MM:SS for long files on capable models, MM:SS otherwise."""
    long_context_models = {config.Model.GEMINI_2_5_FLASH, config.Model.GEMINI_2_5_PRO}
    if model not in long_context_models:
        return "MM:SS"

    duration = _estimate_duration_from_filename(audio_path)
    if duration and duration >= timedelta(hours=1):
        return "MM:SS or H:MM:SS"
    return "MM:SS"


def _estimate_duration_from_filename(path: str) -> timedelta | None:
    """Parse an optional ISO-8601 duration suffix embedded in the filename."""
    name  = pathlib.Path(path).stem
    match = re.search(r"_PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", name)
    if not match:
        return None
    h, m, s = (int(x) if x else 0 for x in match.groups())
    return timedelta(hours=h, minutes=m, seconds=s)


# ── Generation config ─────────────────────────────────────────────────────────

def _build_generation_config(model: str) -> GenerateContentConfig:
    thinking_config = _thinking_config_for(model)
    return GenerateContentConfig(
        temperature=config.TRANSCRIPT_TEMPERATURE,
        top_p=config.TRANSCRIPT_TOP_P,
        seed=config.TRANSCRIPT_SEED,
        response_mime_type="application/json",
        response_schema=VideoTranscription,
        thinking_config=thinking_config,
    )


def _thinking_config_for(model: str) -> ThinkingConfig | None:
    if model == config.Model.GEMINI_2_5_FLASH:
        return ThinkingConfig(
            thinking_budget=config.THINKING_BUDGET_FLASH,
            include_thoughts=False,
        )
    if model == config.Model.GEMINI_2_5_PRO:
        return ThinkingConfig(
            thinking_budget=config.THINKING_BUDGET_PRO,
            include_thoughts=False,
        )
    return None


# ── API call with retry ────────────────────────────────────────────────────────

def _call_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    gen_config: GenerateContentConfig,
) -> GenerateContentResponse:
    retrier = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        wait=tenacity.wait_incrementing(
            start=config.RETRY_WAIT_START_SEC,
            increment=config.RETRY_WAIT_INCREMENT,
        ),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True,
    )

    for attempt in retrier:
        with attempt:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(response: GenerateContentResponse) -> VideoTranscription:
    if not response.candidates:
        log.error("Gemini returned no candidates.")
        return VideoTranscription()

    finish_reason = response.candidates[0].finish_reason
    if finish_reason != FinishReason.STOP:
        log.warning("Unexpected finish reason: %s", finish_reason)

    if not isinstance(response.parsed, VideoTranscription):
        log.error("Could not parse structured response from Gemini.")
        return VideoTranscription()

    return response.parsed


def _extract_token_usage(response: GenerateContentResponse) -> dict[str, int]:
    usage = response.usage_metadata
    if not usage:
        return {"input_token": 0, "output_token": 0, "thought_token": 0}
    return {
        "input_token":  usage.prompt_token_count or 0,
        "output_token": usage.candidates_token_count or 0,
        "thought_token": usage.thoughts_token_count or 0,
    }