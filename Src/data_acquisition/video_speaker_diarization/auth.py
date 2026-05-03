"""
auth.py — Initializes and validates the Gemini API client.

Credentials are read exclusively from environment variables (set in config.py
which reads from os.getenv).  No API keys are ever hardcoded.

Supported authentication modes
-------------------------------
1. Google AI Studio  — set GOOGLE_API_KEY
2. Vertex AI         — set GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT
                       (+ optionally GOOGLE_CLOUD_LOCATION)
"""

from __future__ import annotations

import os
import sys

from google import genai

import config
from logger import get_logger

log = get_logger(__name__)


def build_client() -> genai.Client:
    """
    Build and return an authenticated Gemini client.

    Reads credentials from environment variables only.
    Exits with a clear error message if no valid credentials are found.
    """
    _apply_env_vars()
    _validate_credentials()

    client = genai.Client()
    _log_client_info(client)
    return client


# ── Private helpers ───────────────────────────────────────────────────────────

def _apply_env_vars() -> None:
    """Write config values into os.environ so the genai SDK picks them up."""
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = str(config.GOOGLE_GENAI_USE_VERTEXAI)

    if config.GOOGLE_GENAI_USE_VERTEXAI:
        if config.GOOGLE_CLOUD_PROJECT:
            os.environ["GOOGLE_CLOUD_PROJECT"]  = config.GOOGLE_CLOUD_PROJECT
            os.environ["GOOGLE_CLOUD_LOCATION"] = config.GOOGLE_CLOUD_LOCATION or "global"
    else:
        if config.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY


def _validate_credentials() -> None:
    """Exit early with a helpful message if no credentials are configured."""
    using_vertex = config.GOOGLE_GENAI_USE_VERTEXAI

    if using_vertex and not config.GOOGLE_CLOUD_PROJECT:
        log.error(
            "Vertex AI mode is enabled but GOOGLE_CLOUD_PROJECT is not set.\n"
            "  → export GOOGLE_CLOUD_PROJECT='your-project-id'"
        )
        sys.exit(1)

    if not using_vertex and not config.GOOGLE_API_KEY:
        log.error(
            "No API key found.  Set the GOOGLE_API_KEY environment variable:\n"
            "  → export GOOGLE_API_KEY='your-api-key'"
        )
        sys.exit(1)


def _log_client_info(client: genai.Client) -> None:
    service = "Vertex AI" if client.vertexai else "Google AI Studio"
    log.info("Authenticated with %s", service)

    if hasattr(client, "_api_client"):
        api = client._api_client
        if getattr(api, "project", None):
            log.info(
                "  Project  : %s…   Location: %s",
                api.project[:8],
                getattr(api, "location", "global"),
            )
        elif getattr(api, "api_key", None):
            key = api.api_key
            log.info("  API key  : %s…%s", key[:5], key[-4:])