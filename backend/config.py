"""
config.py — Centralized application settings.

WHY PYDANTIC SETTINGS:
  pydantic_settings reads values from environment variables and .env files
  automatically. This means we never hardcode secrets in source code.
  Pydantic also validates types — if PORT is missing or not an integer,
  the app crashes immediately at startup with a clear error instead of
  silently misbehaving later.

HOW TO CONFIGURE:
  Create a .env file in the uni_os/ directory with:
    WATERLOO_API_KEY=<your key from openapi.data.uwaterloo.ca>
    CANVAS_API_KEY=<token from Canvas > Account > Settings > New Access Token>
  The Google API key is no longer needed — auth is handled by gcloud ADC.
"""

import os  # Used to build the absolute path to the .env file

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All runtime configuration for Uni-OS.

    Pydantic reads these fields from environment variables (or .env) automatically.
    Fields with no default are REQUIRED — the app will fail to start without them.
    Fields with defaults are OPTIONAL — safe to omit if that feature isn't used.
    """

    # ── Legacy / unused (kept so existing .env files don't cause errors) ──────

    # GOOGLE_API_KEY is no longer used since we switched to Vertex AI + ADC.
    # Marked optional so old .env files with this key still load without error.
    GOOGLE_API_KEY: str = ""

    # ── Waterloo OpenData API ────────────────────────────────────────────────

    # Required for all live UW data (courses, terms, events, etc.)
    # Free key from: https://openapi.data.uwaterloo.ca
    WATERLOO_API_KEY: str = ""

    # ── Canvas LMS ──────────────────────────────────────────────────────────

    # Base URL for Canvas. Most UW students use the cloud version.
    CANVAS_API_URL: str = "https://canvas.instructure.com"

    # Student-generated access token from Canvas > Account > Settings.
    # If empty, the Canvas integration will return empty data gracefully.
    CANVAS_API_KEY: str = ""

    # ── Server ──────────────────────────────────────────────────────────────

    # The port Uvicorn (the web server) listens on.
    # 8000 is the FastAPI convention. The frontend points to this port.
    PORT: int = 8000

    # ── Pydantic config ──────────────────────────────────────────────────────

    model_config = SettingsConfigDict(
        # Build an absolute path to .env so the app works regardless of the
        # working directory it's launched from (important for `python -m backend.main`)
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),

        # If the .env file has extra keys we don't know about, just ignore them
        # instead of raising a validation error. Useful when keys are added/removed.
        extra="ignore",
    )


# Create a single, shared Settings instance.
# Every other module imports this object: `from backend.config import settings`
# This means the .env file is read exactly once at startup.
settings = Settings()
