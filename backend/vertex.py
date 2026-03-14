"""
vertex.py — Shared Vertex AI initialization for the entire backend.

WHY THIS FILE EXISTS:
  Vertex AI requires vertexai.init() to be called before any model is used.
  Rather than calling it in every single file, we do it ONCE here at import
  time. Every other module simply imports VERTEX_PROJECT / VERTEX_LOCATION
  from here, so initialization is guaranteed to run exactly once.

AUTHENTICATION:
  No API key is used. Instead, we rely on Application Default Credentials (ADC),
  which reads from ~/.config/gcloud/application_default_credentials.json.
  This is set up by running: gcloud auth application-default login
  ADC is the recommended Google Cloud auth method — it works locally and in
  cloud environments (GKE, Cloud Run, etc.) without changing code.

BILLING:
  All Gemini API calls are charged to the 'hackaton-490211' GCP project,
  which has a $300 free credit. Costs ~$0.001 per conversation at current usage.
"""

import vertexai
from vertexai.generative_models import GenerativeModel

# ── Project configuration ─────────────────────────────────────────────────────

# The GCP project ID where Vertex AI is enabled and billed
VERTEX_PROJECT = "hackaton-490211"

# The GCP region hosting the Gemini models. us-central1 has the widest model
# availability and lowest latency from North America.
VERTEX_LOCATION = "us-central1"

# The best available Gemini model as of March 2026.
# gemini-2.5-flash is faster and cheaper than 2.5-pro while still having
# 1M token context and strong reasoning. Good balance for a chatbot.
DEFAULT_MODEL = "gemini-2.5-flash"

# ── One-time initialization ───────────────────────────────────────────────────

# This runs the moment any module does `from backend.vertex import ...`
# Python's module system guarantees this runs only once per process — if
# multiple modules import this file, Python serves the cached module object.
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)


# ── Factory function ──────────────────────────────────────────────────────────

def vertex_model(model_name: str = DEFAULT_MODEL) -> GenerativeModel:
    """
    Returns a new GenerativeModel instance for the given model name.

    WHY A FUNCTION INSTEAD OF A GLOBAL INSTANCE:
      Each file creates its own model instance because GenerativeModel objects
      are lightweight (they don't hold a connection) and this keeps modules
      independent. If we needed model-level configuration (like system
      instructions or safety settings), each module can customize its own instance.

    Args:
        model_name: Vertex AI model name. Defaults to gemini-2.5-flash.
    """
    return GenerativeModel(model_name)
