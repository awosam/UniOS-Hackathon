"""
sentiment.py — Detects student emotional state and selects the right AI persona.

WHY THIS EXISTS:
  A one-size-fits-all response tone doesn't work well across all student situations.
  A student who's panicking the night before an exam needs very different energy than
  a student calmly planning their next term. SentimentEngine reads the student's
  message and the academic calendar context, then recommends the persona that will
  be most helpful for that specific moment.

  The persona name is returned and can be incorporated into the system prompt
  passed to other agents to adjust their tone accordingly.

THE THREE PERSONAS:
  1. Supportive Anchor  — calm, reassuring, stress-reducing
                          Triggered by: stress language, mention of exams/finals
  2. High-Performance Coach — motivating, direct, goal-focused
                          Triggered by: planning, goal-setting, start of term
  3. Empathetic Peer   — friendly, casual, conversational
                          Default for general questions

DESIGN DECISION — SYNC MODEL:
  SentimentEngine.get_persona() is synchronous (not async) because it's
  currently called from synchronous endpoint handlers in main.py.
  If moved to an async route, create an async version like drafter.py does.
"""

import asyncio                          # Imported for potential future async use
from concurrent.futures import ThreadPoolExecutor  # Would be needed for async version

import vertexai
from vertexai.generative_models import GenerativeModel

from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION  # Shared GCP config

# Initialize Vertex AI once at module load — all Vertex modules do this
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# Module-level thread pool (for future async use, not currently active)
_executor = ThreadPoolExecutor(max_workers=4)

# Shared model instance — same reasoning as drafter.py (stateless, thread-safe)
_model = GenerativeModel("gemini-2.5-flash")


class SentimentEngine:
    """
    Analyzes student message + academic context and picks the best AI persona.

    WHY A CLASS AND NOT A FUNCTION:
      A class makes it easy to add state later (e.g. tracking which persona
      was used in previous messages for consistency) without changing the API.
    """

    def get_persona(self, user_input: str, academic_event: str = "Standard Week") -> str:
        """
        Determines the best AI persona for the current interaction.

        Args:
            user_input:     The student's raw message text.
            academic_event: Calendar context like "Finals Week", "First Week of Term",
                            or "Standard Week". Informs persona selection beyond
                            just the words — a neutral message during finals might
                            still warrant Supportive Anchor.

        Returns:
            A string with the persona name and a one-sentence explanation of why it fits.
            Example: "Supportive Anchor: The student's language indicates high stress..."
        """
        prompt = f"""Analyze this student message and academic context, then choose the most appropriate persona.

Student message: "{user_input}"
Academic calendar context: {academic_event}

Available personas:
1. Supportive Anchor — calm, reassuring, stress-reducing.
   Choose when: student seems panicked/overwhelmed, or it's exam/finals period.
2. High-Performance Coach — motivating, direct, results-focused.
   Choose when: student is in planning mode, setting goals, or start of term.
3. Empathetic Peer — friendly, casual, conversational.
   Choose when: general questions, curiosity, no strong emotional signal.

Reply with ONLY the persona name and one sentence explaining why it fits this context."""

        # Direct synchronous call — acceptable here because this endpoint is not
        # called frequently and doesn't need to be non-blocking
        resp = _model.generate_content(prompt)
        return resp.text


# Single shared instance for all API routes
sentiment_engine = SentimentEngine()
