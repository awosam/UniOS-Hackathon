"""
drafter.py — Generates professional academic documents for students.

WHY THIS AGENT EXISTS:
  Students often need to write formal communications to university administration
  (extension requests, academic appeals, bursary applications) but don't know
  the right tone, structure, or which UW policies to cite. This agent acts as
  an "Admin Concierge" — it produces a complete, ready-to-send draft from just
  a few pieces of student context.

DESIGN DECISIONS:
  - Async-first: generate_draft_async() is the primary method. The sync wrapper
    generate_draft() exists only for compatibility with older non-async callers.
  - Module-level model: _model is initialized once when the module loads,
    not inside the class, because GenerativeModel is stateless and cheap to share.
  - Thread pool: Vertex AI's SDK is synchronous. We run it in _executor so it
    doesn't block FastAPI's async event loop (which would freeze other requests).
"""

import asyncio                          # For running async code and event loop interaction
from concurrent.futures import ThreadPoolExecutor  # Runs sync Vertex code off the main thread
from typing import Dict                 # Type hint for the student_context parameter

import vertexai
from vertexai.generative_models import GenerativeModel

from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION  # Shared project config

# Initialize Vertex AI at module-load time (runs once per process)
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# Thread pool for running the synchronous Vertex SDK without blocking async code.
# max_workers=4 means up to 4 draft generations can happen simultaneously.
_executor = ThreadPoolExecutor(max_workers=4)

# Single shared model instance — GenerativeModel holds no connection state,
# so sharing it across calls is thread-safe and avoids repeated object creation.
_model = GenerativeModel("gemini-2.5-flash")


async def _ai(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns the text response.

    WHY run_in_executor:
      FastAPI runs on an async event loop. If we call _model.generate_content()
      directly (it's a blocking/sync call), the entire server freezes until it
      returns. run_in_executor offloads it to a thread so other async tasks
      (other HTTP requests) can proceed concurrently.
    """
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        _executor,
        lambda: _model.generate_content(prompt)  # This call blocks — safe in a thread
    )
    return resp.text


class DrafterAgent:
    """
    Writes professional academic documents on behalf of students.

    Supported document types (examples):
      - "extension request" — formal email to a professor asking for more time
      - "academic petition" — formal appeal to the faculty's petitions office
      - "bursary application" — financial support request letter
      - "email to advisor" — question or concern to an academic advisor

    The AI uses the student context and any provided policy text to produce a
    complete, correctly-toned document ready to send or adapt.
    """

    async def generate_draft_async(
        self, doc_type: str, student_context: Dict, policy_context: str = ""
    ) -> str:
        """
        Generates the draft document asynchronously.

        Args:
            doc_type:        What type of document to write (e.g. "extension request")
            student_context: Dict of student details — name, course, reason, deadline, etc.
            policy_context:  Optional relevant UW policy text retrieved by policy_decoder.
                             When provided, the AI cites specific rules to strengthen the request.
        """
        prompt = f"""You are the 'Admin Concierge' for Uni-OS — a University of Waterloo academic assistant.
Write a professional, empathetic, and persuasive {doc_type} for the student.

Student context: {student_context}
Relevant UW policy: {policy_context}

Instructions:
- Use a formal, respectful tone appropriate for UW administration.
- If policy is provided, cite it explicitly to support the student's case.
- Use [square brackets] for placeholders the student must fill in (e.g. [Student ID], [Date]).
- Write the complete document — subject line, salutation, body, closing — ready to send."""
        return await _ai(prompt)

    def generate_draft(
        self, doc_type: str, student_context: Dict, policy_context: str = ""
    ) -> str:
        """
        Synchronous wrapper around generate_draft_async.

        WHY THIS EXISTS:
          Some callers (like main.py endpoints that weren't made async) call this
          synchronously. asyncio.run() creates a new event loop for that call.
          Note: this cannot be called from inside an already-running event loop —
          use generate_draft_async() in async contexts (like FastAPI endpoints).
        """
        return asyncio.run(self.generate_draft_async(doc_type, student_context, policy_context))


# Single shared instance — imported by main.py and any other route handler
drafter_agent = DrafterAgent()
