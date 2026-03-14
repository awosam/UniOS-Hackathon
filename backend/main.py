"""
main.py — FastAPI application entry point and HTTP route definitions.

WHY FASTAPI:
  FastAPI is an async-native Python web framework that's ideal for AI backends:
  - Routes can be async def, so they don't block while waiting for Gemini or the UW API
  - Auto-generates OpenAPI docs at /docs — easy to explore the API in a browser
  - Built-in request validation via Pydantic type hints
  - Familiar syntax similar to Flask but far more performant for I/O-heavy workloads

WHY ONE FILE FOR ALL ROUTES:
  For a hackathon project, keeping all routes in main.py is the fastest way to
  build and iterate. In a production app these would be split into routers by feature
  (e.g. routes/chat.py, routes/calendar.py, etc.) to stay organized.

CORS (Cross-Origin Resource Sharing):
  Browsers block JavaScript requests to a different port/domain by default.
  Our frontend runs on localhost:3000 and makes requests to localhost:8000 —
  different ports, so the browser would block this. CORSMiddleware adds the
  correct response headers to allow cross-origin requests.

  allow_origins=["*"] allows ALL origins (any frontend, any domain).
  This is fine for development/hackathons. For production, restrict to your
  frontend domain: allow_origins=["https://yourapp.com"]

API CALL BUDGET PER ENDPOINT:
  /chat (Waterloo data query)  → ≤2 AI calls + N parallel HTTP calls
  /chat (policy / general)     → 1 AI call
  /generate-plan               → 2 AI calls (retrieve → generate, via pathfinder)
  /draft-document              → 1 AI call
  /analyze-persona             → 1 AI call
  All others                   → 0 AI calls
"""

from fastapi import FastAPI, UploadFile, File, HTTPException  # FastAPI core
from fastapi.middleware.cors import CORSMiddleware              # Cross-origin request support

# Import all of our agent and integration modules
# Each of these is initialized once as a global instance in its own file
from backend.config import settings                            # App configuration
from backend.agents.policy_decoder import policy_decoder       # RAG for UW policy PDFs
from backend.agents.pathfinder import pathfinder_app           # LangGraph roadmap generator
from backend.integrations.canvas import canvas_client          # Canvas LMS integration
from backend.integrations.calendar import calendar_fusion      # Study plan generator
from backend.agents.drafter import drafter_agent               # Document draft generator
from backend.agents.sentiment import sentiment_engine          # Persona selector
from backend.agents.memory import memory_engine                # Persistent student context
from backend.integrations.peer_pulse import peer_pulse        # Campus vibes aggregator
from backend.agents.chat_agent import chat_agent               # Main AI brain / router

import os      # For file path operations in the PDF upload endpoint
import shutil  # For efficiently copying the uploaded PDF to disk

# ── Application setup ─────────────────────────────────────────────────────────

# The main FastAPI app instance. The title appears in /docs.
app = FastAPI(title="Uni-OS API", version="0.1.0")

# CORS middleware — must be added before any routes are defined
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production: restrict to ["https://your-frontend.com"]
    allow_credentials=True,   # Allows session cookies (not used now, but future-proof)
    allow_methods=["*"],      # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],      # Allow any request headers (e.g. Authorization, Content-Type)
)


# ── Primary chat route ────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(message: str):
    """
    Main conversational interface for the Uni-OS AI.

    The chat_agent internally decides:
    - If the message is asking for live UW data → fetches from Waterloo API then formats
    - If the message is a general/policy question → answers directly with Gemini
    - If the message involves Canvas → reads from Canvas LMS

    Args:
        message: The student's raw message text (passed as a query parameter).

    Returns:
        {"type": "text", "text": "..."} — the AI's formatted response.
    """
    try:
        response = await chat_agent.get_response(message)
        return response
    except Exception as e:
        # Return a 500 error so the frontend can display a meaningful message
        raise HTTPException(status_code=500, detail=str(e))


# ── PDF policy upload ─────────────────────────────────────────────────────────

@app.post("/ingest-policy")
async def ingest_policy(file: UploadFile = File(...)):
    """
    Accepts a UW policy PDF upload and adds it to the RAG knowledge base.

    After uploading, students can ask specific policy questions (e.g. "Can I appeal
    my MATH 115 grade?") and the AI will ground its answers in the actual document.

    Args:
        file: The uploaded PDF file (multipart form upload).

    Returns:
        {"message": "Policy ingested", "chunks": N} — number of text chunks indexed.
    """
    # Only accept PDFs — other formats (Word, text) are not supported by pypdf
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to the backend/data/ directory for storage
    file_path = f"backend/data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # Efficient streaming copy to disk

    try:
        # Parse the PDF, split into chunks, and index for keyword retrieval
        num_chunks = policy_decoder.ingest_pdf(file_path)
        return {"message": "Policy ingested successfully", "chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Academic roadmap generation ───────────────────────────────────────────────

@app.post("/generate-plan")
async def generate_plan(goal: str, student_record: dict):
    """
    Runs the LangGraph 'Pathfinder' workflow to produce an academic roadmap.

    The workflow: retrieves relevant UW policies → synthesizes with student context → roadmap.

    Args:
        goal:           What the student wants to achieve (e.g. "Switch my major to CS").
        student_record: Academic context — GPA, current major, completed courses, etc.

    Returns:
        {"roadmap": [list of step strings]}
    """
    try:
        # Build the initial state for the LangGraph graph
        # policies and roadmap start empty — the graph fills them in during execution
        initial_state = {
            "goal": goal,
            "student_record": student_record,
            "policies": [],  # Will be populated by the policy_retriever node
            "roadmap": [],   # Will be populated by the roadmap_generator node
        }
        # .invoke() runs the entire workflow synchronously and returns the final state
        final_state = pathfinder_app.invoke(initial_state)
        return {"roadmap": final_state["roadmap"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Canvas assignments ────────────────────────────────────────────────────────

@app.get("/assignments")
async def get_assignments():
    """
    Fetches upcoming assignment deadlines from Canvas using the student's API token.

    Requires CANVAS_API_KEY in .env. Returns empty list if not configured.
    """
    try:
        assignments = canvas_client.get_upcoming_assignments()
        return {"assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Document drafting ─────────────────────────────────────────────────────────

@app.post("/draft-document")
async def draft_document(doc_type: str, student_context: dict, policy_query: str = ""):
    """
    Generates a professional academic document (extension request, petition, etc.).

    If policy_query is provided, retrieves relevant policy text first and includes
    it in the draft so the document cites specific UW rules.

    Args:
        doc_type:        Type of document to write (e.g. "extension request").
        student_context: Student details for personalization (name, course, reason, etc.).
        policy_query:    Optional search terms to retrieve relevant policy text.
    """
    try:
        policy_text = ""
        if policy_query:
            # Retrieve relevant policy chunks to ground the document in real rules
            relevant_docs = policy_decoder.query_policies(policy_query)
            policy_text = "\n".join([d.text for d in relevant_docs])

        draft = drafter_agent.generate_draft(doc_type, student_context, policy_text)
        return {"draft": draft}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Persona analysis ──────────────────────────────────────────────────────────

@app.post("/analyze-persona")
async def analyze_persona(user_input: str, academic_event: str = "Standard Week"):
    """
    Returns the AI persona most appropriate for this student message and moment.

    Can be used by the frontend to adjust the UI tone, color scheme, or avatar
    based on how the student is feeling.

    Args:
        user_input:     The student's message.
        academic_event: Calendar context (e.g. "Finals Week", "Start of Term").
    """
    try:
        persona_info = sentiment_engine.get_persona(user_input, academic_event)
        return {"persona": persona_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory / milestones ───────────────────────────────────────────────────────

@app.post("/record-milestone")
async def record_milestone(description: str):
    """
    Saves a student achievement to long-term memory.

    Persisted to backend/data/personal_memory.json and injected into future
    AI prompts so the AI remembers the student's progress.

    Args:
        description: Human-readable description of the milestone.
                     Example: "Passed CS 136 with 88%, improved from 72% in CS 136"
    """
    try:
        memory_engine.save_milestone(description)
        return {"status": "milestone recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personal-context")
async def get_personal_context():
    """
    Returns a text summary of the student's saved history.

    Used by the frontend to display a "Your Progress" panel, or to debug
    what context the AI has access to.
    """
    try:
        return {"context": memory_engine.get_context_summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-memory")
async def reset_memory():
    """
    DEMO MODE: Wipes all saved student profile data, milestones, and struggles.
    Allows for a fresh demo restart without manually deleting files.
    """
    try:
        memory_engine.clear()
        return {"status": "memory cleared", "message": "Demo reset successful. Starting fresh."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Calendar fusion ───────────────────────────────────────────────────────────

@app.post("/generate-study-blocks")
async def generate_study_blocks(assignments: list, free_slots: list):
    """
    Creates an optimized study schedule from deadlines + free time.

    The AI allocates harder assignments to peak energy slots and spreads work
    across multiple days to prevent last-minute cramming.

    Args:
        assignments: List of upcoming deadlines [{title, due, weight}]
        free_slots:  List of available time windows [{day, start, end}]
    """
    try:
        plan = calendar_fusion.generate_study_plan(assignments, free_slots)
        return {"study_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Campus pulse ──────────────────────────────────────────────────────────────

@app.get("/peer-pulse")
async def get_peer_pulse():
    """
    Returns aggregated campus vibe and study spot occupancy data.

    Used by the frontend dashboard to show how busy campus is.
    """
    try:
        return {"pulse": peer_pulse.get_live_vibes()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/advisor-availability")
async def get_advisor_availability():
    """
    Returns currently available academic advisor time slots.

    NOTE: This is mock data for the hackathon demo.
    In production, this would integrate with UW's booking system (e.g. via WaterlooWorks API).
    """
    # Return sample data that realistically represents what the real integration would show
    return {
        "slots": [
            {"advisor": "Dr. Smith (Engineering)", "time": "Today, 2:00 PM",   "link": "/book/smith"},
            {"advisor": "Sarah J. (Arts)",          "time": "Tomorrow, 10:00 AM", "link": "/book/sarah"},
        ]
    }


# ── Server entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Run with --reload so the server restarts automatically when any .py file changes.
    # This is development-mode only — do not use reload=True in production.
    uvicorn.run("backend.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
