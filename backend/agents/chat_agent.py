"""
chat_agent.py — The primary AI brain of Uni-OS.

WHY THIS FILE IS THE MOST IMPORTANT:
  Every student message goes through this file. It decides:
  1. WHAT type of question it is (Waterloo data vs. general/policy)
  2. WHICH data to fetch (courses? exams? food?)
  3. HOW to respond (format live data, or answer directly with AI)

OVERALL FLOW:
  Student message
    └─► _is_waterloo() — is this about UW data?
         ├─ YES → _fetch_waterloo() — fetch from UW OpenData API
         │           └─► _format_waterloo() — AI formats the data for the student
         └─ NO  → _answer_directly() — single AI call handles everything else

AI CALL BUDGET (why we care about this):
  Vertex AI charges per token. Keeping calls minimal = lower cost + faster responses.
  • Waterloo data query: ≤2 AI calls (subject detect + format) + N parallel HTTP fetches
  • General / policy:    1 AI call (answer directly from Gemini's knowledge)
  • Canvas:              0 AI calls (pure HTTP)

SUBJECT DETECTION STRATEGY:
  To fetch the right courses, we need the UW subject code (e.g. "ECE", "MATH").
  We use a two-step approach that minimizes AI calls:
  Step 1 — Regex (0 AI calls): if the student typed "ECE 222" or "MATH", extract it directly.
  Step 2 — AI (1 AI call): for natural language like "computer engineering", ask Gemini.
  This works for all 258+ UW departments without any dictionary to maintain.

ASYNC THROUGHOUT:
  Every function that calls Vertex AI or the UW API is async. Vertex's SDK is
  synchronous, so we run it in a ThreadPoolExecutor with run_in_executor() to avoid
  blocking FastAPI's event loop. See waterloo_api.py comments for more detail.
"""

import asyncio                           # Core async primitives
import json                              # Serializes Waterloo API data for the AI prompt
import re                                # Regex for subject code detection
from concurrent.futures import ThreadPoolExecutor  # Thread pool for sync Vertex SDK calls
from typing import Any, Dict, List, Optional, Tuple  # Type hints

import vertexai
from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions   # Vertex AI error types (for potential future error handling)

from backend.agents.memory import memory_engine      # Student context (milestones, struggles)
from backend.config import settings                  # App configuration
from backend.integrations.canvas import canvas_client  # Canvas LMS (not audio used in chat flow currently)
from backend.tools.waterloo_api import (             # All UW API fetchers
    fetch_parallel,
    get_courses_async,
    get_events_async,
    get_exams_async,
    get_food_async,
    get_holidays_async,
    get_important_dates_async,
    get_locations_async,
    get_news_async,
    get_terms_async,
    get_term_courses_async,
    get_course_detail_async,
    get_subjects_async,
)

# Thread pool: max 8 concurrent Vertex SDK calls across the whole backend.
# 8 is enough for our usage while keeping memory overhead low.
_executor = ThreadPoolExecutor(max_workers=8)


# ── Global Lists ─────────────────────────────────────────────────────────────

# Common English words that match the 2-6 letter pattern but are not subject codes.
_NOT_A_CODE = {
    "me", "co", "so", "is", "in", "of", "or", "at", "to", "as",
    "an", "be", "do", "go", "he", "it", "my", "no", "on", "up",
    "we", "by", "if", "ok", "show", "are", "the", "for", "and",
}

# The set of UW subject codes our API client can actually fetch data for.
_KNOWN_CODES = {
    "CS", "MATH", "PHYS", "BIOL", "CHEM", "ENGL", "ECON", "STAT",
    "ECE", "SE", "CO", "PMATH", "HIST", "PSYCH", "SOC", "ARTS",
    "MUSIC", "PHIL", "GEOG", "AFM", "ACTSC", "MGMT", "SCI", "AMATH",
    "ENVS", "FINE", "EARTH", "LING", "RS", "GER", "FR", "SPAN",
    "JAPAN", "CHINA", "KOR", "INTEG", "STV", "HLTH", "GBDA",
}

# Matches 'SUBJ ###' or 'SUBJ###' (e.g. CS 135, ECE222)
_COURSE_CODE_RE = re.compile(r'\b([A-Z]{2,6})\s*\d{3}[A-Z]{0,2}\b', re.IGNORECASE)

async def _detect_subject(user_input: str, ai_fn, elective_context: bool = False) -> str:
    """
    Returns the UW subject code most relevant to the student's message.

    STEP 1 — EXPLICIT COURSE CODES (zero AI cost):
      Look for patterns like "CS 135" or "ECE 220". If the student cited a specific
      course, they almost certainly want data about that subject.
      
    STEP 2 — AI DISAMBIGUATION (1 call):
      If no explicit course code is found, we ask Gemini to extract the subject.
      This is crucial for queries like "humanities electives for math majors" where
      both "humanities" (ARTS) and "math" (MATH) are subjects, but only one is the
      target of the request.
    """
    # Step 1: Explicit course code check (e.g. "CS 135")
    match = _COURSE_CODE_RE.search(user_input)
    if match:
        candidate = match.group(1).upper()
        if candidate in _KNOWN_CODES:
            return candidate

    # Step 2: AI extraction with context awareness
    context_hint = " The student is specifically looking for ELECTIVE courses." if elective_context else ""
    prompt = (
        f'Which University of Waterloo subject code (e.g. CS, MATH, PSYCH) is the primary '
        f'subject of interest in this student\'s message?{context_hint}\n'
        f'Message: "{user_input}"\n\n'
        f'If multiple subjects are mentioned, prioritize the one they want more information '
        f'on, NOT their current major.\n'
        f'Reply with ONLY the subject code. If unsure, reply CS.'
    )
    result = await ai_fn(prompt) or "CS"

    # Take the first token and validate it
    for word in result.strip().split():
        clean = word.strip(".,;:)(").upper()
        if clean in _KNOWN_CODES:
            return clean
        if clean.isalpha() and 2 <= len(clean) <= 6:
            return clean

    return "CS"


def _get_target_term(text: str) -> Optional[str]:
    """
    Detects if the student is asking about a specific term (e.g. 'Spring 2026').
    Returns the UW term code (e.g. '1265') or None.
    
    UW Term Logic: 1 + (Year-1900) + Term digit
    Term digits: 1 (Winter), 5 (Spring), 9 (Fall)
    Example: Spring 2026 -> 1 + 126 + 5 = 1265
    """
    lower = text.lower()
    
    # ── Term Keywords ────────────────────────────────────────────────────────
    seasons = {
        "spring": "5", "summer": "5", # Waterloo calls summer 'Spring'
        "winter": "1",
        "fall": "9", "autumn": "9"
    }
    
    # Detect year. Default to 2026 as per user clock if not specified.
    year = 2026
    year_match = re.search(r"\b(202[4-9])\b", lower)
    if year_match:
        year = int(year_match.group(1))
        
    for name, digit in seasons.items():
        if name in lower:
            # Century (1 for 2000s) + Year last 2 digits + Term digit
            # 2026 -> 1 + 26 + 5 = 1265
            yy = str(year)[2:]
            return f"1{yy}{digit}"
            
# ── AI-First Parameter Extraction ──────────────────────────────────────────

def _calculate_term_code(season: str, year: str) -> Optional[str]:
    """Helper to convert AI-extracted season/year into a UW term code."""
    if not season or not year: return None
    try:
        y_int = int(year)
        yy = str(y_int)[2:]
        digits = {"winter": "1", "spring": "5", "summer": "5", "fall": "9"}
        digit = digits.get(season.lower())
        if digit: return f"1{yy}{digit}"
    except (ValueError, TypeError): pass
    return None


async def _fetch_waterloo(user_input: str, ai_fn, intent_data: Dict) -> Tuple[Optional[Any], str]:
    """
    Fetches Waterloo data based on structured AI-extracted intent and params.

    WHY FULLY AI-DRIVEN:
      Manual fallbacks like 'if news in lower' are too brittle. 
      The AI extracted 'tool' and 'params' are now the single source of truth.
    """
    tool = intent_data.get("tool", "").upper()
    params = intent_data.get("params", {})
    
    # Extract common params
    subject = params.get("subject", "").upper()
    catalog = params.get("catalog", "")
    term_code = params.get("term_code") or _calculate_term_code(params.get("season"), params.get("year"))
    query = params.get("query")

    # Dispatch to appropriate fetching tool
    if tool == "EXAMS":
        data, terms = await fetch_parallel(get_exams_async(), get_terms_async())
        return {"exams": data, "term_context": terms}, "exam schedule"

    if tool == "FOOD":
        return await get_food_async(), "campus dining options"

    if tool == "HOLIDAYS":
        return await get_holidays_async(), "university holidays"

    if tool == "NEWS":
        return await get_news_async(8), "campus news"

    if tool == "EVENTS":
        return await get_events_async(8), "campus events"

    if tool == "LOCATIONS":
        return await get_locations_async(query), "locations"

    if tool == "TERMS":
        data, imp = await fetch_parallel(get_terms_async(), get_important_dates_async())
        return {"terms": data, "important_dates": imp}, "term dates"

    if tool == "COURSE_DETAIL":
        return await get_course_detail_async(subject, catalog, term_code), f"{subject} {catalog} details"

    if tool == "TERM_VIEW":
        return await get_term_courses_async(term_code), f"all courses for term {term_code}"

    if tool == "ELECTIVES":
        if subject and subject != "CS":
            courses, terms = await fetch_parallel(get_courses_async(subject), get_terms_async())
            return {"elective_courses": {subject: courses}, "terms": terms}, f"{subject} electives"
        else:
            elective_subjects = ["ENGL", "PSYCH", "ECON", "HIST", "PHIL", "SOC", "MUSIC", "FINE"]
            results = await fetch_parallel(*[get_courses_async(s) for s in elective_subjects])
            combined = {s: r for s, r in zip(elective_subjects, results)}
            terms = await get_terms_async()
            return {"elective_courses": combined, "terms": terms}, "representative electives"

    if tool == "COURSES" or subject:
        # If term is explicitly mentioned but no specific subject? (e.g. "Spring 2026 courses")
        if term_code and not subject:
           return await get_term_courses_async(term_code), f"all courses for term {term_code}"
           
        courses, terms = await fetch_parallel(get_courses_async(subject or "CS"), get_terms_async())
        return {"courses": courses, "terms": terms, "target_term": term_code}, f"{subject or 'unknown'} courses"

    return None, ""


async def _classify_intent(user_input: str, ai_fn, memory_ctx: str = "") -> Dict:
    """
    Primary AI router that extracts both intent AND parameters in one call.
    Uses student memory context to resolve implicit subjects (e.g. 'my major').
    """
    current_year = 2026 # Context-aware for term math
    prompt = f"""You are the Master Router for Uni-OS at UWaterloo.
{memory_ctx}

Classify the student message: "{user_input}"

CATEGORIES & TOOLS:
1. WATERLOO_DATA: 
   - [COURSES]: List of courses for a subject.
   - [COURSE_DETAIL]: Description, credits, or prerequisites for ONE specific course.
   - [TERM_VIEW]: Overview of ALL courses in a term (e.g. "what's offered in Spring?").
   - [EXAMS], [TERMS], [FOOD], [NEWS], [EVENTS], [LOCATIONS], [HOLIDAYS], [ELECTIVES].
2. POLICY_SEARCH: Academic rules, appeals, petitions.
3. CANVAS: Assignments/Grades.
4. GENERAL: Small talk/Advice.

PARAMETER EXTRACTION RULES:
- subject: 2-6 letter code (CS, MATH, ECE, BIOL, CHEM, HLTH, KIN). 
  *CRITICAL*: If the student refers to "my courses", "my degree", or "my major", map their known major to legit Waterloo subject codes. 
  Example: "Pre-Med" -> BIOL or CHEM. "Health Science" -> HLTH or KIN. 
  Defaults to the student's primary academic interest from memory.

- catalog: 3-digit number for COURSE_DETAIL (e.g. 135).
- season: spring, summer, winter, fall.
- year: 2024-2030 (convert "4th year" to future year based on current year 2026).
- query: raw search string for buildings or news.

Response Format: JSON ONLY.
Example: {{"intent": "WATERLOO_DATA", "tool": "COURSE_DETAIL", "params": {{"subject": "MATH", "catalog": "135", "season": "winter"}}}}
"""
    try:
        raw = await ai_fn(prompt)
        if not raw: return {"intent": "GENERAL"}
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        return data
    except Exception:
        return {"intent": "GENERAL"}


# ── ChatAgent class ───────────────────────────────────────────────────────────

class ChatAgent:
    """
    The primary conversation handler for Uni-OS.

    RESPONSIBILITY:
      1. Decide whether to fetch live UW data or answer with AI directly.
      2. Inject student memory context into every AI prompt.
      3. Format all responses coherently for the frontend.

    DESIGN — WHY A CLASS:
      Holding the Gemini model instance as self.model is cleaner than a module-level
      global when the model needs initialization that happens only once per server
      startup. The class also makes it easy to add per-instance state later
      (e.g. conversation history, per-user model configuration).
    """

    def __init__(self):
        # Initialize Vertex AI with our GCP project credentials
        from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

        # The Gemini model instance. GenerativeModel is stateless — it holds no
        # connection or session. Creating it here is cheap.
        self.model = GenerativeModel("gemini-2.5-flash")

    async def _ai(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to Gemini and returns the text response.

        WHY run_in_executor:
          vertexai's generate_content() is a blocking (synchronous) call.
          Calling it directly in an async function blocks the entire FastAPI event
          loop until the response arrives — no other requests can be handled during
          that time. run_in_executor runs it on a background thread, keeping the
          event loop free.

        WHY NO RETRY LOGIC HERE:
          We're on Vertex AI with 300+ RPM. Rate-limit retries are unnecessary.
          If a single call fails, let the top-level exception handler in get_response()
          catch it and return a clean error message.

        Returns:
            Response text string, or None if the response had no content.
        """
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            _executor,
            lambda: self.model.generate_content(prompt)  # Blocking call — safe in thread
        )
        # Check that the response actually has content (safety filters can block some responses)
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.text
        return None

    async def get_response(self, user_input: str) -> Dict:
        """
        Main entry point — handles a student message from start to finish.

        FLOW:
          1. Load student memory context (milestones, struggles) for personalization.
          2. Check if this is a Waterloo data question.
          3a. If yes: fetch live data → format with AI.
          3b. If no: answer directly with a single AI call.
          4. Return {"type": "text", "text": "..."} for the frontend.

        WHY WRAP EVERYTHING IN TRY/EXCEPT:
          If anything unexpected fails (network down, Vertex error, etc.), we catch it
          here and return a clean user-facing message instead of crashing with a 500.
        """
        # Load student history — injected into every AI prompt for personalization
        memory_ctx = (
            memory_engine.get_context_summary()
            if hasattr(memory_engine, "get_context_summary")
            else "No history yet."
        )

        # ── Background Memory Update ──────────────────────────────────────────
        # FIRE AND FORGET: Extract personal info from this message in the background.
        # This keeps the main chat response fast while the 'Memory Janitor' 
        # updates the student's profile, milestones, and struggles behind the scenes.
        asyncio.create_task(self._update_memory_background(user_input))

        try:
            # ── AI-First Intent Routing ──────────────────────────────────────────
            classification = await _classify_intent(user_input, self._ai, memory_ctx)
            intent = classification.get("intent", "GENERAL")
            tool = classification.get("tool")

            # ── Initialize Trace Log ─────────────────────────────────────────────
            trace_log = {
                "memory_context": memory_ctx.replace("\n", " ").replace("Student Profile:", "").strip(),
                "router_intent": intent,
                "router_tool": tool,
                "router_params": classification.get("params", {})
            }

            # ── Branch 1: Waterloo live data ─────────────────────────────────────
            if intent == "WATERLOO_DATA":
                data, label = await _fetch_waterloo(user_input, ai_fn=self._ai, intent_data=classification)
                if data:
                    trace_log["data_source"] = f"Waterloo API ({label})"
                    raw_str = json.dumps(data)
                    trace_log["raw_data_snippet"] = raw_str[:200] + "..." if len(raw_str) > 200 else raw_str
                    
                    response = await self._format_waterloo(user_input, label, data, memory_ctx)
                    return self._append_trace(response, trace_log)

            # ── Branch 2: Policy RAG search ──────────────────────────────────────
            # If AI says POLICY_SEARCH, or if the user asks a very policy-like question
            if intent == "POLICY_SEARCH" or any(w in user_input.lower() for w in ["according to policy", "rule", "appeal"]):
                from backend.agents.policy_decoder import policy_decoder
                answer = policy_decoder.query_with_context(user_input)
                
                trace_log["data_source"] = "Policy RAG Database"
                trace_log["raw_data_snippet"] = "Retrieved top 5 policy chunks..."
                return self._append_trace({"type": "text", "text": answer}, trace_log)

            # ── Branch 3: Canvas LMS ─────────────────────────────────────────────
            # (Reserved for future deep integration, currently handled by tool_router nodes)

            # ── Branch 4: Direct AI response (General/Small talk/Advice) ──────────
            trace_log["data_source"] = "None (Direct AI Response)"
            response = await self._answer_directly(user_input, memory_ctx)
            return self._append_trace(response, trace_log)

        except Exception as e:
            # Log the error server-side for debugging
            print(f"[ChatAgent] Error handling '{user_input[:50]}...': {e}")
            return {
                "type": "text",
                "text": "Something went wrong on my end — please try again in a moment. [Source: Uni-OS]"
            }

    def _append_trace(self, response: Dict, trace_log: Dict) -> Dict:
        """Appends the AI Transparency Trace to the final response."""
        trace_md = "\n\n---\n**🔍 System Trace:**\n```json\n"
        trace_md += json.dumps(trace_log, indent=2)
        trace_md += "\n```"
        
        if "text" in response:
            response["text"] += trace_md
        return response

    async def _format_waterloo(
        self, user_input: str, label: str, data: Any, memory_ctx: str
    ) -> Dict:
        """
        Formats raw Waterloo API data into a friendly, readable student response.

        WHY LET AI FORMAT AND NOT TEMPLATE STRINGS:
          Different questions need different formatting even for the same endpoint.
          "What CS courses are available?" needs a clean list.
          "Is CS 246 offered this term?" needs a yes/no with context.
          AI handles all these variations naturally from a single prompt.

        WHY TRUNCATE AT 6000 CHARS:
          Gemini's 1M token window could fit the entire response, but:
          1. Token cost is proportional to input length
          2. Beyond ~20 courses, students get overwhelmed with information
          3. We already cap API responses to 20 courses in waterloo_api.py
          The 6000-char limit is a safety net for unexpectedly large responses.

        Args:
            user_input:  Original student message (used to personalize the answer).
            label:       Short description of what the data contains (used in prompt).
            data:        Raw API response dict or list.
            memory_ctx:  Student history summary injected for personalization.
        """
        # Serialize to indented JSON so Gemini can read the structure clearly
        data_str = json.dumps(data, indent=2)

        # Safety truncation — prevents accidentally huge prompts
        if len(data_str) > 6000:
            data_str = data_str[:6000] + "\n... (additional results omitted for length)"

        prompt = f"""You are Uni-OS, the University of Waterloo academic companion.
Student history: {memory_ctx}
Student asked: "{user_input}"

Live UW data ({label}):
{data_str}

Write a complete, accurate, student-friendly response using this data.

FORMATTING RULES:
- For single-department courses: list each as "SUBJ### — Title (X units): description"
- For elective/complementary courses: group by department, intro each group with the
  department name, then list 3-5 standout courses from that department.
- For terms: call out exact start/end dates and add/drop deadlines explicitly.
- For term-wide summaries (where 'total_courses' or 'subjects_offered' is provided): 
  explain that you have retrieved the high-level stats for the entire term. List the total 
  courses found and highlight the top 5-10 subjects with the most offerings.
- For first-year questions: highlight 100-level entries specifically.

CRITICAL: This app aims to be a primary source of information, but some academic advice requires an official.
1. ALWAYS prioritize giving a direct, complete answer using the provided data.
2. If the data is present (even if it's a high-level summary), YOU MUST use it to answer the question. NEVER say "I don't have this list" if data is provided below.
3. If the data is missing from the API/Context, but can be found in your general knowledge (for policy), provide that knowledge first.
4. ONLY suggest meeting an academic advisor or checking the official academic calendar IF:
   - The information is NOT in the data/context.
   - The question is highly personalized (e.g., student-specific credit transfers, special petitions).
   - The decision is high-stakes and requires official confirmation that the AI cannot legally or accurately provide.
5. NEVER give a generic brush-off to "check the website" if the answer is available to you.

End with [Source: UWaterloo OpenData API]."""

        text = await self._ai(prompt)
        return {
            "type": "text",
            "text": text or "I retrieved the data but couldn't format it — raw data at uwaterloo.ca"
        }

    async def _answer_directly(self, user_input: str, memory_ctx: str) -> Dict:
        """
        Answers policy, general, and emotional questions with a single AI call.

        WHY ONE PROMPT FOR ALL NON-WATERLOO QUESTIONS:
          Policy (grading rules, co-op requirements), emotional support, and document
          drafting all benefit from the same empathetic, UW-specific AI persona.
          A single well-crafted prompt handles all of them better than separate handlers.

        PROMPT DESIGN:
          The prompt tells Gemini:
          1. Its persona ("academic companion") and the specific university (UW)
          2. What categories of questions it might receive and how to handle each
          3. Source citation format for transparency
          4. Length limit to keep responses scannable (unless drafting, which needs full docs)

        Args:
            user_input:  Student's raw message.
            memory_ctx:  Injected student history for personalized responses.
        """
        prompt = f"""You are Uni-OS, an empathetic University of Waterloo academic companion.
Student history: {memory_ctx}
Student message: "{user_input}"

Respond helpfully, warmly, and specifically:
- Academic policy / grading / passing grades: give precise UW rules and percentages.
  UW uses percent grades, not GPA. Know the difference (Term Average, Cumulative Average).
- Co-op / career: reference WaterlooWorks, 4-month work terms, 5-6 terms for Engineering.
- Document / email drafting: write the complete draft, ready to send, with [placeholder] fields.
- Academic planning / roadmap: give concrete step-by-step UW-specific guidance.
- Emotional support / stress: be warm, encouraging, and practical.

ADVISOR GUIDANCE:
- ALWAYS prioritize providing a direct, accurate answer from your knowledge base first.
- If the student is asking for general information (e.g., "what is the passing grade for ECE?") answer directly.
- ONLY recommend consulting an academic advisor or the academic calendar if:
    a) The information is highly personal (e.g., "why was MY petition denied?")
    b) The information is missing from your knowledge and the API.
    c) The decision is critical and requires official signing/approval.
- Avoid generic brush-offs to "check the website" if you can answer the question.

- Cite your source in [brackets] at the end: [Source: UW Academic Calendar],
  [Source: UW Co-op Education], or [Source: Uni-OS Knowledge] as appropriate.
Keep responses under 400 words unless you're drafting a document (then write the full doc)."""

        text = await self._ai(prompt)
        return {
            "type": "text",
            "text": text or "I couldn't generate a response — please try rephrasing your question."
        }

    async def _update_memory_background(self, user_input: str):
        """
        The 'Memory Janitor' — extracts personal details from conversation.
        
        WHY BACKGROUND TASK:
          Extracting info requires an extra AI call. We don't want the student
          waiting an extra second for every message. By running this in the
          background, we update the profile without affecting chat latency.
        """
        prompt = f"""Extract any new student personal information from this message.
Message: "{user_input}"

Look for:
1. Name (e.g. "I'm Alice")
2. Major (e.g. "I'm in software engineering")
3. Year (e.g. "I'm a 2A student")
4. Milestone (e.g. "I just passed CS 135")
5. Struggle (e.g. "I'm failing calculus")

Return ONLY a JSON object with these keys: "name", "major", "year", "milestone", "struggle".
Use "null" for any fields not found OR if the information is too vague to be an update.
DO NOT return "Unknown" or generic names like "Student".
Example: {{"name": "Alice", "major": "Software Engineering", "year": "2A", "milestone": "Passed CS 135", "struggle": null}}"""

        result = await self._ai(prompt)
        if not result:
            return

        try:
            # Clean the string in case Gemini adds markdown blocks
            clean_json = result.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            # Update Profile
            profile_updates = {}
            for key in ["name", "major", "year"]:
                val = str(data.get(key)).strip().lower()
                if val and val not in ["null", "none", "unknown", "student"]:
                    profile_updates[key] = data[key]
            
            if profile_updates:
                memory_engine.learn_personal_info(profile_updates)
            
            # Update Milestones/Struggles
            if data.get("milestone") and data["milestone"] != "null":
                memory_engine.save_milestone(data["milestone"])
            
            if data.get("struggle") and data["struggle"] != "null":
                # For struggles, we try to split by area/detail if possible, 
                # but for simplicity we'll just use the text as detail.
                memory_engine.add_struggle("General", data["struggle"])
                
        except Exception as e:
            # Silent fail for background tasks to avoid impacting the main thread
            print(f"[MemoryJanitor] Failed to extract info: {e}")


# ── Global instance ───────────────────────────────────────────────────────────

# Created once when the module is imported by main.py.
# All HTTP requests share this single instance (thread-safe because the model is stateless).
chat_agent = ChatAgent()
