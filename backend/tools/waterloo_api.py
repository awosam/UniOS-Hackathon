"""
waterloo_api.py — Async-first client for the University of Waterloo OpenData API.

API DOCUMENTATION: https://openapi.data.uwaterloo.ca

WHY ASYNC-FIRST:
  FastAPI processes each HTTP request on an async event loop. If we make
  synchronous HTTP calls (requests.get) directly in an async route handler,
  the entire server freezes waiting for the response — no other requests can
  be served. The solution:
    - All public functions have an async version (get_courses_async, etc.)
    - Each async function runs the sync call in a thread pool via _aget()
    - This keeps the event loop free to handle other requests concurrently

WHY BOTH SYNC AND ASYNC VERSIONS:
  Some callers (like scripts, tests, or LangGraph nodes) are not async.
  We provide sync versions (get_courses, get_terms, etc.) for these cases.
  They call the same _get() helper so logic isn't duplicated.

PARALLEL FETCHING:
  Use fetch_parallel() when a single response needs multiple data sources.
  Example: answering "what CS courses start this term?" needs BOTH courses AND
  term dates. fetch_parallel() fires both HTTP requests simultaneously, cutting
  wait time roughly in half.

CACHING:
  _current_term_code() is cached with @lru_cache(maxsize=1). The current term
  doesn't change mid-session, so we call /Terms once and reuse the result.
  This avoids a redundant API call on every course lookup.

API KEY:
  Set WATERLOO_API_KEY in your .env file. Get a free key at openapi.data.uwaterloo.ca.
  Without it, the API still responds for some endpoints but throttles heavily.
"""

import asyncio                          # For async await and gather
import os                               # To read WATERLOO_API_KEY from environment
from concurrent.futures import ThreadPoolExecutor  # Runs sync requests.get off the event loop
from datetime import datetime, timezone  # For comparing term start dates to today
from functools import lru_cache         # Caches _current_term_code() — called frequently
from typing import Any, Dict, List, Optional  # Type hints

import requests                         # Standard HTTP library (sync)
from dotenv import load_dotenv          # Reads .env file into os.environ

# Load .env so os.getenv("WATERLOO_API_KEY") works even without server-level env vars
load_dotenv()

# ── Shared HTTP session with connection pooling ───────────────────────────────

# A Session reuses TCP connections between requests (unlike standalone requests.get()).
# This speeds up bursts of API calls (e.g. parallel fetches) by avoiding the
# TCP handshake overhead on every request.
_session = requests.Session()

# Set the API key header on the session so every request includes it automatically.
# Uses the WATERLOO_API_KEY env var, or empty string if not set (still works but throttled).
_session.headers.update({"x-api-key": os.getenv("WATERLOO_API_KEY", "")})

# Thread pool: up to 10 concurrent HTTP requests.
# 10 workers means up to 10 API calls can happen in parallel without blocking each other.
_executor = ThreadPoolExecutor(max_workers=10)

# The base URL for all Waterloo OpenData v3 endpoints
BASE = "https://openapi.data.uwaterloo.ca/v3"


# ── Internal HTTP helpers ─────────────────────────────────────────────────────

def _get(path: str, params: Optional[Dict] = None) -> Any:
    """
    Synchronous GET request using the shared session.

    WHY RETURN A DICT WITH {"error": ...} INSTEAD OF RAISING:
      The caller (chat_agent) passes this data directly to Gemini as context.
      If it's an error dict, Gemini can tell the student "that data is unavailable"
      instead of crashing the entire request. This makes error handling graceful.

    Args:
        path:   The API endpoint path, e.g. "/Courses/1265/CS"
        params: Optional query parameters dict

    Returns:
        Parsed JSON (list or dict) on success, {"error": ...} dict on failure.
    """
    try:
        r = _session.get(f"{BASE}{path}", params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
        # Include the URL in the error dict for easier debugging
        return {"error": f"HTTP {r.status_code}", "url": f"{BASE}{path}"}
    except Exception as e:
        return {"error": str(e)}


async def _aget(path: str, params: Optional[Dict] = None) -> Any:
    """
    Async wrapper around _get(). Runs the blocking HTTP call in the thread pool.

    WHY run_in_executor AND NOT asyncio-native HTTP (like aiohttp):
      Using aiohttp would require a different API and additional dependency.
      run_in_executor lets us reuse the existing requests Session (with connection
      pooling and the API key header already set) without any code changes.
      For I/O-bound work like HTTP, there's negligible performance difference.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: _get(path, params))


async def fetch_parallel(*fetch_fns) -> List[Any]:
    """
    Fires multiple async fetch coroutines simultaneously and returns all results.

    WHY asyncio.gather:
      gather() starts all coroutines at the same time and waits for all to finish.
      Sequential awaits would add each latency: 300ms + 300ms = 600ms.
      Parallel with gather: max(300ms, 300ms) = 300ms — roughly 2× faster.

    Example usage:
        courses, terms = await fetch_parallel(
            get_courses_async("CS"),
            get_terms_async(),
        )
    """
    return await asyncio.gather(*fetch_fns)


# ── Term code detection (cached) ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _current_term_code() -> str:
    """
    Finds and returns the term code for the most recently started term.

    WHY CACHED:
      This is called every time we look up courses or exams. Without caching,
      every course lookup would make an extra /Terms API call. With lru_cache,
      it's called once and the result is reused for the entire server lifetime.

    TERM CODE FORMAT:
      UW term codes are 4-digit numbers like "1265" (Spring 2026).
      We find terms whose start date ≤ today, then pick the code with the
      highest value (most recent started term).

    Returns:
        A term code string, or "1265" (Spring 2026) as a hardcoded fallback.
    """
    data = _get("/Terms")
    if isinstance(data, list):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Filter to terms that have already started
        started = [t for t in data if (t.get("termBeginDate") or "") <= today + "T"]
        if started:
            # Pick the one with the highest term code (most recent)
            best = sorted(started, key=lambda t: t.get("termCode", "0"), reverse=True)[0]
            return best.get("termCode", "1265")
    return "1265"   # Spring 2026 — hardcoded fallback if the API fails at startup


# ── Sync API functions ────────────────────────────────────────────────────────
# These are used by LangGraph nodes, synchronous scripts, and other non-async callers.

def get_courses(subject: str = "CS", term: Optional[str] = None) -> Any:
    """
    Fetches the course list for a given subject in the current (or specified) term.

    WHY WE SUMMARIZE TO 20 COURSES AND TRUNCATE DESCRIPTIONS:
      The full API response can contain 200+ courses with multi-paragraph descriptions.
      Passing all of this to Gemini would use thousands of tokens unnecessarily.
      We keep the 20 most relevant courses (first 20 returned, roughly ordered by
      catalog number) with descriptions truncated to 180 chars — enough for context.

    Args:
        subject: UW subject code (e.g. "CS", "ECE", "MATH"). Case-insensitive.
        term:    Override term code. If None, uses the current term.

    Returns:
        Dict with {term, subject, total, courses: [{code, title, units, description}]}
    """
    tc = term or _current_term_code()
    data = _get(f"/Courses/{tc}/{subject.upper()}")
    if isinstance(data, list):
        return {
            "term": tc,
            "subject": subject.upper(),
            "total": len(data),  # Include total so AI can say "X courses are offered"
            "courses": [
                {
                    # Combine subjectCode + catalogNumber into display code (e.g. "CS246")
                    "code": f"{c.get('subjectCode','')}{c.get('catalogNumber','')}",
                    "title": c.get("title", ""),
                    "units": c.get("units", ""),
                    # Truncate descriptions to keep prompt size manageable
                    "description": (c.get("description") or "")[:180],
                }
                for c in data[:20]  # Cap at 20 courses to avoid huge prompts
            ],
        }
    return data  # Return the error dict as-is if the request failed


def get_terms() -> Any:
    """
    Returns recent past and upcoming academic terms with their dates.

    WHY 4 PAST + 2 FUTURE:
      Students need recent history (to reference what they've completed) and
      near-future terms (to plan registration, co-op, etc.).
      Returning all 30+ terms from the API would be noisy and wasteful.
    """
    data = _get("/Terms")
    if isinstance(data, list):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Sort past terms newest-first, take 4
        past = sorted(
            [t for t in data if (t.get("termBeginDate", "")) <= today + "T"],
            key=lambda t: t["termCode"], reverse=True
        )[:4]
        # Sort upcoming terms oldest-first, take 2 (next two terms on the calendar)
        future = sorted(
            [t for t in data if (t.get("termBeginDate", "")) > today + "T"],
            key=lambda t: t["termCode"]
        )[:2]
        return past + future  # Combine: [most recent 4 past] + [next 2 upcoming]
    return data


def get_subjects() -> Any:
    """Returns the full list of UW subject codes and their names."""
    return _get("/Subjects")


def get_important_dates() -> Any:
    """Returns UW academic calendar important dates (add/drop deadlines, etc.)."""
    return _get("/ImportantDates")


def get_exams(term: Optional[str] = None) -> Any:
    """Returns the exam schedule for the current (or specified) term."""
    tc = term or _current_term_code()
    return _get(f"/ExamSchedules/{tc}")


def get_locations(query: Optional[str] = None) -> Any:
    """
    Returns campus building/room locations.
    If query is provided, searches for a specific location by name.
    """
    if query:
        return _get(f"/Locations/search/{query}")
    return _get("/Locations")


def get_food_outlets() -> Any:
    """Returns UW campus food service outlets and their current status."""
    return _get("/FoodServices/outlets")


def get_holidays() -> Any:
    """Returns paid university holidays (days when campus is officially closed)."""
    return _get("/HolidayDates/paidholidays")


def get_news(n: int = 6) -> Any:
    """Returns the n most recent UW news articles."""
    return _get(f"/Wcms/latestnews/{n}")


def get_events(n: int = 6) -> Any:
    """Returns the n most upcoming UW campus events."""
    return _get(f"/Wcms/latestevents/{n}")


def get_course_detail(subject: str, catalog: str, term: Optional[str] = None) -> Any:
    """Returns detailed information about a single specific course."""
    tc = term or _current_term_code()
    return _get(f"/Courses/{tc}/{subject.upper()}/{catalog}")


# ── Async versions ────────────────────────────────────────────────────────────
# Each async function is a direct mirror of its sync counterpart.
# The only difference: they use _aget() instead of _get().
# FastAPI route handlers and chat_agent.py use these to avoid blocking the event loop.

async def get_courses_async(subject: str = "CS", term: Optional[str] = None) -> Any:
    """Async version of get_courses."""
    tc = term or _current_term_code()
    data = await _aget(f"/Courses/{tc}/{subject.upper()}")
    if isinstance(data, list):
        return {
            "term": tc,
            "subject": subject.upper(),
            "total": len(data),
            "courses": [
                {
                    "code": f"{c.get('subjectCode','')}{c.get('catalogNumber','')}",
                    "title": c.get("title", ""),
                    "units": c.get("units", ""),
                    "description": (c.get("description") or "")[:180],
                }
                for c in data[:20]
            ],
        }
    return data


async def get_term_courses_async(term: str) -> Any:
    """
    Fetches ALL courses offered in a specific term code (e.g. '1265').
    
    WARNING: This can return 1000+ items. We summarize it by returning 
    a counts per subject so the AI can orient the student.
    """
    data = await _aget(f"/Courses/{term}")
    if isinstance(data, list):
        # Summarize by subject to keep the AI prompt manageable
        subject_counts = {}
        for c in data:
            s = c.get("subjectCode", "UNKNOWN")
            subject_counts[s] = subject_counts.get(s, 0) + 1
        
        return {
            "term": term,
            "total_courses": len(data),
            "subjects_offered": subject_counts,
            "sample_courses": [
                f"{c.get('subjectCode')}{c.get('catalogNumber')}: {c.get('title')}"
                for c in data[:15]
            ]
        }
    return data


async def get_terms_async() -> Any:
    data = await _aget("/Terms")
    if isinstance(data, list):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        past   = sorted([t for t in data if (t.get("termBeginDate","")) <= today+"T"],
                        key=lambda t: t["termCode"], reverse=True)[:4]
        future = sorted([t for t in data if (t.get("termBeginDate","")) >  today+"T"],
                        key=lambda t: t["termCode"])[:2]
        return past + future
    return data


async def get_exams_async(term: Optional[str] = None) -> Any:
    tc = term or _current_term_code()
    return await _aget(f"/ExamSchedules/{tc}")


async def get_locations_async(query: Optional[str] = None) -> Any:
    if query:
        return await _aget(f"/Locations/search/{query}")
    return await _aget("/Locations")


async def get_food_async() -> Any:
    return await _aget("/FoodServices/outlets")


async def get_holidays_async() -> Any:
    return await _aget("/HolidayDates/paidholidays")


async def get_news_async(n: int = 6) -> Any:
    return await _aget(f"/Wcms/latestnews/{n}")


async def get_events_async(n: int = 8) -> Any:
    return await _aget(f"/Wcms/latestevents/{n}")


async def get_posts_async(n: int = 8) -> Any:
    """Returns the n most recent blog posts from all UW WCMS sites."""
    return await _aget(f"/Wcms/latestposts/{n}")


async def get_wcms_sites_async() -> Any:
    """Returns all active and published WCMS sites."""
    return await _aget("/Wcms")


async def get_course_detail_async(subject: str, catalog: str, term: Optional[str] = None) -> Any:
    """Async version of get_course_detail."""
    tc = term or _current_term_code()
    return await _aget(f"/Courses/{tc}/{subject.upper()}/{catalog}")


async def get_subjects_async() -> Any:
    """Async version of get_subjects."""
    return await _aget("/Subjects")


async def get_important_dates_async() -> Any:
    """Async version of get_important_dates."""
    return await _aget("/ImportantDates")


async def get_academic_orgs_async() -> Any:
    """Returns all academic organizations (faculties, departments)."""
    return await _aget("/AcademicOrganizations")


async def get_academic_org_async(code: str) -> Any:
    """Returns a specific academic organization by its code (e.g. 'MAT', 'ART')."""
    return await _aget(f"/AcademicOrganizations/{code.upper()}")


async def get_class_schedules_async(subject: str, catalog: str, term: Optional[str] = None) -> Any:
    """Returns class schedule (section times, rooms, instructors) for a specific course."""
    tc = term or _current_term_code()
    return await _aget(f"/ClassSchedules/{tc}/{subject.upper()}/{catalog}")


async def get_subjects_by_org_async(org_code: str) -> Any:
    """Returns all subjects (e.g. CS, MATH, STAT) associated with an academic organization."""
    return await _aget(f"/Subjects/associatedto/{org_code.upper()}")


async def get_current_term_async() -> Any:
    """Returns the current term data."""
    return await _aget("/Terms/current")


# ── TOOL CATALOG ─────────────────────────────────────────────────────────────
# Gemini reads this catalog to contextually decide which API(s) to call.
# Each entry has: description (for the AI), fn (callable), params (what to extract).

TOOL_CATALOG = {
    "courses": {
        "description": "Get a list of courses offered in a subject (e.g. CS, MATH, BIOL, SOC) for a term. Use when asking about courses in a department or subject area.",
        "params": ["subject", "term_code"],
    },
    "course_detail": {
        "description": "Get detailed info about ONE specific course by subject and catalog number (e.g. CS 246, BIOL 130). Use when asking about a specific course.",
        "params": ["subject", "catalog_number", "term_code"],
    },
    "class_schedule": {
        "description": "Get the class schedule (section times, rooms, instructors) for a specific course. Use when asking about when/where a class meets.",
        "params": ["subject", "catalog_number", "term_code"],
    },
    "subjects": {
        "description": "Get the full list of all subject codes and their names (e.g. CS = Computer Science). Use when someone asks what subjects exist or what a code means.",
        "params": [],
    },
    "subjects_by_org": {
        "description": "Get all subjects that belong to an academic organization/faculty (e.g. all subjects under the Math faculty, or all Arts subjects). Use when asking about what a faculty offers.",
        "params": ["org_code"],
    },
    "academic_orgs": {
        "description": "Get all academic organizations (faculties, departments) at UWaterloo. Use when someone asks about faculties, departments, or organizational structure.",
        "params": [],
    },
    "academic_org_detail": {
        "description": "Get details about a specific academic organization by its code (e.g. MAT for Math faculty, ART for Arts). Use when asking about a specific faculty or department.",
        "params": ["org_code"],
    },
    "exams": {
        "description": "Get the exam schedule for a term. Use when asking about final exams, exam dates, or exam locations.",
        "params": ["term_code"],
    },
    "terms": {
        "description": "Get academic term dates (start, end, registration deadlines). Use when asking about term dates, when a semester starts/ends, or academic calendar.",
        "params": [],
    },
    "current_term": {
        "description": "Get the current active term info. Use when asking what term it currently is.",
        "params": [],
    },
    "important_dates": {
        "description": "Get important academic dates (add/drop deadlines, fee deadlines, convocation). Use when asking about deadlines or key academic dates.",
        "params": [],
    },
    "locations": {
        "description": "Search for campus buildings and rooms by name. Use when asking where a building is or about campus locations.",
        "params": ["query"],
    },
    "food": {
        "description": "Get campus food service outlets and their status. Use when asking about food, restaurants, cafeterias, or dining on campus.",
        "params": [],
    },
    "holidays": {
        "description": "Get university paid holidays (campus closure dates). Use when asking about holidays or when campus is closed.",
        "params": [],
    },
    "news": {
        "description": "Get the latest UWaterloo news articles from the WCMS. Use when asking about campus news, announcements, or recent happenings.",
        "params": [],
    },
    "events": {
        "description": "Get upcoming UWaterloo campus events from the WCMS. Use when asking about events, workshops, clubs, activities, social gatherings, or what's happening on campus. Also use to find event recommendations for the student based on their interests.",
        "params": [],
    },
    "posts": {
        "description": "Get the latest blog posts from all UWaterloo WCMS sites. Use when asking about blog posts, student life stories, research highlights, or campus community content. Good for surfacing interest-relevant social content.",
        "params": [],
    },
}


# ── Backwards-compatibility aliases ──────────────────────────────────────────
get_waterloo_courses         = get_courses
get_waterloo_terms           = get_terms
get_waterloo_subjects        = get_subjects
get_waterloo_important_dates = get_important_dates
get_waterloo_exams           = get_exams
get_waterloo_locations       = get_locations
get_waterloo_food_outlets    = get_food_outlets
get_waterloo_holidays        = get_holidays
get_waterloo_news            = get_news
get_waterloo_events          = get_events
get_course_details           = get_course_detail
