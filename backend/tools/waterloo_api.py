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
import json
import os                               # To read WATERLOO_API_KEY from environment
import re
from concurrent.futures import ThreadPoolExecutor  # Runs sync requests.get off the event loop
from datetime import datetime, timezone  # For comparing term start dates to today
from functools import lru_cache         # Caches _current_term_code() — called frequently
from typing import Any, Dict, List, Optional  # Type hints
from urllib.parse import quote

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


def _clean_location_query(query: str) -> str:
    """Normalize location search: remove possessives, extra spaces."""
    query = re.sub(r"'s\b", "", query)   # "dean's" → "dean"
    query = re.sub(r"\s+", " ", query).strip()
    return query


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
        # #region agent log
        try:
            _log = {"sessionId": "ab5517", "location": "waterloo_api._get", "message": "HTTP non-200", "data": {"path": path, "status_code": r.status_code}, "timestamp": __import__("time").time_ns() // 1_000_000, "hypothesisId": "B"}
            open("/Users/apple/UniOS-Hackathon/.cursor/debug-ab5517.log", "a").write(json.dumps(_log) + "\n")
        except Exception:
            pass
        # #endregion
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
    Query is cleaned (possessives stripped) and URL-encoded before use in path.
    """
    if query:
        cleaned = _clean_location_query(query)
        encoded = quote(cleaned, safe="")
        path = f"/Locations/search/{encoded}"
        # #region agent log
        try:
            _log = {"sessionId": "ab5517", "location": "waterloo_api.get_locations", "message": "locations path", "data": {"query": query, "cleaned": cleaned, "encoded": encoded, "path": path}, "timestamp": __import__("time").time_ns() // 1_000_000, "hypothesisId": "A"}
            open("/Users/apple/UniOS-Hackathon/.cursor/debug-ab5517.log", "a").write(json.dumps(_log) + "\n")
        except Exception:
            pass
        # #endregion
        return _get(path)
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
        cleaned = _clean_location_query(query)
        encoded = quote(cleaned, safe="")
        path = f"/Locations/search/{encoded}"
        # #region agent log
        try:
            _log = {"sessionId": "ab5517", "location": "waterloo_api.get_locations_async", "message": "locations path", "data": {"query": query, "cleaned": cleaned, "encoded": encoded, "path": path}, "timestamp": __import__("time").time_ns() // 1_000_000, "hypothesisId": "A"}
            open("/Users/apple/UniOS-Hackathon/.cursor/debug-ab5517.log", "a").write(json.dumps(_log) + "\n")
        except Exception:
            pass
        # #endregion
        return await _aget(path)
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


async def get_term_by_code_async(code: str) -> Any:
    """Returns term data for a specific term code (e.g. '1265')."""
    return await _aget(f"/Terms/{code}")


async def get_subject_by_code_async(code: str) -> Any:
    """Returns subject data for a specific subject code (e.g. 'CS', 'SOC')."""
    return await _aget(f"/Subjects/{code.upper()}")


async def get_food_franchises_async() -> Any:
    """Returns all food service franchise data (Tim Hortons, etc.)."""
    return await _aget("/FoodServices/franchises")


async def get_food_outlet_by_name_async(name: str) -> Any:
    """Returns a specific food outlet by name. User input is URL-encoded for the path."""
    encoded = quote(name, safe="")
    path = f"/FoodServices/outlets/{encoded}"
    # #region agent log
    try:
        _log = {"sessionId": "ab5517", "location": "waterloo_api.get_food_outlet_by_name_async", "message": "food outlet path", "data": {"name": name, "encoded": encoded, "path": path}, "timestamp": __import__("time").time_ns() // 1_000_000, "hypothesisId": "C"}
        open("/Users/apple/UniOS-Hackathon/.cursor/debug-ab5517.log", "a").write(json.dumps(_log) + "\n")
    except Exception:
        pass
    # #endregion
    return await _aget(path)


async def get_food_franchise_by_name_async(name: str) -> Any:
    """Returns a specific food franchise by name. User input is URL-encoded for the path."""
    encoded = quote(name, safe="")
    path = f"/FoodServices/franchises/{encoded}"
    # #region agent log
    try:
        _log = {"sessionId": "ab5517", "location": "waterloo_api.get_food_franchise_by_name_async", "message": "food franchise path", "data": {"name": name, "encoded": encoded, "path": path}, "timestamp": __import__("time").time_ns() // 1_000_000, "hypothesisId": "C"}
        open("/Users/apple/UniOS-Hackathon/.cursor/debug-ab5517.log", "a").write(json.dumps(_log) + "\n")
    except Exception:
        pass
    # #endregion
    return await _aget(path)


async def get_location_by_code_async(code: str) -> Any:
    """Returns a specific campus building by its location code (e.g. 'DC', 'MC')."""
    return await _aget(f"/Locations/{code.upper()}")


async def get_holidays_by_year_async(year: str) -> Any:
    """Returns paid holidays for a specific year."""
    return await _aget(f"/HolidayDates/paidholidays/{year}")


async def get_important_dates_by_year_async(year: str) -> Any:
    """Returns important academic dates for a specific academic year."""
    return await _aget(f"/ImportantDates/{year}")


async def get_scheduled_courses_async(term: Optional[str] = None) -> Any:
    """Returns all course IDs with active schedules in a given term."""
    tc = term or _current_term_code()
    return await _aget(f"/ClassSchedules/{tc}")


async def get_wcms_site_events_async(site_id: str, n: int = 10) -> Any:
    """Returns events from a specific WCMS site by site ID."""
    return await _aget(f"/Wcms/{site_id}/events")


async def get_wcms_site_posts_async(site_id: str, n: int = 10) -> Any:
    """Returns blog posts from a specific WCMS site by site ID."""
    return await _aget(f"/Wcms/{site_id}/posts")


async def get_wcms_site_news_async(site_id: str, n: int = 10) -> Any:
    """Returns news from a specific WCMS site by site ID."""
    return await _aget(f"/Wcms/{site_id}/news")


# ── TOOL CATALOG ─────────────────────────────────────────────────────────────
# Gemini reads this catalog to contextually decide which API(s) to call.
# Each entry has: description (for the AI), fn (callable), params (what to extract).

TOOL_CATALOG = {
    # ── Courses ──────────────────────────────────────────────────────────────
    "courses": {
        "description": "Get a list of courses offered in a subject (e.g. CS, MATH, BIOL, SOC) for a term. Use when asking about courses in a department or subject area.",
        "params": ["subject", "term_code"],
    },
    "course_detail": {
        "description": "Get detailed info about ONE specific course by subject and catalog number (e.g. CS 246, BIOL 130). Use when asking about a specific course.",
        "params": ["subject", "catalog_number", "term_code"],
    },
    # ── Class Schedules ───────────────────────────────────────────────────────
    "class_schedule": {
        "description": "Get the class schedule (section times, rooms, instructors) for a specific course. Use when asking about when/where a class meets.",
        "params": ["subject", "catalog_number", "term_code"],
    },
    "scheduled_courses": {
        "description": "Get all course IDs that have active class schedules in a term. Use when asking what courses are currently being scheduled or offered this term.",
        "params": ["term_code"],
    },
    # ── Subjects ──────────────────────────────────────────────────────────────
    "subjects": {
        "description": "Get the full list of all subject codes and their names (e.g. CS = Computer Science). Use when someone asks what subjects exist or what a code means.",
        "params": [],
    },
    "subject_detail": {
        "description": "Get info about a specific subject by its code (e.g. 'CS', 'SOC', 'KIN'). Use when asking what a specific subject code means or represents.",
        "params": ["subject_code"],
    },
    "subjects_by_org": {
        "description": "Get all subjects that belong to an academic organization/faculty (e.g. all subjects under the Math faculty, or all Arts subjects). Use when asking about what a faculty offers.",
        "params": ["org_code"],
    },
    # ── Academic Organizations ────────────────────────────────────────────────
    "academic_orgs": {
        "description": "Get all academic organizations (faculties, departments) at UWaterloo. Use when someone asks about faculties, departments, or organizational structure.",
        "params": [],
    },
    "academic_org_detail": {
        "description": "Get details about a specific academic organization by its code (e.g. MAT for Math faculty, ART for Arts, ENG for Engineering, AHS for Applied Health Sciences). Use when asking about a specific faculty or department.",
        "params": ["org_code"],
    },
    # ── Exams ─────────────────────────────────────────────────────────────────
    "exams": {
        "description": "Get the exam schedule for a term. Use when asking about final exams, exam dates, or exam locations.",
        "params": ["term_code"],
    },
    # ── Terms ─────────────────────────────────────────────────────────────────
    "terms": {
        "description": "Get academic term dates (start, end, registration deadlines). Use when asking about term dates, when a semester starts/ends, or academic calendar.",
        "params": [],
    },
    "current_term": {
        "description": "Get the current active term info. Use when asking what term it currently is.",
        "params": [],
    },
    "term_detail": {
        "description": "Get details for a specific term by its code (e.g. '1265' for Spring 2026). Use when asking about a particular semester by its code.",
        "params": ["term_code"],
    },
    # ── Important Dates ───────────────────────────────────────────────────────
    "important_dates": {
        "description": "Get important academic dates (add/drop deadlines, fee deadlines, convocation) for the current period. Use when asking about deadlines or key academic dates.",
        "params": [],
    },
    "important_dates_by_year": {
        "description": "Get important academic dates for a specific academic year. Use when someone asks about deadlines or key dates for a particular year (e.g. '2026').",
        "params": ["year"],
    },
    # ── Locations ─────────────────────────────────────────────────────────────
    "locations": {
        "description": "Search for campus buildings and rooms by name. Use when asking where a building is or about campus locations by name (e.g. 'Davis Centre', 'Math').",
        "params": ["query"],
    },
    "location_by_code": {
        "description": "Get a specific campus building by its exact building code (e.g. 'DC' for Davis Centre, 'MC' for Math Computer, 'STC' for Science). Use when you know the exact building code.",
        "params": ["location_code"],
    },
    # ── Food Services ─────────────────────────────────────────────────────────
    "food": {
        "description": "Get all campus food service outlets and their current status. Use when asking about food, restaurants, cafeterias, or dining on campus in general.",
        "params": [],
    },
    "food_by_name": {
        "description": "Get a specific campus food outlet by its name (e.g. 'Tim Hortons', 'Subway'). Use when asking about a specific restaurant or outlet on campus.",
        "params": ["outlet_name"],
    },
    "food_franchises": {
        "description": "Get all food service franchise chains on campus (e.g. Tim Hortons, Starbucks, Subway). Use when asking about chain restaurants or franchises on campus.",
        "params": [],
    },
    "food_franchise_by_name": {
        "description": "Get details about a specific food franchise chain by name. Use when asking about a specific chain restaurant like 'Tim Hortons' or 'Starbucks'.",
        "params": ["franchise_name"],
    },
    # ── Holidays ──────────────────────────────────────────────────────────────
    "holidays": {
        "description": "Get all university paid holidays (campus closure dates). Use when asking about holidays or when campus is closed.",
        "params": [],
    },
    "holidays_by_year": {
        "description": "Get paid holidays for a specific year. Use when asking about holidays in a particular year (e.g. 'holidays in 2026').",
        "params": ["year"],
    },
    # ── WCMS News / Events / Posts (all sites) ────────────────────────────────
    "news": {
        "description": "Get the latest UWaterloo news articles from all WCMS sites. Use when asking about campus news, announcements, or recent happenings.",
        "params": [],
    },
    "events": {
        "description": "Get upcoming events from all UWaterloo WCMS sites. Use when asking about events, workshops, clubs, activities, social gatherings, or what's happening on campus. Also use proactively when the student has interests that might match events.",
        "params": [],
    },
    "posts": {
        "description": "Get the latest blog posts from all UWaterloo WCMS sites. Use when asking about blog posts, student life stories, research highlights, or campus community content. Good for interest-based social recommendations.",
        "params": [],
    },
    # ── WCMS Site-Specific (by department/faculty site) ───────────────────────
    "site_events": {
        "description": "Get events from a specific UWaterloo department/faculty WCMS site by its site ID. Use when asking about events from a specific department or faculty (e.g. Math faculty events, CS department events).",
        "params": ["site_id"],
    },
    "site_posts": {
        "description": "Get blog posts from a specific UWaterloo department/faculty WCMS site by site ID. Use when asking about news or posts from a specific department.",
        "params": ["site_id"],
    },
    "site_news": {
        "description": "Get news from a specific UWaterloo department/faculty WCMS site by site ID. Use when asking about news from a specific department or faculty.",
        "params": ["site_id"],
    },
    "wcms_sites": {
        "description": "Get all active UWaterloo WCMS department/faculty sites and their IDs. Use this FIRST if you need a site ID before calling site_events, site_posts, or site_news.",
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
