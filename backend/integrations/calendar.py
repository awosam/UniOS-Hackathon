"""
calendar.py — Fuses Canvas assignment deadlines with student availability.

WHY THIS EXISTS:
  Students often have a list of upcoming deadlines and a rough sense of when
  they're free, but struggle to create a realistic study schedule. CalendarFusion
  takes both inputs and uses AI to create an optimized, burnout-aware plan.

  The word "fusion" comes from merging two data sources:
    1. Canvas deadlines (what needs to be done, by when)
    2. Free time slots (when the student can actually work)

THE AI'S ROLE:
  Scheduling is hard to do optimally by hand — you need to consider:
  - Which tasks are hardest and need the best mental energy (morning vs. evening)
  - How to spread work so it doesn't pile up before a deadline
  - Which tasks are longest and need multiple sessions
  The AI handles all of this naturally in a single prompt.

ASYNC DESIGN:
  generate_study_plan_async() is the primary method.
  generate_study_plan() is a sync wrapper for compatibility.
  See drafter.py comments for explanation of run_in_executor pattern.
"""

import asyncio                          # Async support for the event loop
from concurrent.futures import ThreadPoolExecutor  # Runs sync Vertex SDK off the event loop
from typing import List, Dict           # Type hints for assignments and free_slots

import vertexai
from vertexai.generative_models import GenerativeModel

from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION  # Shared GCP config

# Vertex AI init — runs once at module load
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# Thread pool: up to 4 concurrent study-plan generations
_executor = ThreadPoolExecutor(max_workers=4)

# Shared GenerativeModel — same instance reused for all calls (stateless, thread-safe)
_model = GenerativeModel("gemini-2.5-flash")


class CalendarFusion:
    """
    Creates personalized, AI-optimized study schedules.

    Input format for assignments (list of dicts):
        [{"title": "CS 246 A3", "due": "2026-03-20", "weight": "15%"}, ...]

    Input format for free_slots (list of dicts):
        [{"day": "Tuesday", "start": "14:00", "end": "17:00"}, ...]
    """

    async def generate_study_plan_async(
        self, assignments: List[Dict], free_slots: List[Dict]
    ) -> str:
        """
        Generates an optimized study schedule asynchronously.

        Args:
            assignments: Upcoming deadlines from Canvas (title, due date, weight).
            free_slots:  Student's available time windows (day, start time, end time).

        Returns:
            A formatted study plan as a string, with specific time blocks allocated
            to each assignment and brief reasoning for the scheduling decisions.
        """
        prompt = f"""You are the 'Logistics Strategist' for Uni-OS — a University of Waterloo academic planner.

Upcoming deadlines: {assignments}
Student's free time slots: {free_slots}

Create a detailed 'Study Fusion' plan following these principles:
1. Allocate harder/heavier-weight tasks to morning or early-afternoon slots
   (peak cognitive performance for most students).
2. Suggest specific blocks with exact times:
   e.g. "Tuesday 2:00pm–4:00pm: Work on CS 246 Assignment 3 (linked lists section)"
3. Spread work across multiple days to avoid last-minute all-nighters.
4. Leave at least one slot buffer before each deadline for review.
5. Briefly explain your scheduling reasoning at the end
   (why this order, why this timing) so the student understands the logic."""

        # Offload the blocking Vertex SDK call to the thread pool
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            _executor,
            lambda: _model.generate_content(prompt)
        )
        return resp.text

    def generate_study_plan(
        self, assignments: List[Dict], free_slots: List[Dict]
    ) -> str:
        """
        Synchronous wrapper around generate_study_plan_async.
        Useful for calling from non-async contexts (like scripts or tests).
        Do NOT call this from inside a running async event loop — use the async version.
        """
        return asyncio.run(self.generate_study_plan_async(assignments, free_slots))


# Single shared instance imported by main.py
calendar_fusion = CalendarFusion()
