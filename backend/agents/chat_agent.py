"""
chat_agent.py — AI-First Contextual Router for Uni-OS.

ARCHITECTURE:
  Instead of keyword matching or regex, this agent gives Gemini a TOOL_CATALOG
  (plain-English descriptions of every available API) and lets Gemini reason
  about which tool(s) to call based on the student's question context.

FLOW:
  1. Student asks a question
  2. Gemini reads the tool catalog + student memory context
  3. Gemini returns which tool(s) to call and what parameters to pass
  4. Agent dynamically executes the selected tool(s) in parallel
  5. Gemini synthesizes the API data into a student-friendly response
  6. System Trace is appended for transparency
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import vertexai
from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions

from backend.agents.memory import memory_engine
from backend.config import settings
from backend.tools.waterloo_api import (
    TOOL_CATALOG,
    fetch_parallel,
    get_courses_async,
    get_course_detail_async,
    get_class_schedules_async,
    get_subjects_async,
    get_subjects_by_org_async,
    get_academic_orgs_async,
    get_academic_org_async,
    get_exams_async,
    get_terms_async,
    get_current_term_async,
    get_important_dates_async,
    get_locations_async,
    get_food_async,
    get_holidays_async,
    get_news_async,
    get_events_async,
    get_posts_async,
)

_executor = ThreadPoolExecutor(max_workers=8)


# ── Build the catalog prompt once at import time ─────────────────────────────

def _build_catalog_prompt() -> str:
    """Converts the TOOL_CATALOG dict into a numbered list for the AI prompt."""
    lines = []
    for i, (name, info) in enumerate(TOOL_CATALOG.items(), 1):
        params_str = ", ".join(info["params"]) if info["params"] else "none"
        lines.append(f"{i}. **{name}** (params: {params_str}): {info['description']}")
    return "\n".join(lines)

_CATALOG_PROMPT = _build_catalog_prompt()


# ── Contextual Intent Classification ─────────────────────────────────────────

async def _classify_intent(user_input: str, ai_fn, memory_ctx: str = "") -> Dict:
    """
    Gives Gemini the full tool catalog and lets it reason about which
    tool(s) to call. No keyword matching — pure contextual understanding.
    """
    prompt = f"""You are the intelligent router for Uni-OS, a University of Waterloo student assistant.

STUDENT CONTEXT:
{memory_ctx}

AVAILABLE TOOLS:
{_CATALOG_PROMPT}

STUDENT MESSAGE: "{user_input}"

YOUR TASK: Analyze the student's message and decide:
1. Does this question need data from the university API? If yes, pick the best tool(s).
2. What parameters should be passed to each tool?

RULES:
- If the question is general conversation, return {{"intent": "GENERAL"}}
- If the question needs API data, return {{"intent": "WATERLOO_DATA", "tools": [...]}}
- Each tool entry: {{"name": "<tool_name>", "params": {{...}}}}
- You may select MULTIPLE tools if the question requires combining data
- For "subject" params, use the official UW subject code (e.g. "CS" for Computer Science, "SOC" for Sociology, "BIOL" for Biology)
- For "org_code" params, use the faculty code (e.g. "MAT" for Mathematics, "ART" for Arts, "SCI" for Science, "ENG" for Engineering, "AHS" for Applied Health Sciences, "ENV" for Environment)
- For "catalog_number" params, use just the number (e.g. "246" not "CS246")
- If no term_code is specified, omit it (the system will use the current term)
- Think about what the student MEANS, not just what they literally say

Return ONLY valid JSON, no explanation.

EXAMPLES:
- "What CS courses are there?" → {{"intent": "WATERLOO_DATA", "tools": [{{"name": "courses", "params": {{"subject": "CS"}}}}]}}
- "Tell me about CS 246" → {{"intent": "WATERLOO_DATA", "tools": [{{"name": "course_detail", "params": {{"subject": "CS", "catalog_number": "246"}}}}]}}
- "What subjects does the Math faculty offer?" → {{"intent": "WATERLOO_DATA", "tools": [{{"name": "subjects_by_org", "params": {{"org_code": "MAT"}}}}]}}
- "When are exams and what term is it?" → {{"intent": "WATERLOO_DATA", "tools": [{{"name": "exams", "params": {{}}}}, {{"name": "current_term", "params": {{}}}}]}}
- "How are you?" → {{"intent": "GENERAL"}}"""

    try:
        raw = await ai_fn(prompt)
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"intent": "GENERAL"}


# ── Dynamic Tool Executor ─────────────────────────────────────────────────────

async def _execute_tools(tool_selections: List[Dict]) -> List[Tuple[str, Any]]:
    """
    Takes the AI's tool selections and dynamically calls the matching functions.
    Returns a list of (tool_name, result) tuples.
    """
    # Map tool names to their async functions
    _TOOL_FNS = {
        "courses": lambda p: get_courses_async(p.get("subject", "CS"), p.get("term_code")),
        "course_detail": lambda p: get_course_detail_async(p.get("subject", "CS"), p.get("catalog_number", "100"), p.get("term_code")),
        "class_schedule": lambda p: get_class_schedules_async(p.get("subject", "CS"), p.get("catalog_number", "100"), p.get("term_code")),
        "subjects": lambda p: get_subjects_async(),
        "subjects_by_org": lambda p: get_subjects_by_org_async(p.get("org_code", "MAT")),
        "academic_orgs": lambda p: get_academic_orgs_async(),
        "academic_org_detail": lambda p: get_academic_org_async(p.get("org_code", "MAT")),
        "exams": lambda p: get_exams_async(p.get("term_code")),
        "terms": lambda p: get_terms_async(),
        "current_term": lambda p: get_current_term_async(),
        "important_dates": lambda p: get_important_dates_async(),
        "locations": lambda p: get_locations_async(p.get("query")),
        "food": lambda p: get_food_async(),
        "holidays": lambda p: get_holidays_async(),
        "news": lambda p: get_news_async(8),
        "events": lambda p: get_events_async(12),
        "posts": lambda p: get_posts_async(12),
    }

    async def _call_one(tool: Dict) -> Tuple[str, Any]:
        name = tool.get("name", "")
        params = tool.get("params", {})
        fn = _TOOL_FNS.get(name)
        if fn:
            result = await fn(params)
            return (name, result)
        return (name, {"error": f"Unknown tool: {name}"})

    # Execute all selected tools in parallel
    results = await asyncio.gather(*[_call_one(t) for t in tool_selections])
    return list(results)


# ── Chat Agent ────────────────────────────────────────────────────────────────

class ChatAgent:
    def __init__(self):
        from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        self.model = GenerativeModel("gemini-2.5-flash")

    async def _ai(self, prompt: str) -> Optional[str]:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(_executor, lambda: self.model.generate_content(prompt))
        return resp.text if resp.candidates and resp.candidates[0].content.parts else None

    async def get_response(self, user_input: str) -> Dict:
        memory_ctx = memory_engine.get_context_summary() if hasattr(memory_engine, "get_context_summary") else "No history yet."
        # Run memory janitor and proactive recommender in background
        asyncio.create_task(self._update_memory_background(user_input))

        try:
            # Step 1: Let Gemini reason about which tools to call
            classification = await _classify_intent(user_input, self._ai, memory_ctx)
            intent = classification.get("intent", "GENERAL")

            trace_log = {
                "memory_context": memory_ctx.replace("\n", " ").replace("Student Profile:", "").strip(),
                "router_intent": intent,
                "tools_selected": classification.get("tools", []),
            }

            if intent == "WATERLOO_DATA" and classification.get("tools"):
                # Step 2: Execute all selected tools in parallel
                tool_results = await _execute_tools(classification["tools"])

                # Build a combined data payload for Gemini
                combined_data = {}
                tool_labels = []
                for name, result in tool_results:
                    combined_data[name] = result
                    tool_labels.append(name)

                trace_log["data_sources"] = tool_labels
                raw_str = json.dumps(combined_data, default=str)
                trace_log["raw_data_snippet"] = raw_str[:300] + "..." if len(raw_str) > 300 else raw_str

                # Step 3: Let Gemini synthesize the data into a response
                response = await self._format_response(user_input, combined_data, tool_labels, memory_ctx)

                # Step 4: Proactively recommend WCMS content if user has interests
                recommendation = await self._check_wcms_for_interests()
                if recommendation:
                    response["text"] += f"\n\n💡 **You might like:** {recommendation}"

                return self._append_trace(response, trace_log)

            # Fallback: no API needed, answer directly
            trace_log["data_sources"] = ["none (direct AI)"]
            response = await self._answer_directly(user_input, memory_ctx)

            # Also check interests for direct responses
            recommendation = await self._check_wcms_for_interests()
            if recommendation:
                response["text"] += f"\n\n💡 **You might like:** {recommendation}"

            return self._append_trace(response, trace_log)

        except Exception as e:
            print(f"[ChatAgent] Error: {e}")
            return {"type": "text", "text": "Something went wrong while processing your request. Please try again or rephrase. [Source: Uni-OS]"}

    async def _format_response(self, user_input: str, data: Dict, labels: List[str], memory_ctx: str) -> Dict:
        """Lets Gemini read the raw API data and craft a student-friendly answer."""
        data_str = json.dumps(data, indent=2, default=str)[:8000]
        label_str = ", ".join(labels)

        prompt = f"""You are Uni-OS, a smart and friendly University of Waterloo academic assistant.

Student context: {memory_ctx}
Student asked: "{user_input}"

I fetched data from these sources: {label_str}
Here is the raw data:
{data_str}

YOUR TASK:
- Answer the student's question using ONLY the data provided above
- Format your response with clear markdown (headers, bullets, bold for course codes)
- If data is missing or an error occurred, acknowledge it honestly
- Be concise but comprehensive
- End with [Source: UWaterloo OpenData]"""

        text = await self._ai(prompt)
        return {"type": "text", "text": text}

    async def _answer_directly(self, user_input: str, memory_ctx: str) -> Dict:
        prompt = f"""You are Uni-OS, a smart and friendly University of Waterloo academic assistant.
Student context: {memory_ctx}
Student asked: "{user_input}"

Answer helpfully. If you don't have specific UW data, say so and suggest where they can find it."""
        text = await self._ai(prompt)
        return {"type": "text", "text": text}

    def _append_trace(self, response: Dict, trace_log: Dict) -> Dict:
        response["text"] += f"\n\n---\n**🔍 System Trace:**\n```json\n{json.dumps(trace_log, indent=2)}\n```"
        return response

    async def _update_memory_background(self, user_input: str):
        """
        The 'Memory Janitor' — extracts personal details AND interests from conversation.
        Runs in the background so it doesn't affect chat latency.
        """
        prompt = f"""Extract any new student personal information from this message.
Message: "{user_input}"

Look for:
1. Name (e.g. "I'm Alice")
2. Major (e.g. "I'm in Kinesiology")
3. Year (e.g. "I'm a 1st year")
4. Milestone (e.g. "I just passed BIOL 130")
5. Struggle (e.g. "I'm failing calculus")
6. Interests/hobbies (e.g. "I love chess", "I play soccer", "I'm into photography")
   - Extract each interest as a SHORT word or phrase (e.g. "chess", "soccer", "photography")
   - Only include genuine hobbies/interests, not academic subjects

Return ONLY a JSON object with these keys: "name", "major", "year", "milestone", "struggle", "interests".
For "interests", return a list of strings (e.g. ["chess", "hiking"]) or an empty list [].
Use "null" for scalar fields not found.
Example: {{"name": "Alice", "major": "Software Engineering", "year": "2A", "milestone": null, "struggle": null, "interests": ["chess", "hiking"]}}"""

        result = await self._ai(prompt)
        if not result:
            return

        try:
            clean_json = result.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            profile_updates = {}
            for key in ["name", "major", "year"]:
                val = str(data.get(key)).strip().lower()
                if val and val not in ["null", "none", "unknown", "student"]:
                    profile_updates[key] = data[key]

            if profile_updates:
                memory_engine.learn_personal_info(profile_updates)

            if data.get("milestone") and data["milestone"] != "null":
                memory_engine.save_milestone(data["milestone"])

            if data.get("struggle") and data["struggle"] != "null":
                memory_engine.add_struggle("General", data["struggle"])

            # Store interests/hobbies for WCMS personalized recommendations
            for interest in (data.get("interests") or []):
                if interest and str(interest).lower() not in ["null", "none", ""]:
                    memory_engine.add_interest(interest)

        except Exception as e:
            print(f"[MemoryJanitor] Failed to extract info: {e}")

    async def _check_wcms_for_interests(self) -> Optional[str]:
        """
        Proactive WCMS Recommender — runs after every response.
        Fetches the latest events and blog posts, then asks Gemini if any
        of them match the student's stored interests.
        Returns a short 1-sentence recommendation string, or None.
        """
        interests = memory_engine.get_interests()
        if not interests:
            return None  # No interests stored yet — skip

        try:
            # Fetch events and posts in parallel
            events_data, posts_data = await asyncio.gather(
                get_events_async(12),
                get_posts_async(12),
            )

            # Summarize for the AI (keep it short)
            def _summarize(items, key_field="title", date_field=None):
                if not isinstance(items, list):
                    return "(unavailable)"
                lines = []
                for item in items[:10]:
                    title = item.get(key_field, item.get("name", ""))
                    date = item.get(date_field, "") if date_field else ""
                    lines.append(f"- {title}" + (f" ({date[:10]})" if date else ""))
                return "\n".join(lines) if lines else "(none)"

            events_str = _summarize(events_data, "title", "startDate")
            posts_str = _summarize(posts_data, "title", "postedDate")

            prompt = f"""The student's known interests are: {', '.join(interests)}.

Upcoming UWaterloo events:
{events_str}

Recent UWaterloo blog posts:
{posts_str}

Do any of these events or posts relate to the student's interests?
- If YES: return ONE short, friendly sentence recommending the most relevant one. Example: "There's an upcoming Chess Club tournament on March 20th — you might enjoy it!"
- If NO match exists: return exactly the word NONE.

Return ONLY the sentence or NONE."""

            result = await self._ai(prompt)
            if result and result.strip().upper() != "NONE" and len(result.strip()) > 5:
                return result.strip()
        except Exception as e:
            print(f"[WCMS Recommender] Error: {e}")

        return None

chat_agent = ChatAgent()