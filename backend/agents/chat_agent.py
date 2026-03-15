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
from backend.agents.policy_decoder import policy_decoder
from backend.config import settings
from backend.tools.policy_index import search_policies as search_policies_index
from backend.tools.waterloo_api import (
    TOOL_CATALOG,
    fetch_parallel,
    get_courses_async,
    get_course_detail_async,
    get_class_schedules_async,
    get_scheduled_courses_async,
    get_subjects_async,
    get_subject_by_code_async,
    get_subjects_by_org_async,
    get_academic_orgs_async,
    get_academic_org_async,
    get_exams_async,
    get_terms_async,
    get_term_by_code_async,
    get_current_term_async,
    get_important_dates_async,
    get_important_dates_by_year_async,
    get_locations_async,
    get_location_by_code_async,
    get_food_async,
    get_food_franchises_async,
    get_food_outlet_by_name_async,
    get_food_franchise_by_name_async,
    get_holidays_async,
    get_holidays_by_year_async,
    get_news_async,
    get_events_async,
    get_posts_async,
    get_wcms_sites_async,
    get_wcms_site_events_async,
    get_wcms_site_posts_async,
    get_wcms_site_news_async,
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
        # Courses
        "courses": lambda p: get_courses_async(p.get("subject", "CS"), p.get("term_code")),
        "course_detail": lambda p: get_course_detail_async(p.get("subject", "CS"), p.get("catalog_number", "100"), p.get("term_code")),
        # Class Schedules
        "class_schedule": lambda p: get_class_schedules_async(p.get("subject", "CS"), p.get("catalog_number", "100"), p.get("term_code")),
        "scheduled_courses": lambda p: get_scheduled_courses_async(p.get("term_code")),
        # Subjects
        "subjects": lambda p: get_subjects_async(),
        "subject_detail": lambda p: get_subject_by_code_async(p.get("subject_code", "CS")),
        "subjects_by_org": lambda p: get_subjects_by_org_async(p.get("org_code", "MAT")),
        # Academic Organizations
        "academic_orgs": lambda p: get_academic_orgs_async(),
        "academic_org_detail": lambda p: get_academic_org_async(p.get("org_code", "MAT")),
        # Exams
        "exams": lambda p: get_exams_async(p.get("term_code")),
        # Terms
        "terms": lambda p: get_terms_async(),
        "current_term": lambda p: get_current_term_async(),
        "term_detail": lambda p: get_term_by_code_async(p.get("term_code", "1251")),
        # Important Dates
        "important_dates": lambda p: get_important_dates_async(),
        "important_dates_by_year": lambda p: get_important_dates_by_year_async(p.get("year", "2026")),
        # Locations
        "locations": lambda p: get_locations_async(p.get("query")),
        "location_by_code": lambda p: get_location_by_code_async(p.get("location_code", "DC")),
        # Food
        "food": lambda p: get_food_async(),
        "food_by_name": lambda p: get_food_outlet_by_name_async(p.get("outlet_name", "")),
        "food_franchises": lambda p: get_food_franchises_async(),
        "food_franchise_by_name": lambda p: get_food_franchise_by_name_async(p.get("franchise_name", "")),
        # Holidays
        "holidays": lambda p: get_holidays_async(),
        "holidays_by_year": lambda p: get_holidays_by_year_async(p.get("year", "2026")),
        # WCMS (all sites)
        "news": lambda p: get_news_async(8),
        "events": lambda p: get_events_async(12),
        "posts": lambda p: get_posts_async(12),
        # WCMS (site-specific)
        "wcms_sites": lambda p: get_wcms_sites_async(),
        "site_events": lambda p: get_wcms_site_events_async(p.get("site_id", "")),
        "site_posts": lambda p: get_wcms_site_posts_async(p.get("site_id", "")),
        "site_news": lambda p: get_wcms_site_news_async(p.get("site_id", "")),
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

            # Fallback: no API needed. Try policy RAG first for policy-like questions.
            policy_chunks = search_policies_index(user_input)
            if not policy_chunks:
                try:
                    _pd_chunks = policy_decoder.query_policies(user_input)
                    policy_chunks = [{"text": c.text, "section": getattr(c, "source", ""), "subsection": f"page {getattr(c, 'page', '')}", "url": ""} for c in _pd_chunks]
                except Exception:
                    policy_chunks = []
            if policy_chunks:
                trace_log["data_sources"] = ["policy (RAG)"]
                policy_context = "\n\n".join(
                    f"Source: {c.get('section', '')} > {c.get('subsection', '')} — {c.get('url', '')}\n{c.get('text', '')}"
                    for c in policy_chunks
                )
                response = await self._answer_with_policy(user_input, memory_ctx, policy_context)
            else:
                trace_log["data_sources"] = ["none (direct AI)"]
                response = await self._answer_directly(user_input, memory_ctx)

            # Also check interests for direct responses
            recommendation = await self._check_wcms_for_interests()
            if recommendation:
                response["text"] += f"\n\n💡 **You might like:** {recommendation}"

            return self._append_trace(response, trace_log)

        except Exception as e:
            print(f"[ChatAgent] Error: {e}")
            err_msg = str(e)[:300]
            help_text = (
                "\n\n**What you can do:**\n"
                "1. Run the policy index build once so policy questions get cited answers: "
                "`python -m backend.scripts.build_policy_index` (or POST /scrape-policies).\n"
                "2. If this keeps happening, check Vertex AI / Gemini configuration (project, credentials)."
            )
            return {"type": "text", "text": f"Something went wrong while processing your request. Please try again or rephrase.\n\n**Error:** {err_msg}{help_text}\n\n[Source: Uni-OS]"}

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

    async def _answer_with_policy(self, user_input: str, memory_ctx: str, chunks: List[Dict]) -> Dict:
        """Answer using retrieved UW policy excerpts; injects inline citations and sources footer."""
        context_str, citation_map = _format_policy_context(chunks)
        prompt = f"""You are Uni-OS, a University of Waterloo academic assistant.

Student context: {memory_ctx}
Student asked: "{user_input}"

Use the policy excerpts below to answer. After EVERY sentence that uses information from a source, place its citation number in square brackets like [1] or [2]. Multiple citations on one sentence are fine: [1][2].
Do NOT put all citations at the end — cite inline immediately after the relevant sentence.
Do NOT make up information not in the sources.

POLICY SOURCES:
{context_str}

Answer:"""
        raw_text = await self._ai(prompt)
        linked_text = _inject_citations(raw_text, citation_map)
        return {"type": "text", "text": linked_text, "citation_map": citation_map}


def _format_policy_context(chunks: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
    """
    Returns (context_string, citation_map).
    context_string: numbered chunks for the prompt.
    citation_map: {1: {url, label}, ...}
    """
    context_lines = []
    citation_map = {}
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section", "")
        subsection = chunk.get("subsection", "")
        url = chunk.get("url", "")
        label = f"{section} > {subsection}" if subsection else (section or "Policy")
        context_lines.append(f"[{i}] SOURCE: {label}\n{chunk.get('text', '')}")
        citation_map[i] = {"url": url, "label": label}
    return "\n\n".join(context_lines), citation_map


def _inject_citations(text: str, citation_map: Dict[int, Dict]) -> str:
    """Replace [1], [2] with markdown links; append deduplicated sources footer. Uses (?!\\]) so [1] inside [[1]](url) is not double-matched."""
    used_nums = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))

    def _replace(match):
        n = int(match.group(1))
        cite = citation_map.get(n)
        if not cite:
            return match.group(0)
        url, label = cite["url"], cite["label"]
        return f"[[{n}]]({url} \"{label}\")"

    linked = re.sub(r"\[(\d+)\](?!\])", _replace, text)
    if used_nums:
        sources = ["\n\n**Sources**"]
        for n in used_nums:
            cite = citation_map.get(n)
            if cite:
                sources.append(f"**{n}.** [{cite['label']}]({cite['url']})")
        linked += "\n".join(sources)
    return linked


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