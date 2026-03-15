"""
pathfinder.py — LangGraph agentic workflow for academic goal planning.

WHY LANGGRAPH:
  Some student requests require multi-step reasoning that a single AI prompt
  can't handle well. For example: "How do I switch from Math to CS?" requires:
    1. Retrieving the relevant UW transfer policies
    2. Synthesizing those policies with the student's specific grades
    3. Producing a step-by-step roadmap grounded in real rules

  LangGraph models this as a directed graph ("workflow") where each node
  is a function that reads from and writes to a shared "state" dict.
  This makes complex multi-step AI logic composable and debuggable.

THE WORKFLOW HAS TWO NODES:
  1. policy_retriever: Fetches relevant policy chunks from policy_decoder's store.
  2. roadmap_generator: Uses Gemini to synthesize those chunks into a roadmap.

  Data flows: input state → policy_retriever → roadmap_generator → final state

WHY NOT JUST ONE BIG PROMPT:
  Splitting into nodes means:
  - Each node has a clear, testable responsibility
  - The graph can be extended (add approval checks, email drafting, etc.)
  - LangGraph provides retry, branching, and parallelism support for free

HOW TO CALL:
  result = pathfinder_app.invoke({
      "goal": "Switch to Computer Science",
      "student_record": {"GPA": 82, "completed_terms": 2, "current_major": "Math"},
      "policies": [],   # filled by the graph
      "roadmap": [],    # filled by the graph
  })
  print(result["roadmap"])  # list of step strings
"""

import os
import sys

# Allow running this file directly with `python backend/agents/pathfinder.py`
# Without this, `from backend.agents...` imports would fail because Python
# wouldn't know where the `backend` package root is.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from typing import List, TypedDict  # TypedDict defines the shape of the shared graph state

import vertexai
from vertexai.generative_models import GenerativeModel
from langgraph.graph import StateGraph, END  # LangGraph core: graph builder and terminal node

from backend.agents.policy_decoder import policy_decoder  # RAG system for policy retrieval (PDF fallback)
from backend.tools.policy_index import search_policies as search_policies_index  # ChromaDB web-scraped policy
from backend.vertex import VERTEX_PROJECT, VERTEX_LOCATION  # Shared GCP config

# Initialize Vertex AI at module load (once per process)
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

# Shared model instance — stateless and thread-safe
_model = GenerativeModel("gemini-2.5-flash")


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared data structure that flows between nodes in the LangGraph workflow.

    WHY TYPEDDICT:
      TypedDict gives Python type hints for dict keys without the overhead of
      a dataclass. LangGraph requires the state to be a dict-like object.
      Every node receives the full state and returns updated fields.

    Fields:
        goal:           The student's academic objective in natural language.
        student_record: Academic context — GPA, completed courses, current program, etc.
        policies:       Policy text retrieved by policy_retriever (filled during run).
        roadmap:        Step-by-step action plan produced by roadmap_generator (filled during run).
    """
    goal: str
    student_record: dict
    policies: List[str]  # Populated by Node 1 (policy_retriever)
    roadmap: List[str]   # Populated by Node 2 (roadmap_generator)


# ── Node 1: Policy retrieval ──────────────────────────────────────────────────

def retrieve_policies(state: AgentState) -> AgentState:
    """
    Searches policy index (ChromaDB web-scraped) first; falls back to policy_decoder (PDF) if empty.
    Formats chunks with subsection-level citations: "Source: Section > Subsection — URL\\ntext".

    WHY RETRIEVAL BEFORE GENERATION:
      Asking Gemini to generate a roadmap without grounding it in policy text
      risks hallucination — the AI might invent rules that don't exist or get
      deadlines wrong. By retrieving real policy first, we constrain the AI to
      produce accurate, policy-backed steps.

    Args:
        state: Current graph state. Reads state["goal"], writes state["policies"].

    Returns:
        Updated state with policies filled in.
    """
    index_chunks = search_policies_index(state["goal"], k=3)
    if index_chunks:
        state["policies"] = [
            f"Source: {c['section']} > {c['subsection']} — {c['url']}\n{c['text']}"
            for c in index_chunks
        ]
        return state

    # Fallback: PDF-based policy_decoder (no subsection/url)
    chunks = policy_decoder.query_policies(state["goal"])
    state["policies"] = [f"[Source: {c.source}, page {c.page}]\n{c.text}" for c in chunks]
    return state


# ── Node 2: Roadmap generation ────────────────────────────────────────────────

def generate_roadmap(state: AgentState) -> AgentState:
    """
    Takes retrieved policy text + student context and generates an actionable roadmap.

    WHY NUMBERED STEPS:
      Students need a clear "do thing 1, then thing 2" sequence, not a paragraph.
      The prompt explicitly requests numbered steps so the output is structured.

    WHY HIGHLIGHT GPAAND DEADLINES:
      These are the most common failure points — students often miss that a
      program switch requires a minimum average, or that applications close in
      a specific month. Making these prominent prevents costly mistakes.

    Args:
        state: Current state. Reads goal, student_record, policies. Writes roadmap.

    Returns:
        Updated state with roadmap filled in as a list of strings (one per line).
    """
    prompt = f"""You are the 'Strategic Architect' for Uni-OS — a University of Waterloo academic planner.
Turn university bureaucracy into a clear, high-success roadmap for this student.

Student's goal: {state['goal']}
Student's academic context: {state['student_record']}
Relevant UW policy excerpts: {state['policies']}

Create a numbered step-by-step roadmap:
- Every step must be grounded in the provided policy excerpts.
- Highlight GPA requirements, application deadlines, and exact office names in BOLD.
- Be specific to UW (mention Quest, WaterlooWorks, Faculty offices by name).
- End with a "Key Risks" section listing the most common pitfalls to avoid."""

    response = _model.generate_content(prompt)

    # Split the AI's response into a list of lines for structured storage
    # The frontend or caller can then render each line as a list item
    state["roadmap"] = response.text.split("\n")
    return state


# ── Build and compile the LangGraph workflow ──────────────────────────────────

# Create a new graph with our AgentState as the shared data structure
workflow = StateGraph(AgentState)

# Register our two node functions with descriptive names
workflow.add_node("policy_retriever", retrieve_policies)
workflow.add_node("roadmap_generator", generate_roadmap)

# Define the execution order: start at policy_retriever, then go to roadmap_generator, then stop
workflow.set_entry_point("policy_retriever")           # Node 1 runs first
workflow.add_edge("policy_retriever", "roadmap_generator")  # Node 1 → Node 2
workflow.add_edge("roadmap_generator", END)            # Node 2 → done

# Compile the graph into a runnable object.
# pathfinder_app.invoke(state) runs the full two-node workflow synchronously.
pathfinder_app = workflow.compile()
