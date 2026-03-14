"""
memory.py — Persistent student context store.

WHY THIS EXISTS:
  Large language models have no memory between conversations — every new message
  starts from scratch. MemoryEngine solves this by persisting important context
  (achievements, known struggles) to a JSON file on disk.

  This context is injected into every AI prompt so Gemini can say things like:
  "Last time you mentioned struggling with MATH 115..." — making the AI feel
  like a real academic companion rather than a generic chatbot.

WHY JSON AND NOT A DATABASE:
  For a hackathon with one user, a JSON file is perfectly adequate.
  It requires zero setup, is human-readable, and persists between server restarts.
  In a production app with multiple users this would be replaced with a real DB.

USAGE:
  memory_engine.save_milestone("Passed CS 136 with 88%")
  memory_engine.add_struggle("linear algebra", "struggled with eigenvalues")
  context = memory_engine.get_context_summary()  # injected into AI prompts
"""

import json          # Standard library — reads/writes the JSON storage file
import os            # Used to check if the file exists before reading
from datetime import datetime  # Timestamps each milestone with when it was recorded
from typing import Dict, List  # Type hints make the data structures self-documenting


class MemoryEngine:
    """
    Stores and retrieves persistent student context.

    Data is stored in a JSON file so it survives server restarts.
    The file has three keys:
      - "profile":    dict of personal details (name, major, year)
      - "milestones": list of achievements (e.g. "Passed CS 136 with 88%")
      - "struggles":  list of known difficulties (e.g. "eigenvalues in MATH 115")
    """

    def __init__(self, file_path: str = "backend/data/personal_memory.json"):
        # Where we persist memory between server restarts.
        # The path is relative to where the server is launched (uni_os/).
        self.file_path = file_path

        # Load any previously saved data immediately on startup
        self.memory = self._load_memory()
        
        # DEMO MODE: Wipe memory on every server restart.
        # This ensures every demo run starts with a blank slate.
        self.clear()

    def clear(self) -> None:
        """
        Wipes all in-memory data and deletes the JSON file from disk.
        Resets the student profile to the default 'blank slate'.
        """
        self.memory = {
            "profile": {"name": "Student", "major": "Unknown", "year": "Unknown"},
            "milestones": [], 
            "struggles": []
        }
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        self._persist()

    def _load_memory(self) -> Dict:
        """
        Reads the JSON file from disk and returns it as a Python dict.

        WHY CHECK os.path.exists FIRST:
          If the file doesn't exist yet (first run), json.load would raise a
          FileNotFoundError. We handle it gracefully by returning a blank template.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                return json.load(f)

        # First-run default — blank slate for a new student
        return {
            "profile": {"name": "Student", "major": "Unknown", "year": "Unknown"},
            "milestones": [], 
            "struggles": []
        }

    def save_milestone(self, description: str):
        """
        Records a student achievement with a timestamp.

        WHY KEEP A LIST (not replace):
          We accumulate milestones over time so the AI can reference growth.
          "You've gone from struggling with CS 136 to passing CS 246!" requires
          knowing both events.
        """
        self.memory["milestones"].append({
            "date": datetime.now().isoformat(),  # ISO format is human-readable and sortable
            "content": description,
        })
        self._persist()  # Save immediately so data isn't lost if server crashes

    def add_struggle(self, area: str, detail: str):
        """
        Records a subject area or concept the student finds difficult.

        This context makes the AI proactively supportive — e.g. if a student
        mentions struggling with eigenvalues, the AI remembers this for future
        MATH questions and offers extra encouragement.
        """
        self.memory["struggles"].append({"area": area, "detail": detail})
        self._persist()

    def learn_personal_info(self, updates: Dict[str, str]):
        """
        Updates the student's personal profile (name, major, year).
        
        Args:
            updates: A dictionary of fields to update (e.g. {"name": "Alice"})
        """
        for key, value in updates.items():
            if key in self.memory["profile"]:
                # Only update if the value is meaningful (not 'Unknown', 'Student', or empty)
                val_clean = str(value).strip().lower()
                if val_clean and val_clean not in ["unknown", "student", "none", "null"]:
                    self.memory["profile"][key] = value
        self._persist()

    def get_context_summary(self) -> str:
        """
        Produces a short text summary of recent context.
        Synchronizes with disk first to ensure it's not stale.
        """
        self.memory = self._load_memory() # Refresh from disk
        profile = self.memory["profile"]
        summary = f"Student Profile: {profile['name']} ({profile['year']} year {profile['major']} major)\n"
        summary += "Recent context:\n"

        if self.memory["milestones"]:
            # Take the 3 most recent milestones
            recent = [m["content"] for m in self.memory["milestones"][-3:]]
            summary += f"- Recent achievements: {', '.join(recent)}\n"

        if self.memory["struggles"]:
            # Take the 3 most recent struggle areas
            recent = [s["area"] for s in self.memory["struggles"][-3:]]
            summary += f"- Known struggles: {', '.join(recent)}\n"

        return summary

    def _persist(self) -> None:
        """
        Writes the current in-memory state back to the JSON file.

        Called after every mutation (save_milestone, add_struggle) so that
        a server restart never loses data. The indent=4 makes the JSON
        human-readable so a developer can manually inspect it if needed.
        """
        # Ensure the data/ directory exists (it won't on first run)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=4)


# Create one shared instance for the entire backend.
# All modules that need memory import this object: `from backend.agents.memory import memory_engine`
memory_engine = MemoryEngine()
