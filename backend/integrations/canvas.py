from canvasapi import Canvas  # Python wrapper for the Canvas LMS REST API
from backend.config import settings  # Import our project configuration
from typing import List, Dict  # For clear type descriptions

class CanvasClient:
    """
    CanvasClient handles interaction with the student's Learning Management System (LMS).
    It fetches assignments, deadlines, and course info.
    """
    def __init__(self):
        self.canvas = None
        # Only initialize if the student has provided an API key in their .env
        if settings.CANVAS_API_KEY:
            self.canvas = Canvas(settings.CANVAS_API_URL, settings.CANVAS_API_KEY)

    def get_upcoming_assignments(self) -> List[Dict]:
        """
        Fetches upcoming assignments for the user across all their active courses.
        """
        # If no API key is provided, return an empty list (Uni-OS falls back to generic mode)
        if not self.canvas:
            return []
        
        # Get the current logged-in student's account
        user = self.canvas.get_current_user()
        assignments = []
        # Get all courses where the student is currently enrolled
        courses = user.get_courses(enrollment_state="active")
        
        for course in courses:
            try:
                # Ask Canvas for assignments that are 'upcoming' (not past due)
                course_assignments = course.get_assignments(bucket="upcoming")
                for assign in course_assignments:
                    # Simplify the complex Canvas object into a simple dictionary for our frontend
                    assignments.append({
                        "course": course.name,
                        "name": assign.name,
                        "due_at": assign.due_at,
                        "points_possible": assign.points_possible,
                        "description": assign.description
                    })
            except Exception:
                # If a course is locked or lacks permission, just skip it gracefully
                continue
                
        return assignments

# Create a global client instance for the backend to use
canvas_client = CanvasClient()
