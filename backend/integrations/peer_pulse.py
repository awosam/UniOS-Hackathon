import random  # For generating mockup data
from typing import Dict, List  # Type hints

class PeerPulse:
    """
    PeerPulse aggregates campus data to provide live 'vibes'.
    It helps students find the perfect study spot based on current capacity and noise levels.
    """
    
    def __init__(self):
        # Mock data representing various campus hotspots
        self.locations = {
            "Matheson Commons": {"base_capacity": 500, "vibe": "Productive Buzz"},
            "5th Floor Lounge": {"base_capacity": 50, "vibe": "Dead Silent"},
            "Student Union Hub": {"base_capacity": 300, "vibe": "Social & Loud"},
            "Library Sub-Level 2": {"base_capacity": 100, "vibe": "Deep Focus"}
        }

    def get_live_vibes(self) -> List[Dict]:
        """
        Simulates live occupancy and vibes for each location.
        In the future, this would be powered by real-time sensor data.
        """
        pulse = []
        for name, info in self.locations.items():
            # Generate a random occupancy percentage for the demo
            current_occupancy = random.randint(10, 100)
            pulse.append({
                "location": name,
                "occupancy_percent": current_occupancy,
                "vibe": info["vibe"],
                # Simple logic to determine if a place is 'packed' or 'quiet'
                "status": "Packed" if current_occupancy > 85 else "Quiet" if current_occupancy < 30 else "Balanced"
            })
        return pulse

# Global instance for the backend to use
peer_pulse = PeerPulse()
