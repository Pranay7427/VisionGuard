"""
VisionGuard - Bidirectional People & Vehicle Counting Engine
Implements 2D Line-Segment Vector Intersection and Directional Cross-Product Crossing Detection.
"""

import time
import math
from typing import List, Dict, Tuple, Any, Optional
from visionguard.config import TripwireConfig


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Tests if 3 points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Returns True if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class CountingEngine:
    def __init__(self):
        self.tripwires: Dict[str, TripwireConfig] = {}
        # Global tally
        self.counts = {
            "total_in": 0,
            "total_out": 0,
            "people_in": 0,
            "people_out": 0,
            "vehicles_in": 0,
            "vehicles_out": 0
        }
        # Per tripwire tally: line_id -> {'in': int, 'out': int, 'people': int, 'vehicles': int}
        self.line_stats: Dict[str, Dict[str, int]] = {}

    def set_tripwires(self, tripwires: List[TripwireConfig]):
        self.tripwires = {tw.id: tw for tw in tripwires}
        for tw in tripwires:
            if tw.id not in self.line_stats:
                self.line_stats[tw.id] = {"in": 0, "out": 0, "people": 0, "vehicles": 0}

    def process_tracks(self, tracks: List[Any], frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Evaluates track trajectories against active tripwires.
        Returns list of crossing event records.
        """
        events = []

        for track in tracks:
            if len(track.history) < 2:
                continue

            # Need previous point and current point
            curr_pt = (float(track.history[-1][0]), float(track.history[-1][1]))
            prev_pt = (float(track.history[-2][0]), float(track.history[-2][1]))

            # Check all active tripwires
            for tw_id, tw in self.tripwires.items():
                if not tw.enabled:
                    continue

                # Avoid duplicate counts for same track on same line in a short burst
                if tw_id in track.counted_tripwires:
                    continue

                # Denormalize tripwire coordinates if given in 0.0 - 1.0 range
                p1_x = tw.start.x * frame_width if tw.start.x <= 1.0 else tw.start.x
                p1_y = tw.start.y * frame_height if tw.start.y <= 1.0 else tw.start.y
                p2_x = tw.end.x * frame_width if tw.end.x <= 1.0 else tw.end.x
                p2_y = tw.end.y * frame_height if tw.end.y <= 1.0 else tw.end.y

                p1 = (p1_x, p1_y)
                p2 = (p2_x, p2_y)

                # Test intersection between trajectory segment and tripwire line
                if segments_intersect(prev_pt, curr_pt, p1, p2):
                    # Compute vector cross product to determine crossing direction
                    # Line vector P1 -> P2
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]

                    # Vector from P1 to current centroid
                    cross_curr = dx * (curr_pt[1] - p1[1]) - dy * (curr_pt[0] - p1[0])

                    direction = "IN" if cross_curr > 0 else "OUT"

                    # Check if line direction allows this crossing
                    if tw.direction != "BOTH" and tw.direction != direction:
                        continue

                    track.counted_tripwires.add(tw_id)

                    # Update statistics
                    is_person = track.label in ["person", "pedestrian", "guard", "intruder"]
                    is_vehicle = track.label in ["car", "truck", "bus", "motorbike", "vehicle"]

                    if direction == "IN":
                        self.counts["total_in"] += 1
                        if is_person:
                            self.counts["people_in"] += 1
                        elif is_vehicle:
                            self.counts["vehicles_in"] += 1
                        self.line_stats[tw_id]["in"] += 1
                    else:
                        self.counts["total_out"] += 1
                        if is_person:
                            self.counts["people_out"] += 1
                        elif is_vehicle:
                            self.counts["vehicles_out"] += 1
                        self.line_stats[tw_id]["out"] += 1

                    if is_person:
                        self.line_stats[tw_id]["people"] += 1
                    elif is_vehicle:
                        self.line_stats[tw_id]["vehicles"] += 1

                    events.append({
                        "event_type": "TRIPWIRE_CROSSED",
                        "tripwire_id": tw_id,
                        "tripwire_name": tw.name,
                        "track_id": track.track_id,
                        "direction": direction,
                        "label": track.label,
                        "confidence": track.confidence,
                        "box": track.box,
                        "centroid": track.centroid,
                        "timestamp": time.time()
                    })

        return events

    def get_summary(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.copy(),
            "line_stats": {k: v.copy() for k, v in self.line_stats.items()}
        }

    def reset(self):
        for k in self.counts:
            self.counts[k] = 0
        for tw_id in self.line_stats:
            self.line_stats[tw_id] = {"in": 0, "out": 0, "people": 0, "vehicles": 0}
