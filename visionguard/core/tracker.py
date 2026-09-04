"""
VisionGuard - Multi-Object Euclidean & Centroid Tracker
Maintains persistent track IDs, motion trajectories, velocity, and compass heading.
"""

import numpy as np
import math
import time
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
from scipy.spatial import distance as dist


class TrackedObject:
    def __init__(self, track_id: int, box: Tuple[int, int, int, int], centroid: Tuple[int, int], label: str = "object", confidence: float = 0.8):
        self.track_id = track_id
        self.box = box  # (x, y, w, h)
        self.centroid = centroid  # (cx, cy)
        self.label = label
        self.confidence = confidence
        self.disappeared = 0
        self.history = deque(maxlen=40)  # past (x, y, timestamp)
        self.history.append((centroid[0], centroid[1], time.time()))
        self.created_at = time.time()
        self.updated_at = time.time()
        self.velocity = (0.0, 0.0)  # (vx, vy) in px/sec
        self.speed = 0.0
        self.heading = "STATIONARY"  # N, S, E, W, etc.
        self.color = self._generate_color(track_id)
        self.counted_tripwires = set()  # set of tripwire_ids already counted for this track
        self.zone_entry_times = {}      # zone_id -> entry_timestamp
        self.zone_breached_alerts = set() # zone_id for which alert already emitted

    def _generate_color(self, track_id: int) -> Tuple[int, int, int]:
        # Generate visually distinct neon colors
        np.random.seed(track_id * 97)
        hue = (track_id * 47) % 180
        # Convert HSV (H, 220, 240) to BGR for OpenCV
        hsv_pixel = np.uint8([[[hue, 220, 240]]])
        import cv2
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def update(self, box: Tuple[int, int, int, int], centroid: Tuple[int, int], label: Optional[str] = None, confidence: Optional[float] = None):
        now = time.time()
        dt = now - self.updated_at

        if dt > 0.001 and len(self.history) > 0:
            last_cx, last_cy, _ = self.history[-1]
            vx = (centroid[0] - last_cx) / dt
            vy = (centroid[1] - last_cy) / dt
            # Exponential smoothing for velocity
            self.velocity = (0.6 * vx + 0.4 * self.velocity[0], 0.6 * vy + 0.4 * self.velocity[1])
            self.speed = math.hypot(self.velocity[0], self.velocity[1])
            self.heading = self._compute_heading(self.velocity[0], self.velocity[1])

        # Smooth bounding box
        old_x, old_y, old_w, old_h = self.box
        new_x, new_y, new_w, new_h = box
        smoothed_box = (
            int(0.3 * new_x + 0.7 * old_x),
            int(0.3 * new_y + 0.7 * old_y),
            int(0.3 * new_w + 0.7 * old_w),
            int(0.3 * new_h + 0.7 * old_h)
        )
        self.box = smoothed_box
        self.centroid = centroid
        self.history.append((centroid[0], centroid[1], now))
        self.disappeared = 0
        self.updated_at = now
        if label:
            self.label = label
        if confidence:
            self.confidence = confidence

    def _compute_heading(self, vx: float, vy: float) -> str:
        if math.hypot(vx, vy) < 10.0:
            return "STATIONARY"
        angle_deg = (math.degrees(math.atan2(vy, vx)) + 360) % 360
        # 0 = E, 90 = S, 180 = W, 270 = N
        dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        idx = int((angle_deg + 22.5) / 45.0) % 8
        return dirs[idx]


class ObjectTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: float = 90.0):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, box: Tuple[int, int, int, int], centroid: Tuple[int, int], label: str = "object", confidence: float = 0.8) -> TrackedObject:
        track = TrackedObject(self.next_track_id, box, centroid, label, confidence)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def deregister(self, track_id: int):
        if track_id in self.tracks:
            del self.tracks[track_id]

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Takes list of dicts with keys: 'box', 'centroid', 'label', 'confidence'
        Matches with active tracks and registers/deregisters objects.
        """
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    self.deregister(track_id)
            return list(self.tracks.values())

        input_centroids = np.array([d["centroid"] for d in detections], dtype=np.float32)
        input_boxes = [d["box"] for d in detections]
        input_labels = [d.get("label", "object") for d in detections]
        input_confs = [d.get("confidence", 0.8) for d in detections]

        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.register(input_boxes[i], tuple(input_centroids[i].astype(int)), input_labels[i], input_confs[i])
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids], dtype=np.float32)

            # Compute Euclidean distance matrix between active tracks and new input detections
            D = dist.cdist(track_centroids, input_centroids)

            # Find optimal matches by sorting distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                track_id = track_ids[row]
                self.tracks[track_id].update(
                    input_boxes[col],
                    tuple(input_centroids[col].astype(int)),
                    input_labels[col],
                    input_confs[col]
                )

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Check tracks that disappeared in this frame
            for row in unused_rows:
                track_id = track_ids[row]
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    self.deregister(track_id)

            # Register brand new tracks
            for col in unused_cols:
                self.register(
                    input_boxes[col],
                    tuple(input_centroids[col].astype(int)),
                    input_labels[col],
                    input_confs[col]
                )

        return list(self.tracks.values())
