"""
VisionGuard - Analytics Core Unit Tests
Tests Motion Detection, Object Tracker, Tripwire Crossing, Polygon Intrusion, and Event Logger.
"""

import unittest
import numpy as np
import cv2
import time
import os

from visionguard.config import Point, IntrusionZoneConfig, TripwireConfig, ZoneAlertSeverity
from visionguard.core.motion_detector import MotionDetector
from visionguard.core.tracker import ObjectTracker, TrackedObject
from visionguard.core.detector import ObjectDetector
from visionguard.core.counting_engine import CountingEngine
from visionguard.core.intrusion_detector import IntrusionDetector
from visionguard.core.event_logger import EventLogger
from visionguard.core.video_source import VideoSourceManager


class TestAnalyticsCore(unittest.TestCase):

    def test_synthetic_video_source(self):
        """Verify synthetic video scene generator produces valid frames."""
        source = VideoSourceManager()
        source.set_source("simulation", "perimeter")
        ret, frame = source.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (540, 960, 3))
        source.release()

    def test_motion_detector(self):
        """Verify MOG2 background subtraction and motion contour extraction."""
        detector = MotionDetector(history=100, var_threshold=16)
        
        # Static background frames
        bg = np.zeros((300, 400, 3), dtype=np.uint8)
        for _ in range(15):
            detector.process_frame(bg, min_area=100)

        # Frame with a moving bright rectangle
        moving_frame = bg.copy()
        cv2.rectangle(moving_frame, (100, 100), (160, 160), (255, 255, 255), -1)

        mask, boxes, heatmap = detector.process_frame(moving_frame, min_area=200)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(heatmap)
        self.assertGreaterEqual(len(boxes), 1)
        self.assertGreater(detector.motion_percentage, 0.0)

    def test_object_tracker(self):
        """Verify centroid tracking, ID persistence, and trajectory calculation."""
        tracker = ObjectTracker(max_disappeared=5, max_distance=60.0)

        # Step 1: Detect object at (100, 100)
        det1 = [{"box": (90, 90, 20, 20), "centroid": (100, 100), "label": "person", "confidence": 0.9}]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        track_id = tracks1[0].track_id
        self.assertEqual(tracks1[0].label, "person")

        # Step 2: Object moves to (110, 100)
        time.sleep(0.01)
        det2 = [{"box": (100, 90, 20, 20), "centroid": (110, 100), "label": "person", "confidence": 0.92}]
        tracks2 = tracker.update(det2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, track_id)
        self.assertEqual(len(tracks2[0].history), 2)
        self.assertIn(tracks2[0].heading, ["E", "SE", "NE"])

    def test_tripwire_counting(self):
        """Verify vector line crossing detection and in/out tallies."""
        engine = CountingEngine()
        tripwire = TripwireConfig(
            id="test_line",
            name="Test Line",
            start=Point(x=0.0, y=0.5),
            end=Point(x=1.0, y=0.5),
            direction="BOTH"
        )
        engine.set_tripwires([tripwire])

        # Create simulated track moving downwards across y=0.5 (y=270 in 540px frame)
        track = TrackedObject(1, (100, 200, 30, 30), (100, 200), label="person", confidence=0.9)
        track.history.append((100, 200, time.time() - 0.1))
        track.history.append((100, 320, time.time()))  # Crossed from y=200 to y=320

        events = engine.process_tracks([track], frame_width=960, frame_height=540)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tripwire_id"], "test_line")
        self.assertIn(events[0]["direction"], ["IN", "OUT"])
        self.assertEqual(engine.counts["total_in"] + engine.counts["total_out"], 1)
        self.assertEqual(engine.counts["people_in"] + engine.counts["people_out"], 1)

    def test_polygon_intrusion_detection(self):
        """Verify point-in-polygon containment and dwell time breach logic."""
        detector = IntrusionDetector()
        zone = IntrusionZoneConfig(
            id="restricted_zone",
            name="Vault",
            polygon=[Point(x=0.2, y=0.2), Point(x=0.8, y=0.2), Point(x=0.8, y=0.8), Point(x=0.2, y=0.8)],
            severity=ZoneAlertSeverity.RESTRICTED,
            dwell_threshold_sec=1.0,
            enabled=True
        )
        detector.set_zones([zone])

        # Target inside zone (at 50% width, 50% height)
        track = TrackedObject(1, (450, 250, 30, 30), (480, 270), label="intruder", confidence=0.95)
        events, status = detector.process_tracks([track], frame_width=960, frame_height=540)

        self.assertTrue(status["restricted_zone"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "ZONE_INTRUSION")
        self.assertEqual(events[0]["severity"], "CRITICAL")
        self.assertTrue(events[0]["trigger_recording"])

    def test_event_logger_and_export(self):
        """Verify SQLite event persistence, query filters, and CSV export."""
        test_db = os.path.join(os.path.dirname(__file__), "test_events.db")
        if os.path.exists(test_db):
            os.remove(test_db)

        logger = EventLogger(db_path=test_db)
        logger.log_event(
            event_type="ZONE_INTRUSION",
            severity="CRITICAL",
            zone_id="zone_1",
            zone_name="Perimeter Gate",
            track_id=42,
            object_class="person",
            details="Critical intrusion breach detected"
        )

        events = logger.get_events(limit=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["track_id"], 42)
        self.assertEqual(events[0]["severity"], "CRITICAL")

        stats = logger.get_event_stats()
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["severity_counts"]["CRITICAL"], 1)

        csv_text = logger.export_csv()
        self.assertIn("ZONE_INTRUSION", csv_text)
        self.assertIn("Perimeter Gate", csv_text)

        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    unittest.main()
