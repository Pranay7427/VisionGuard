"""
VisionGuard - Master Video Analytics Pipeline Orchestrator
Coordinates Frame Capture, Motion Subtraction, Object Tracking, Counting, Zone Intrusion, HUD Overlays, and Streaming.
"""

import cv2
import numpy as np
import time
import threading
import psutil
from typing import Dict, Any, Tuple, Optional, List

from visionguard.config import AnalyticsConfig, IntrusionZoneConfig, TripwireConfig, Point, ZoneAlertSeverity
from visionguard.core.video_source import VideoSourceManager
from visionguard.core.motion_detector import MotionDetector
from visionguard.core.tracker import ObjectTracker, TrackedObject
from visionguard.core.detector import ObjectDetector
from visionguard.core.counting_engine import CountingEngine
from visionguard.core.intrusion_detector import IntrusionDetector
from visionguard.core.recorder import EventRecorder
from visionguard.core.event_logger import EventLogger


class VisionGuardPipeline:
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        
        # Initialize sub-modules
        self.source = VideoSourceManager(target_fps=30)
        self.motion_detector = MotionDetector(history=500, var_threshold=25, detect_shadows=True)
        self.tracker = ObjectTracker(max_disappeared=15, max_distance=100.0)
        self.detector = ObjectDetector(model_type=self.config.detector_model, conf_threshold=self.config.confidence_threshold)
        self.counting_engine = CountingEngine()
        self.intrusion_detector = IntrusionDetector()
        self.recorder = EventRecorder(pre_roll_sec=self.config.pre_roll_sec, post_roll_sec=self.config.post_roll_sec, fps=self.config.recording_fps)
        self.logger = EventLogger()

        # Seed default default sample zones and tripwires
        self._init_default_zones_and_lines()

        # Pipeline Runtime State
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.frame_lock = threading.Lock()

        # Performance & Telemetry
        self.fps_processing = 0.0
        self.cpu_usage_pct = 0.0
        self.ram_usage_pct = 0.0
        self.active_tracks_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_telemetry_time = time.time()
        self.recent_events_cache: List[Dict[str, Any]] = []

    def _init_default_zones_and_lines(self):
        """Initializes balanced default demo zones and tripwires."""
        default_zones = [
            IntrusionZoneConfig(
                id="zone_restricted_perimeter",
                name="Restricted Courtyard",
                polygon=[
                    Point(x=0.36, y=0.28),
                    Point(x=0.72, y=0.28),
                    Point(x=0.72, y=0.74),
                    Point(x=0.36, y=0.74)
                ],
                color="#ef4444",
                severity=ZoneAlertSeverity.RESTRICTED,
                dwell_threshold_sec=2.0,
                enabled=True
            )
        ]
        default_tripwires = [
            TripwireConfig(
                id="tripwire_entry_gate",
                name="Main Crossing Line",
                start=Point(x=0.10, y=0.50),
                end=Point(x=0.90, y=0.50),
                color="#00f2fe",
                direction="BOTH",
                enabled=True
            )
        ]
        self.config.zones = default_zones
        self.config.tripwires = default_tripwires
        self.intrusion_detector.set_zones(default_zones)
        self.counting_engine.set_tripwires(default_tripwires)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
        self.source.release()

    def update_config(self, new_config: AnalyticsConfig):
        self.config = new_config
        self.motion_detector.update_parameters(new_config.motion_sensitivity, new_config.min_contour_area)
        self.detector.model_type = new_config.detector_model
        self.detector.conf_threshold = new_config.confidence_threshold
        self.intrusion_detector.set_zones(new_config.zones)
        self.counting_engine.set_tripwires(new_config.tripwires)
        self.recorder.pre_roll_sec = new_config.pre_roll_sec
        self.recorder.post_roll_sec = new_config.post_roll_sec

    def _processing_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            loop_start = time.time()

            # 1. Grab raw frame from video source
            ret, frame = self.source.read_frame()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            self.latest_raw_frame = frame.copy()

            # 2. Push frame to recorder circular buffer
            self.recorder.push_frame(frame)

            # 3. Motion Detection
            motion_boxes = []
            fg_mask = None
            colored_heatmap = None
            if self.config.motion_detection_enabled or self.config.show_motion_mask or self.config.show_heatmap:
                fg_mask, motion_boxes, colored_heatmap = self.motion_detector.process_frame(
                    frame, min_area=self.config.min_contour_area
                )

            # 4. Object Detection (DNN, HOG or Geometric Classifier)
            detections = []
            if self.config.people_counting_enabled or self.config.vehicle_counting_enabled or self.config.object_tracking_enabled:
                detections = self.detector.detect(frame, motion_boxes)

            # 5. Multi-Object Tracking
            active_tracks = []
            if self.config.object_tracking_enabled:
                active_tracks = self.tracker.update(detections)
                self.active_tracks_count = len(active_tracks)

            # 6. Tripwire Counting
            crossing_events = []
            if self.config.people_counting_enabled or self.config.vehicle_counting_enabled:
                crossing_events = self.counting_engine.process_tracks(active_tracks, w, h)
                for ev in crossing_events:
                    # Capture snapshot if configured
                    snap_info = None
                    if self.config.show_telemetry_hud:
                        snap_info = self.recorder.capture_snapshot(
                            frame,
                            event_type=f"TRIPWIRE {ev['direction']}",
                            metadata={"tripwire_name": ev["tripwire_name"], "track_id": ev["track_id"], "severity": "INFO"}
                        )

                    # Log to database
                    logged = self.logger.log_event(
                        event_type="TRIPWIRE_CROSSED",
                        severity="INFO",
                        tripwire_id=ev["tripwire_id"],
                        tripwire_name=ev["tripwire_name"],
                        track_id=ev["track_id"],
                        object_class=ev["label"],
                        confidence=ev["confidence"],
                        snapshot_url=snap_info["url"] if snap_info else None,
                        details=f"Object #{ev['track_id']} ({ev['label']}) crossed {ev['tripwire_name']} ({ev['direction']})"
                    )
                    self._add_recent_event(logged)

            # 7. Region Intrusion Detection
            intrusion_events = []
            zone_alert_status = {}
            if self.config.intrusion_detection_enabled:
                intrusion_events, zone_alert_status = self.intrusion_detector.process_tracks(active_tracks, w, h)
                for ev in intrusion_events:
                    snap_info = None
                    rec_id = None

                    # Trigger snapshot
                    if ev.get("trigger_snapshot", False):
                        snap_info = self.recorder.capture_snapshot(
                            frame,
                            event_type=ev["event_type"],
                            metadata={"zone_name": ev["zone_name"], "track_id": ev["track_id"], "severity": ev["severity"]}
                        )

                    # Trigger auto-recording
                    if self.config.auto_recording_enabled and ev.get("trigger_recording", False):
                        rec_id = self.recorder.trigger_auto_recording(ev["event_type"], ev["zone_name"])

                    # Log event
                    logged = self.logger.log_event(
                        event_type=ev["event_type"],
                        severity=ev["severity"],
                        zone_id=ev["zone_id"],
                        zone_name=ev["zone_name"],
                        track_id=ev["track_id"],
                        object_class=ev["label"],
                        confidence=ev["confidence"],
                        snapshot_url=snap_info["url"] if snap_info else None,
                        recording_url=f"/recordings/{rec_id}.mp4" if rec_id else None,
                        details=f"Intrusion breach in '{ev['zone_name']}' by #{ev['track_id']} ({ev['label']}), dwell: {ev.get('dwell_time_sec', 0)}s"
                    )
                    self._add_recent_event(logged)

            # 8. Render HUD Annotations & Cyber Visuals
            annotated = self._render_hud(frame, fg_mask, colored_heatmap, active_tracks, zone_alert_status)

            # 9. Encode to JPEG for fast MJPEG streaming
            _, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with self.frame_lock:
                self.latest_annotated_frame = annotated
                self.latest_jpeg_bytes = jpeg.tobytes()

            # 10. Frame rate & system metrics regulation
            fps_counter += 1
            self.frame_count += 1
            if time.time() - fps_timer >= 1.0:
                self.fps_processing = round(fps_counter / (time.time() - fps_timer), 1)
                fps_counter = 0
                fps_timer = time.time()
                try:
                    self.cpu_usage_pct = round(psutil.cpu_percent(), 1)
                    self.ram_usage_pct = round(psutil.virtual_memory().percent, 1)
                except Exception:
                    pass

            # Maintain smooth processing target
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    def _render_hud(
        self,
        frame: np.ndarray,
        fg_mask: Optional[np.ndarray],
        heatmap: Optional[np.ndarray],
        tracks: List[TrackedObject],
        zone_alert_status: Dict[str, bool]
    ) -> np.ndarray:
        """Renders cyber-security overlays, bounding boxes, trajectories, zones, and telemetry."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Background Mask / Heatmap blending
        if self.config.show_heatmap and heatmap is not None:
            cv2.addWeighted(heatmap, 0.45, canvas, 0.55, 0, canvas)
        elif self.config.show_motion_mask and fg_mask is not None:
            mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0 # Green tint
            cv2.addWeighted(mask_colored, 0.35, canvas, 0.65, 0, canvas)

        # 2. Render Intrusion Zones
        if self.config.show_zones:
            for zone in self.config.zones:
                if not zone.enabled or len(zone.polygon) < 3:
                    continue

                pts = []
                for pt in zone.polygon:
                    px = int(pt.x * w if pt.x <= 1.0 else pt.x)
                    py = int(pt.y * h if pt.y <= 1.0 else pt.y)
                    pts.append([px, py])
                poly_np = np.array(pts, dtype=np.int32)

                is_breached = zone_alert_status.get(zone.id, False)

                # Zone Colors
                if is_breached:
                    # Flashing warning neon red
                    color_bgr = (40, 40, 245)
                    fill_alpha = 0.35
                else:
                    color_bgr = (50, 200, 70) if zone.severity == ZoneAlertSeverity.MONITORED else (40, 160, 240) if zone.severity == ZoneAlertSeverity.CAUTION else (40, 70, 230)
                    fill_alpha = 0.15

                # Semi-transparent polygon fill
                zone_overlay = canvas.copy()
                cv2.fillPoly(zone_overlay, [poly_np], color_bgr)
                cv2.addWeighted(zone_overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)

                # Border with glowing outline
                cv2.polylines(canvas, [poly_np], True, color_bgr, 2, cv2.LINE_AA)

                # Label tag
                tag_x, tag_y = pts[0][0], pts[0][1] - 8
                label_text = f"ZONE: {zone.name.upper()}"
                if is_breached:
                    label_text += " [BREACH ALERT!]"
                cv2.putText(canvas, label_text, (tag_x, max(20, tag_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Render Tripwire Lines
        if self.config.show_tripwires:
            for tw in self.config.tripwires:
                if not tw.enabled:
                    continue
                p1_x = int(tw.start.x * w if tw.start.x <= 1.0 else tw.start.x)
                p1_y = int(tw.start.y * h if tw.start.y <= 1.0 else tw.start.y)
                p2_x = int(tw.end.x * w if tw.end.x <= 1.0 else tw.end.x)
                p2_y = int(tw.end.y * h if tw.end.y <= 1.0 else tw.end.y)

                # Glowing tripwire line (Cyan #00f2fe = BGR (254, 242, 0))
                cv2.line(canvas, (p1_x, p1_y), (p2_x, p2_y), (254, 242, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (p1_x, p1_y), 5, (254, 242, 0), -1)
                cv2.circle(canvas, (p2_x, p2_y), 5, (254, 242, 0), -1)

                # Midpoint arrow and label
                mid_x = (p1_x + p2_x) // 2
                mid_y = (p1_y + p2_y) // 2
                cv2.putText(canvas, f"LINE: {tw.name} ({tw.direction})", (mid_x - 40, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (254, 242, 0), 1, cv2.LINE_AA)

        # 4. Render Tracks, Trajectories, and Bounding Boxes
        for trk in tracks:
            bx, by, bw, bh = trk.box
            cx, cy = trk.centroid

            # Draw trajectory path history
            if self.config.show_tracking_trails and len(trk.history) > 1:
                hist_pts = [(int(pt[0]), int(pt[1])) for pt in trk.history]
                for i in range(1, len(hist_pts)):
                    alpha = i / float(len(hist_pts))
                    thickness = max(1, int(alpha * 3))
                    cv2.line(canvas, hist_pts[i - 1], hist_pts[i], trk.color, thickness, cv2.LINE_AA)

            # Draw target bounding box
            if self.config.show_bounding_boxes:
                # Corner bracket style for cyber aesthetic
                line_len = min(15, bw // 3, bh // 3)
                color = trk.color

                # Thin bounding box
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 1, cv2.LINE_AA)
                
                # Highlighted corners
                cv2.line(canvas, (bx, by), (bx + line_len, by), color, 2)
                cv2.line(canvas, (bx, by), (bx, by + line_len), color, 2)
                cv2.line(canvas, (bx + bw, by), (bx + bw - line_len, by), color, 2)
                cv2.line(canvas, (bx + bw, by), (bx + bw, by + line_len), color, 2)
                cv2.line(canvas, (bx, by + bh), (bx + line_len, by + bh), color, 2)
                cv2.line(canvas, (bx, by + bh), (bx, by + bh - line_len), color, 2)
                cv2.line(canvas, (bx + bw, by + bh), (bx + bw - line_len, by + bh), color, 2)
                cv2.line(canvas, (bx + bw, by + bh), (bx + bw, by + bh - line_len), color, 2)

                # Centroid crosshair
                cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)

            # Draw track label badge
            if self.config.show_track_ids:
                badge_text = f"#{trk.track_id} {trk.label.upper()} {int(trk.confidence * 100)}%"
                badge_w = len(badge_text) * 8 + 10
                cv2.rectangle(canvas, (bx, max(0, by - 20)), (bx + badge_w, max(20, by)), (15, 20, 25), -1)
                cv2.rectangle(canvas, (bx, max(0, by - 20)), (bx + badge_w, max(20, by)), trk.color, 1)
                cv2.putText(canvas, badge_text, (bx + 4, max(14, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 255), 1, cv2.LINE_AA)

        # 5. Top Telemetry Banner
        if self.config.show_telemetry_hud:
            banner_overlay = canvas.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, 32), (10, 14, 20), -1)
            cv2.addWeighted(banner_overlay, 0.8, canvas, 0.2, 0, canvas)

            # Brand & Source
            cv2.putText(canvas, "VISIONGUARD CORE // LIVE", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 242, 254), 1, cv2.LINE_AA)

            # Metrics
            counts_summary = self.counting_engine.counts
            metrics_str = f"FPS: {self.fps_processing:.1f} | CPU: {self.cpu_usage_pct}% | RAM: {self.ram_usage_pct}% | TRACKS: {len(tracks)} | IN: {counts_summary['total_in']} | OUT: {counts_summary['total_out']}"
            cv2.putText(canvas, metrics_str, (w - 530, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 240), 1, cv2.LINE_AA)

            # Recording Indicator
            if self.recorder.is_recording:
                cv2.circle(canvas, (w - 550, 18), 6, (0, 0, 255), -1)
                cv2.putText(canvas, "REC", (w - 590, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        return canvas

    def _add_recent_event(self, event_dict: Dict[str, Any]):
        self.recent_events_cache.insert(0, event_dict)
        if len(self.recent_events_cache) > 40:
            self.recent_events_cache.pop()

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Returns JSON-serializable live telemetry payload for WebSockets & REST APIs."""
        counts = self.counting_engine.get_summary()
        zones_summary = self.intrusion_detector.get_summary()

        return {
            "fps": self.fps_processing,
            "cpu_usage": self.cpu_usage_pct,
            "ram_usage": self.ram_usage_pct,
            "active_tracks_count": self.active_tracks_count,
            "motion_intensity": self.motion_detector.motion_intensity,
            "motion_percentage": round(self.motion_detector.motion_percentage, 1),
            "is_recording": self.recorder.is_recording,
            "counts": counts["counts"],
            "line_stats": counts["line_stats"],
            "active_breaches": zones_summary["active_breaches"],
            "zone_stats": zones_summary["zone_stats"],
            "recent_events": self.recent_events_cache[:10],
            "source_type": self.source.source_type,
            "source_param": self.source.source_param,
            "timestamp": time.time()
        }
