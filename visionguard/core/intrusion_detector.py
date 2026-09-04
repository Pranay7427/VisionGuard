"""
VisionGuard - Polygon Region Intrusion & Dwell Time Detection Engine
Implements OpenCV Point-in-Polygon Testing, Multi-tier Severity, and Dwell Breach Triggers.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional
from visionguard.config import IntrusionZoneConfig, ZoneAlertSeverity


class IntrusionDetector:
    def __init__(self):
        self.zones: Dict[str, IntrusionZoneConfig] = {}
        self.active_breaches: Dict[str, List[int]] = {}  # zone_id -> list of track_ids currently inside
        self.zone_stats: Dict[str, Dict[str, Any]] = {}  # zone_id -> breach counters and history

    def set_zones(self, zones: List[IntrusionZoneConfig]):
        self.zones = {z.id: z for z in zones}
        for z in zones:
            if z.id not in self.zone_stats:
                self.zone_stats[z.id] = {
                    "breach_count": 0,
                    "total_dwell_sec": 0.0,
                    "last_breach_time": None
                }
            if z.id not in self.active_breaches:
                self.active_breaches[z.id] = []

    def process_tracks(self, tracks: List[Any], frame_width: int, frame_height: int) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """
        Evaluates all active tracks against configured polygon zones.
        Returns:
            - events: list of newly triggered intrusion / dwell breach event dicts
            - zone_alert_status: dict of zone_id -> is_in_breach (for visual rendering)
        """
        events = []
        now = time.time()
        zone_alert_status = {z_id: False for z_id in self.zones}

        # Clear current active breaches mapping for fresh evaluation
        for z_id in self.zones:
            self.active_breaches[z_id] = []

        for z_id, zone in self.zones.items():
            if not zone.enabled or len(zone.polygon) < 3:
                continue

            # Convert normalized polygon vertices to pixel array for OpenCV
            pts = []
            for pt in zone.polygon:
                px = int(pt.x * frame_width if pt.x <= 1.0 else pt.x)
                py = int(pt.y * frame_height if pt.y <= 1.0 else pt.y)
                pts.append([px, py])
            poly_np = np.array(pts, dtype=np.int32)

            for track in tracks:
                cx, cy = track.centroid
                
                # Check point in polygon using cv2.pointPolygonTest
                # >= 0 means on edge or inside polygon
                dist = cv2.pointPolygonTest(poly_np, (float(cx), float(cy)), False)
                is_inside = (dist >= 0)

                if is_inside:
                    self.active_breaches[z_id].append(track.track_id)
                    zone_alert_status[z_id] = True

                    # Track entry time
                    if z_id not in track.zone_entry_times:
                        track.zone_entry_times[z_id] = now

                    dwell_time = now - track.zone_entry_times[z_id]

                    # Check trigger criteria
                    should_alert = False
                    event_type = "ZONE_INTRUSION"
                    severity = "WARNING"

                    if zone.severity == ZoneAlertSeverity.RESTRICTED:
                        # Immediate intrusion alert
                        if z_id not in track.zone_breached_alerts:
                            should_alert = True
                            severity = "CRITICAL"
                            event_type = "ZONE_INTRUSION"
                    elif zone.severity == ZoneAlertSeverity.CAUTION:
                        # Dwell time breach
                        if dwell_time >= zone.dwell_threshold_sec and z_id not in track.zone_breached_alerts:
                            should_alert = True
                            severity = "WARNING"
                            event_type = "ZONE_DWELL_BREACH"
                    elif zone.severity == ZoneAlertSeverity.MONITORED:
                        # Informational log only
                        if z_id not in track.zone_breached_alerts and dwell_time >= 1.0:
                            should_alert = True
                            severity = "INFO"
                            event_type = "ZONE_INTRUSION"

                    if should_alert:
                        track.zone_breached_alerts.add(z_id)
                        self.zone_stats[z_id]["breach_count"] += 1
                        self.zone_stats[z_id]["last_breach_time"] = now

                        events.append({
                            "event_type": event_type,
                            "severity": severity,
                            "zone_id": z_id,
                            "zone_name": zone.name,
                            "track_id": track.track_id,
                            "label": track.label,
                            "confidence": track.confidence,
                            "dwell_time_sec": round(dwell_time, 2),
                            "box": track.box,
                            "centroid": track.centroid,
                            "timestamp": now,
                            "trigger_snapshot": True,
                            "trigger_recording": (severity in ["WARNING", "CRITICAL"])
                        })
                else:
                    # If track has exited zone, reset zone dwell and alert state for this track
                    if z_id in track.zone_entry_times:
                        dwell = now - track.zone_entry_times[z_id]
                        self.zone_stats[z_id]["total_dwell_sec"] += dwell
                        del track.zone_entry_times[z_id]
                    if z_id in track.zone_breached_alerts:
                        track.zone_breached_alerts.remove(z_id)

        return events, zone_alert_status

    def get_summary(self) -> Dict[str, Any]:
        return {
            "active_breaches": {k: list(v) for k, v in self.active_breaches.items()},
            "zone_stats": {k: v.copy() for k, v in self.zone_stats.items()}
        }
