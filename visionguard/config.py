"""
VisionGuard - Configuration & Data Models
Defines settings, detection modes, zone and tripwire configurations.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
SNAPSHOTS_DIR = os.path.join(MEDIA_DIR, "snapshots")
RECORDINGS_DIR = os.path.join(MEDIA_DIR, "recordings")
UPLOADS_DIR = os.path.join(MEDIA_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "visionguard.db")

# Ensure directories exist
for path in [MEDIA_DIR, SNAPSHOTS_DIR, RECORDINGS_DIR, UPLOADS_DIR, DATA_DIR]:
    os.makedirs(path, exist_ok=True)


class ZoneAlertSeverity(str, Enum):
    RESTRICTED = "RESTRICTED"  # Immediate alert upon entry
    CAUTION = "CAUTION"        # Dwell time based alert
    MONITORED = "MONITORED"    # Silent tracking & logging only


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    MOTION_DETECTED = "MOTION_DETECTED"
    ZONE_INTRUSION = "ZONE_INTRUSION"
    ZONE_DWELL_BREACH = "ZONE_DWELL_BREACH"
    TRIPWIRE_CROSSED = "TRIPWIRE_CROSSED"
    RECORDING_SAVED = "RECORDING_SAVED"
    SNAPSHOT_TAKEN = "SNAPSHOT_TAKEN"
    SYSTEM_ALERT = "SYSTEM_ALERT"


class Point(BaseModel):
    x: float  # Normalized 0.0 to 1.0 or pixel coordinate
    y: float


class IntrusionZoneConfig(BaseModel):
    id: str
    name: str
    polygon: List[Point]  # List of vertices (normalized 0.0 - 1.0)
    color: str = "#ef4444"
    severity: ZoneAlertSeverity = ZoneAlertSeverity.RESTRICTED
    dwell_threshold_sec: float = 2.0
    enabled: bool = True


class TripwireConfig(BaseModel):
    id: str
    name: str
    start: Point
    end: Point
    color: str = "#00f2fe"
    direction: str = "BOTH"  # "IN", "OUT", or "BOTH"
    enabled: bool = True


class AnalyticsConfig(BaseModel):
    # Module Toggles
    motion_detection_enabled: bool = True
    object_tracking_enabled: bool = True
    people_counting_enabled: bool = True
    vehicle_counting_enabled: bool = True
    intrusion_detection_enabled: bool = True
    auto_recording_enabled: bool = True
    heatmap_enabled: bool = False

    # Detection & Sensitivity
    motion_sensitivity: int = 50  # 1 to 100
    min_contour_area: int = 800  # pixels
    detector_model: str = "mobilenet_ssd"  # "mobilenet_ssd", "hog_svm", "contour_heuristic"
    confidence_threshold: float = 0.45

    # Visual HUD Overlay Toggles
    show_bounding_boxes: bool = True
    show_track_ids: bool = True
    show_motion_mask: bool = False
    show_heatmap: bool = False
    show_tracking_trails: bool = True
    show_zones: bool = True
    show_tripwires: bool = True
    show_telemetry_hud: bool = True

    # Recording settings
    pre_roll_sec: int = 4
    post_roll_sec: int = 6
    recording_fps: int = 20

    # Active Zones and Tripwires
    zones: List[IntrusionZoneConfig] = []
    tripwires: List[TripwireConfig] = []
