# VisionGuard — CPU-Based Intelligent Video Analytics System

**VisionGuard** is a high-performance, CPU-optimized intelligent video analytics platform. It combines multiple modular OpenCV computer vision techniques to provide real-time perimeter surveillance, bidirectional people/vehicle counting, polygon region intrusion detection, automated incident video clipping, instant snapshot capturing, structured audit logging, and a web-based command center dashboard.

---

## Key Features & OpenCV Modules

### 1. CPU-Optimized Motion Detection & Thermal Heatmap
- **MOG2 / KNN Background Subtraction**: Dynamic background modeling with shadow suppression (`detectShadows=True`).
- **Morphological Noise Filtering**: Multi-stage opening, closing, and dilation structuring elements to isolate clean entity contours.
- **Cumulative Thermal Motion Heatmap**: Float matrix accumulating continuous movement with decay rates, rendered via `cv2.COLORMAP_TURBO` / `cv2.COLORMAP_JET`.

### 2. Multi-Object Tracking & Trajectories
- **Centroid & Euclidean Distance Tracker**: Maintains persistent track IDs, computes real-time velocity vectors ($v_x, v_y$), speed in px/sec, and 8-compass direction heading (N, NE, E, SE, S, SW, W, NW).
- **Motion Trails**: Renders color-coded historical path vectors showing trajectory history.

### 3. Bidirectional People & Vehicle Counting
- **Virtual Tripwires**: Vector line-segment intersection algorithms with cross-product direction calculation.
- **Per-Class Tallies**: Tallying for Pedestrians, Vehicles, Inflow, and Outflow across arbitrary custom lines.

### 4. Polygon Region Intrusion Detection (ROI)
- **Point-in-Polygon Containment**: Ray-casting and distance testing via `cv2.pointPolygonTest`.
- **Multi-Tier Severity**:
  - `RESTRICTED`: Instant alarm, snapshot capture, and automatic video recording.
  - `CAUTION`: Dwell-time violation threshold (e.g. alert if target remains $> 2.0$s).
  - `MONITORED`: Silent activity audit and logging.

### 5. Automated Incident Recording & Snapshots
- **Circular Pre-Roll Buffer**: Keeps a 4-second memory buffer of past frames. Upon an intrusion breach, saves pre-roll + live incident + post-roll frames into dedicated `.mp4` video clips.
- **Watermarked Snapshots**: High-resolution security captures with timestamp, camera tag, track ID, and severity metadata banner.

### 6. Security Audit Log & Export
- **Persistent SQLite Database**: Structured indexing of all security incidents, searchable and filterable by severity (`CRITICAL`, `WARNING`, `INFO`).
- **CSV & JSON Export**: One-click download of audit data for compliance and reporting.

### 7. Cyber Command Center Web Dashboard
- **Real-Time Live Video Stream**: Low-latency MJPEG streaming with HUD layer toggles (Bounding boxes, Track IDs, Motion mask, Thermal heatmap, Trails, Zones, Tripwires).
- **Interactive Canvas Studio**: Draw polygon intrusion zones and tripwire counting lines directly on the live video canvas.
- **Web Audio API Alarm Synthesizer**: Generates audible warning beeps, crossing chirps, and dual-tone security siren alerts in-browser without external audio files.
- **Live Analytics Charts**: Traffic flow timeline, object classification distribution, hourly volume, and zone breach frequency powered by Chart.js.
- **Incident Playback Gallery**: Integrated media player for reviewing recorded `.mp4` video clips and full-resolution snapshots.

---

## Live System Snapshot Demonstration

Below is an automated incident snapshot captured during an intrusion breach in the Restricted Courtyard zone:

![VisionGuard Incident Snapshot](file:///C:/Users/prana/.gemini/antigravity-ide/brain/b30f529b-2204-4201-9df1-4c0620d7d908/snapshot_demo.jpg)

---

## Verification & Test Results

### 1. Automated Unit & Integration Tests
All 10 test suites passed cleanly in under 0.3 seconds:
- `test_synthetic_video_source`: Verified synthetic CCTV generator produces valid frames across perimeter, traffic, and pedestrian scenarios.
- `test_motion_detector`: Verified MOG2 background subtraction, shadow filtering, and thermal heatmap accumulation.
- `test_object_tracker`: Verified centroid tracking, persistent ID assignment, velocity, and trajectory history.
- `test_tripwire_counting`: Verified vector line-segment crossing and bidirectional in/out tallies.
- `test_polygon_intrusion_detection`: Verified point-in-polygon containment, dwell time breach alerts, and automatic trigger emission.
- `test_event_logger_and_export`: Verified SQLite schema, indexed event retrieval, and CSV formatting.
- `test_root_and_config`: Verified FastAPI startup, root HTML delivery, and configuration endpoints.
- `test_zones_crud` & `test_tripwires_crud`: Verified REST creation, retrieval, and deletion of custom ROIs and lines.
- `test_instant_actions_and_stats`: Verified telemetry aggregation, snapshot triggers, and media listing.

### 2. Live Deployment Status
- Server running at **`http://localhost:8000`**
- WebSocket telemetry streaming at **`ws://localhost:8000/ws/telemetry`**
- Live video stream available at **`http://localhost:8000/api/stream`**

---

## How to Run & Use VisionGuard

### Start the Application
```bash
python run.py
```
Open your web browser and navigate to **`http://localhost:8000`**.

### Usage Guide
1. **Switch Video Source**: Use the top-left dropdown to select from built-in simulation scenarios (Perimeter, Traffic, Pedestrian) or your physical webcam (Webcam 0/1) or upload a custom video file.
2. **Draw Security Zones**: Click `+ Zone` in the left panel, click vertices on the live video canvas to enclose an area, and save.
3. **Draw Tripwires**: Click `+ Line`, click a start point and end point to establish an in/out counting boundary.
4. **Toggle HUD Overlays**: Use the toolbar beneath the video to toggle bounding boxes, track IDs, motion trails, or the thermal heatmap.
5. **Review Incidents**: Click the *Recordings & Snapshots* tab in the bottom deck to play back recorded intrusion clips or inspect high-res snapshots.
