"""
VisionGuard - FastAPI Backend Server
Serves MJPEG Stream, WebSockets Telemetry, REST APIs, and Cyber Command Center Frontend.
"""

import os
import time
import asyncio
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from visionguard.config import (
    AnalyticsConfig, IntrusionZoneConfig, TripwireConfig,
    SNAPSHOTS_DIR, RECORDINGS_DIR, UPLOADS_DIR, BASE_DIR
)
from visionguard.core.pipeline import VisionGuardPipeline

# Initialize FastAPI App
app = FastAPI(
    title="VisionGuard Intelligent Video Analytics API",
    description="CPU-based real-time video analytics, motion tracking, zone intrusion, and tripwire counting.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Analytics Pipeline
pipeline = VisionGuardPipeline()

# Mount Static Directories for Media
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event():
    pipeline.start()
    print("[VisionGuard] Video Analytics Pipeline started successfully.")


@app.on_event("shutdown")
def shutdown_event():
    pipeline.stop()
    print("[VisionGuard] Pipeline stopped.")


# ---------------------------------------------------------
# Live Video Stream (MJPEG)
# ---------------------------------------------------------
def generate_mjpeg():
    """Generates continuous MJPEG multipart stream from pipeline buffer."""
    while True:
        with pipeline.frame_lock:
            jpeg_bytes = pipeline.latest_jpeg_bytes

        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.03)  # ~30 FPS streaming yield


@app.get("/api/stream")
def stream_video():
    """Streams live video with real-time HUD annotations."""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------
# Real-Time Telemetry WebSocket
# ---------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            telemetry = pipeline.get_telemetry_snapshot()
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.1)  # 10Hz updates
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ---------------------------------------------------------
# REST APIs: System Configuration & Controls
# ---------------------------------------------------------
@app.get("/api/config")
def get_config():
    return pipeline.config


@app.post("/api/config")
def update_config(config: AnalyticsConfig):
    pipeline.update_config(config)
    return {"status": "success", "config": pipeline.config}


@app.post("/api/source")
def switch_source(source_type: str = Form(...), param: str = Form("")):
    """Switch camera source (webcam, file, simulation, rtsp)."""
    pipeline.source.set_source(source_type, param)
    return {"status": "success", "source_type": source_type, "param": param}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a local video file and switches analysis source to it."""
    filename = f"upload_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.source.set_source("file", filepath)
    return {"status": "success", "filename": filename, "filepath": filepath}


# ---------------------------------------------------------
# REST APIs: Zones & Tripwires
# ---------------------------------------------------------
@app.get("/api/zones")
def get_zones():
    return pipeline.config.zones


@app.post("/api/zones")
def save_zone(zone: IntrusionZoneConfig):
    # Update existing or add new
    existing_idx = next((i for i, z in enumerate(pipeline.config.zones) if z.id == zone.id), -1)
    if existing_idx >= 0:
        pipeline.config.zones[existing_idx] = zone
    else:
        pipeline.config.zones.append(zone)

    pipeline.intrusion_detector.set_zones(pipeline.config.zones)
    return {"status": "success", "zones": pipeline.config.zones}


@app.delete("/api/zones/{zone_id}")
def delete_zone(zone_id: str):
    pipeline.config.zones = [z for z in pipeline.config.zones if z.id != zone_id]
    pipeline.intrusion_detector.set_zones(pipeline.config.zones)
    return {"status": "success", "zones": pipeline.config.zones}


@app.get("/api/tripwires")
def get_tripwires():
    return pipeline.config.tripwires


@app.post("/api/tripwires")
def save_tripwire(tripwire: TripwireConfig):
    existing_idx = next((i for i, tw in enumerate(pipeline.config.tripwires) if tw.id == tripwire.id), -1)
    if existing_idx >= 0:
        pipeline.config.tripwires[existing_idx] = tripwire
    else:
        pipeline.config.tripwires.append(tripwire)

    pipeline.counting_engine.set_tripwires(pipeline.config.tripwires)
    return {"status": "success", "tripwires": pipeline.config.tripwires}


@app.delete("/api/tripwires/{tripwire_id}")
def delete_tripwire(tripwire_id: str):
    pipeline.config.tripwires = [tw for tw in pipeline.config.tripwires if tw.id != tripwire_id]
    pipeline.counting_engine.set_tripwires(pipeline.config.tripwires)
    return {"status": "success", "tripwires": pipeline.config.tripwires}


# ---------------------------------------------------------
# REST APIs: Instant Actions (Snapshot, Record, Reset)
# ---------------------------------------------------------
@app.post("/api/snapshot")
def manual_snapshot():
    with pipeline.frame_lock:
        frame = pipeline.latest_annotated_frame or pipeline.latest_raw_frame

    if frame is None:
        raise HTTPException(status_code=400, detail="No active video frame available")

    snap = pipeline.recorder.capture_snapshot(frame, event_type="MANUAL_SNAPSHOT", metadata={"severity": "INFO"})
    logged = pipeline.logger.log_event(
        event_type="SNAPSHOT_TAKEN",
        severity="INFO",
        snapshot_url=snap["url"],
        details=f"Manual snapshot captured: {snap['filename']}"
    )
    pipeline._add_recent_event(logged)
    return {"status": "success", "snapshot": snap}


@app.post("/api/record/start")
def start_recording():
    rec_id = pipeline.recorder.start_manual_recording()
    logged = pipeline.logger.log_event(
        event_type="SYSTEM_ALERT",
        severity="INFO",
        details=f"Manual video recording initiated (#{rec_id})"
    )
    pipeline._add_recent_event(logged)
    return {"status": "success", "recording_id": rec_id}


@app.post("/api/record/stop")
def stop_recording():
    rec_id = pipeline.recorder.stop_manual_recording()
    if rec_id:
        logged = pipeline.logger.log_event(
            event_type="RECORDING_SAVED",
            severity="INFO",
            recording_url=f"/recordings/{rec_id}.mp4",
            details=f"Manual video clip saved: {rec_id}.mp4"
        )
        pipeline._add_recent_event(logged)
    return {"status": "success", "recording_id": rec_id}


@app.post("/api/counts/reset")
def reset_counts():
    pipeline.counting_engine.reset()
    return {"status": "success", "message": "Counts reset to zero."}


# ---------------------------------------------------------
# REST APIs: Event Logs & Analytics
# ---------------------------------------------------------
@app.get("/api/events")
def get_events(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    zone_id: Optional[str] = None
):
    events = pipeline.logger.get_events(limit, offset, severity, event_type, zone_id)
    return {"events": events}


@app.get("/api/events/export/csv")
def export_events_csv():
    csv_data = pipeline.logger.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visionguard_events.csv"}
    )


@app.delete("/api/events")
def clear_events():
    pipeline.logger.clear_events()
    pipeline.recent_events_cache.clear()
    return {"status": "success", "message": "Event history cleared."}


@app.get("/api/stats")
def get_statistics():
    event_stats = pipeline.logger.get_event_stats()
    telemetry = pipeline.get_telemetry_snapshot()
    return {
        "event_stats": event_stats,
        "live_telemetry": telemetry
    }


# ---------------------------------------------------------
# REST APIs: Recordings & Snapshots Gallery
# ---------------------------------------------------------
@app.get("/api/recordings")
def list_recordings():
    # Scan recordings directory and return files
    files = []
    if os.path.exists(RECORDINGS_DIR):
        for fname in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if fname.endswith(".mp4"):
                fpath = os.path.join(RECORDINGS_DIR, fname)
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                mtime = os.path.getmtime(fpath)
                files.append({
                    "id": fname.replace(".mp4", ""),
                    "filename": fname,
                    "url": f"/recordings/{fname}",
                    "size_mb": size_mb,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                })
    return {"recordings": files}


@app.get("/api/snapshots")
def list_snapshots():
    files = []
    if os.path.exists(SNAPSHOTS_DIR):
        for fname in sorted(os.listdir(SNAPSHOTS_DIR), reverse=True):
            if fname.endswith(".jpg"):
                fpath = os.path.join(SNAPSHOTS_DIR, fname)
                size_kb = round(os.path.getsize(fpath) / 1024, 1)
                mtime = os.path.getmtime(fpath)
                files.append({
                    "id": fname.replace(".jpg", ""),
                    "filename": fname,
                    "url": f"/snapshots/{fname}",
                    "size_kb": size_kb,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                })
    return {"snapshots": files}


# ---------------------------------------------------------
# Root Web Dashboard HTML Route
# ---------------------------------------------------------
@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>VisionGuard Server is Running. Frontend files loading...</h1>")


if __name__ == "__main__":
    uvicorn.run("visionguard.server:app", host="0.0.0.0", port=8000, reload=False)
