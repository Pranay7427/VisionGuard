"""
VisionGuard - Automated Event Recording & Snapshot Capture Engine
Maintains Circular Pre-Roll Frame Buffer, Asynchronous MP4 Video Writer, and Security Snapshots.
"""

import cv2
import numpy as np
import time
import os
import threading
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from visionguard.config import SNAPSHOTS_DIR, RECORDINGS_DIR


class EventRecorder:
    def __init__(self, pre_roll_sec: int = 4, post_roll_sec: int = 6, fps: int = 20):
        self.pre_roll_sec = pre_roll_sec
        self.post_roll_sec = post_roll_sec
        self.fps = fps
        self.max_buffer_len = int(pre_roll_sec * fps)

        # Circular buffer for pre-roll frames: stores (frame, timestamp)
        self.frame_buffer = deque(maxlen=self.max_buffer_len)
        self.lock = threading.Lock()

        # Active recording state
        self.is_recording = False
        self.is_manual_recording = False
        self.current_recording_id: Optional[str] = None
        self.current_recording_path: Optional[str] = None
        self.recording_start_time: float = 0.0
        self.recording_frames_queue = deque()
        self.post_roll_end_time: float = 0.0
        self.writer_thread: Optional[threading.Thread] = None
        self.stop_writer_flag = threading.Event()
        self.recordings_history: List[Dict[str, Any]] = []
        self.snapshots_history: List[Dict[str, Any]] = []

    def push_frame(self, frame: np.ndarray):
        """Pushes current live frame to circular pre-roll buffer and active writer queue."""
        now = time.time()
        with self.lock:
            self.frame_buffer.append((frame.copy(), now))
            if self.is_recording:
                self.recording_frames_queue.append(frame.copy())
                # Check if recording should end (for automatic event recordings)
                if not self.is_manual_recording and now >= self.post_roll_end_time:
                    self._stop_recording_async()

    def trigger_auto_recording(self, event_name: str, zone_name: str = "") -> Optional[str]:
        """Triggers an automated incident clip recording if not already recording."""
        now = time.time()
        with self.lock:
            if self.is_recording:
                # Extend post-roll cooldown
                self.post_roll_end_time = now + self.post_roll_sec
                return self.current_recording_id

            # Start new recording
            self.is_recording = True
            self.is_manual_recording = False
            self.recording_start_time = now
            self.post_roll_end_time = now + self.post_roll_sec

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_event = event_name.replace(" ", "_").lower()
            rec_id = f"event_{clean_event}_{timestamp_str}"
            filename = f"{rec_id}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, filename)
            self.current_recording_id = rec_id
            self.current_recording_path = filepath

            # Prepend all frames from circular pre-roll buffer
            self.recording_frames_queue.clear()
            for buf_frame, _ in self.frame_buffer:
                self.recording_frames_queue.append(buf_frame)

            # Start background writer thread
            self.stop_writer_flag.clear()
            self.writer_thread = threading.Thread(
                target=self._writer_worker,
                args=(filepath, rec_id, event_name, zone_name),
                daemon=True
            )
            self.writer_thread.start()
            return rec_id

    def start_manual_recording(self) -> str:
        """Starts manual user-initiated recording."""
        now = time.time()
        with self.lock:
            if self.is_recording:
                return self.current_recording_id or ""

            self.is_recording = True
            self.is_manual_recording = True
            self.recording_start_time = now
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_id = f"manual_{timestamp_str}"
            filename = f"{rec_id}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, filename)
            self.current_recording_id = rec_id
            self.current_recording_path = filepath

            self.recording_frames_queue.clear()
            for buf_frame, _ in self.frame_buffer:
                self.recording_frames_queue.append(buf_frame)

            self.stop_writer_flag.clear()
            self.writer_thread = threading.Thread(
                target=self._writer_worker,
                args=(filepath, rec_id, "MANUAL_RECORDING", "User Action"),
                daemon=True
            )
            self.writer_thread.start()
            return rec_id

    def stop_manual_recording(self) -> Optional[str]:
        """Stops active manual recording."""
        with self.lock:
            if self.is_recording and self.is_manual_recording:
                rec_id = self.current_recording_id
                self._stop_recording_async()
                return rec_id
            return None

    def _stop_recording_async(self):
        self.is_recording = False
        self.stop_writer_flag.set()

    def _writer_worker(self, filepath: str, rec_id: str, event_type: str, details: str):
        """Background thread writing queued frames to MP4 file."""
        writer = None
        written_count = 0
        h, w = 0, 0

        while not self.stop_writer_flag.is_set() or len(self.recording_frames_queue) > 0:
            if len(self.recording_frames_queue) > 0:
                frame = self.recording_frames_queue.popleft()
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(filepath, fourcc, float(self.fps), (w, h))

                if writer.isOpened():
                    writer.write(frame)
                    written_count += 1
            else:
                time.sleep(0.04)

        if writer is not None:
            writer.release()

        # Log recording record
        duration = round(written_count / float(self.fps), 1) if self.fps > 0 else 0.0
        size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        rec_info = {
            "id": rec_id,
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "url": f"/recordings/{os.path.basename(filepath)}",
            "event_type": event_type,
            "details": details,
            "duration_sec": duration,
            "size_bytes": size_bytes,
            "timestamp": time.time(),
            "formatted_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.recordings_history.insert(0, rec_info)

    def capture_snapshot(self, frame: np.ndarray, event_type: str = "MANUAL", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Captures a high-resolution snapshot with visual security watermark.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        snap_id = f"snap_{timestamp_str}"
        filename = f"{snap_id}.jpg"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)

        # Create annotated snapshot with watermark
        watermarked = frame.copy()
        h, w = watermarked.shape[:2]

        # Top banner with security metadata
        overlay = watermarked.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (15, 20, 25), -1)
        cv2.rectangle(overlay, (0, h - 30), (w, h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, watermarked, 0.25, 0, watermarked)

        # Watermark Text
        date_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(watermarked, f"VISIONGUARD CCTV // {event_type.upper()}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1)
        cv2.putText(watermarked, date_text, (w - 240, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        if metadata:
            detail_str = f"Track: #{metadata.get('track_id', 'N/A')} | Zone: {metadata.get('zone_name', 'N/A')} | Severity: {metadata.get('severity', 'INFO')}"
            cv2.putText(watermarked, detail_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 240), 1)

        # Save JPEG
        cv2.imwrite(filepath, watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        snap_info = {
            "id": snap_id,
            "filename": filename,
            "filepath": filepath,
            "url": f"/snapshots/{filename}",
            "event_type": event_type,
            "timestamp": time.time(),
            "formatted_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }
        self.snapshots_history.insert(0, snap_info)
        return snap_info
