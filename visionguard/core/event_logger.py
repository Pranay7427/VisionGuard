"""
VisionGuard - Persistent SQLite Event Logger & Audit Engine
Records and indexes security events, triggers, line crossings, zone intrusions, with CSV/JSON export.
"""

import sqlite3
import time
import os
import json
import csv
import io
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from visionguard.config import DB_PATH


class EventLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    formatted_time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    zone_id TEXT,
                    zone_name TEXT,
                    tripwire_id TEXT,
                    tripwire_name TEXT,
                    track_id INTEGER,
                    object_class TEXT,
                    confidence REAL,
                    snapshot_url TEXT,
                    recording_url TEXT,
                    details TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON events(severity)")
            conn.commit()
            conn.close()

    def log_event(
        self,
        event_type: str,
        severity: str = "INFO",
        zone_id: Optional[str] = None,
        zone_name: Optional[str] = None,
        tripwire_id: Optional[str] = None,
        tripwire_name: Optional[str] = None,
        track_id: Optional[int] = None,
        object_class: Optional[str] = None,
        confidence: Optional[float] = None,
        snapshot_url: Optional[str] = None,
        recording_url: Optional[str] = None,
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        now = time.time()
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (
                    timestamp, formatted_time, event_type, severity,
                    zone_id, zone_name, tripwire_id, tripwire_name,
                    track_id, object_class, confidence,
                    snapshot_url, recording_url, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, formatted_time, event_type, severity,
                zone_id, zone_name, tripwire_id, tripwire_name,
                track_id, object_class, confidence,
                snapshot_url, recording_url, details
            ))
            event_id = cursor.lastrowid
            conn.commit()
            conn.close()

        return {
            "id": event_id,
            "timestamp": now,
            "formatted_time": formatted_time,
            "event_type": event_type,
            "severity": severity,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "tripwire_id": tripwire_id,
            "tripwire_name": tripwire_name,
            "track_id": track_id,
            "object_class": object_class,
            "confidence": confidence,
            "snapshot_url": snapshot_url,
            "recording_url": recording_url,
            "details": details
        }

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
        zone_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if severity and severity != "ALL":
                query += " AND severity = ?"
                params.append(severity)
            if event_type and event_type != "ALL":
                query += " AND event_type = ?"
                params.append(event_type)
            if zone_id and zone_id != "ALL":
                query += " AND zone_id = ?"
                params.append(zone_id)

            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def get_event_stats(self) -> Dict[str, Any]:
        """Calculates aggregated metrics for dashboard analytics."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total counts by severity
            cursor.execute("SELECT severity, COUNT(*) FROM events GROUP BY severity")
            severity_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Counts by event type
            cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Zone breach counts
            cursor.execute("SELECT zone_name, COUNT(*) FROM events WHERE zone_name IS NOT NULL GROUP BY zone_name")
            zone_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Object class distribution
            cursor.execute("SELECT object_class, COUNT(*) FROM events WHERE object_class IS NOT NULL GROUP BY object_class")
            class_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Total events
            cursor.execute("SELECT COUNT(*) FROM events")
            total_events = cursor.fetchone()[0]

            conn.close()

            return {
                "total_events": total_events,
                "severity_counts": {
                    "INFO": severity_counts.get("INFO", 0),
                    "WARNING": severity_counts.get("WARNING", 0),
                    "CRITICAL": severity_counts.get("CRITICAL", 0)
                },
                "type_counts": type_counts,
                "zone_counts": zone_counts,
                "class_counts": class_counts
            }

    def export_csv(self) -> str:
        """Exports all events as CSV string."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY id DESC")
            rows = cursor.fetchall()
            conn.close()

            output = io.StringIO()
            if len(rows) > 0:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
            else:
                writer = csv.writer(output)
                writer.writerow(["id", "timestamp", "formatted_time", "event_type", "severity", "details"])

            return output.getvalue()

    def clear_events(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            conn.commit()
            conn.close()
