"""
VisionGuard - API Integration Unit Tests
Tests FastAPI REST endpoints, configuration updates, zones/tripwires APIs, and event stats.
"""

import unittest
from starlette.testclient import TestClient
from visionguard.server import app, pipeline


class TestServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_config(self):
        """Test root endpoint and configuration retrieval."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

        res_config = self.client.get("/api/config")
        self.assertEqual(res_config.status_code, 200)
        data = res_config.json()
        self.assertIn("motion_sensitivity", data)
        self.assertIn("zones", data)
        self.assertIn("tripwires", data)

    def test_zones_crud(self):
        """Test adding, retrieving, and deleting polygon zones."""
        zone_payload = {
            "id": "test_api_zone",
            "name": "Server Room",
            "polygon": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.4, "y": 0.1},
                {"x": 0.4, "y": 0.4},
                {"x": 0.1, "y": 0.4}
            ],
            "color": "#10b981",
            "severity": "RESTRICTED",
            "dwell_threshold_sec": 1.5,
            "enabled": True
        }

        # Create
        res_post = self.client.post("/api/zones", json=zone_payload)
        self.assertEqual(res_post.status_code, 200)
        zones = res_post.json()["zones"]
        self.assertTrue(any(z["id"] == "test_api_zone" for z in zones))

        # Get
        res_get = self.client.get("/api/zones")
        self.assertEqual(res_get.status_code, 200)
        self.assertTrue(any(z["id"] == "test_api_zone" for z in res_get.json()))

        # Delete
        res_del = self.client.delete("/api/zones/test_api_zone")
        self.assertEqual(res_del.status_code, 200)
        self.assertFalse(any(z["id"] == "test_api_zone" for z in res_del.json()["zones"]))

    def test_tripwires_crud(self):
        """Test creating and deleting tripwire lines."""
        tw_payload = {
            "id": "test_api_tw",
            "name": "Corridor Tripwire",
            "start": {"x": 0.2, "y": 0.3},
            "end": {"x": 0.8, "y": 0.3},
            "color": "#00f2fe",
            "direction": "BOTH",
            "enabled": True
        }

        # Create
        res_post = self.client.post("/api/tripwires", json=tw_payload)
        self.assertEqual(res_post.status_code, 200)

        # Delete
        res_del = self.client.delete("/api/tripwires/test_api_tw")
        self.assertEqual(res_del.status_code, 200)

    def test_instant_actions_and_stats(self):
        """Test snapshot trigger, stats API, and recordings list."""
        # Stats
        res_stats = self.client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        data = res_stats.json()
        self.assertIn("event_stats", data)
        self.assertIn("live_telemetry", data)

        # Recordings list
        res_rec = self.client.get("/api/recordings")
        self.assertEqual(res_rec.status_code, 200)

        # Snapshots list
        res_snaps = self.client.get("/api/snapshots")
        self.assertEqual(res_snaps.status_code, 200)

        # Reset counts
        res_reset = self.client.post("/api/counts/reset")
        self.assertEqual(res_reset.status_code, 200)


if __name__ == "__main__":
    unittest.main()
