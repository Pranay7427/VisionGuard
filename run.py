"""
VisionGuard - CPU-Based Intelligent Video Analytics System Entrypoint
Run: python run.py
"""

import uvicorn
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    print("=" * 65)
    print("  VISIONGUARD — CPU-Based Intelligent Video Analytics System")
    print("=" * 65)
    print("  * Web Dashboard: http://localhost:8000")
    print("  * Live Stream:   http://localhost:8000/api/stream")
    print("  * Telemetry WS:  ws://localhost:8000/ws/telemetry")
    print("=" * 65)
    uvicorn.run("visionguard.server:app", host="0.0.0.0", port=8000, reload=False)
