"""
VisionGuard - Unified Video Source Manager
Supports Physical Webcam, Video Files, RTSP/HTTP Streams, and High-Fidelity Synthetic Simulations.
"""

import cv2
import numpy as np
import time
import threading
import math
import random
import os
from typing import Optional, Tuple, Dict, Any


class SimulatedEntity:
    def __init__(self, entity_id: int, entity_type: str, x: float, y: float, vx: float, vy: float, color: Tuple[int, int, int], size: Tuple[int, int]):
        self.id = entity_id
        self.type = entity_type  # "person", "car", "truck", "intruder"
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.width, self.height = size
        self.angle = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0
        self.life = 0
        self.max_life = random.randint(400, 1200)

    def update(self, bounds_w: int, bounds_h: int):
        self.x += self.vx
        self.y += self.vy
        self.life += 1

        # Bounce or wrap around edges
        if self.x < -self.width:
            self.x = bounds_w + self.width
        elif self.x > bounds_w + self.width:
            self.x = -self.width

        if self.y < -self.height:
            self.y = bounds_h + self.height
        elif self.y > bounds_h + self.height:
            self.y = -self.height

        # Add subtle natural wobble
        if self.type in ["person", "intruder"]:
            self.vx += random.uniform(-0.15, 0.15)
            self.vy += random.uniform(-0.15, 0.15)
            # Speed clamping
            speed = math.hypot(self.vx, self.vy)
            if speed > 3.0:
                self.vx = (self.vx / speed) * 3.0
                self.vy = (self.vy / speed) * 3.0


class SyntheticSceneGenerator:
    """Generates dynamic, realistic CPU-rendered video scenes for instant testing."""
    def __init__(self, width: int = 960, height: int = 540, scene_type: str = "perimeter"):
        self.width = width
        self.height = height
        self.scene_type = scene_type
        self.frame_count = 0
        self.entities = []
        self.next_id = 1
        self._init_scene()

    def _init_scene(self):
        self.entities.clear()
        if self.scene_type == "traffic":
            # 2-lane road with cars, trucks, motorcycles
            for _ in range(7):
                self._spawn_vehicle()
        elif self.scene_type == "pedestrian":
            # Public courtyard with walking pedestrians
            for _ in range(8):
                self._spawn_pedestrian()
        else:  # "perimeter"
            # High-security restricted facility with patrol guards and an occasional intruder
            for _ in range(4):
                self._spawn_guard()
            self._spawn_intruder()

    def _spawn_vehicle(self):
        y_lane = random.choice([self.height * 0.38, self.height * 0.62])
        direction = 1 if y_lane > self.height * 0.5 else -1
        vx = direction * random.uniform(3.5, 6.0)
        is_truck = random.random() < 0.25
        entity_type = "truck" if is_truck else "car"
        size = (90, 48) if is_truck else (65, 34)
        colors = [
            (35, 75, 215),   # Red car
            (210, 180, 50),  # Cyan/Blue car
            (45, 180, 75),   # Green
            (230, 230, 235), # White
            (50, 50, 55),    # Charcoal
            (20, 140, 220)   # Orange
        ]
        color = random.choice(colors)
        x = -100 if direction == 1 else self.width + 100
        self.entities.append(SimulatedEntity(self.next_id, entity_type, x, y_lane + random.uniform(-10, 10), vx, 0, color, size))
        self.next_id += 1

    def _spawn_pedestrian(self):
        x = random.uniform(50, self.width - 50)
        y = random.uniform(50, self.height - 50)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.2, 2.4)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        colors = [(220, 120, 50), (60, 180, 240), (180, 80, 200), (80, 200, 120), (230, 230, 230)]
        self.entities.append(SimulatedEntity(self.next_id, "person", x, y, vx, vy, random.choice(colors), (22, 22)))
        self.next_id += 1

    def _spawn_guard(self):
        x = random.uniform(100, self.width - 100)
        y = random.uniform(self.height * 0.55, self.height * 0.85)
        vx = random.uniform(-1.5, 1.5)
        vy = random.uniform(-0.8, 0.8)
        self.entities.append(SimulatedEntity(self.next_id, "person", x, y, vx, vy, (80, 180, 80), (24, 24)))
        self.next_id += 1

    def _spawn_intruder(self):
        # Starts from top/side and moves toward center/restricted zone
        x = random.uniform(50, self.width * 0.4)
        y = random.uniform(20, 80)
        target_x = self.width * 0.5 + random.uniform(-60, 60)
        target_y = self.height * 0.45 + random.uniform(-40, 40)
        angle = math.atan2(target_y - y, target_x - x)
        speed = random.uniform(1.5, 2.2)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        self.entities.append(SimulatedEntity(self.next_id, "intruder", x, y, vx, vy, (40, 40, 220), (24, 24)))
        self.next_id += 1

    def generate_frame(self) -> np.ndarray:
        self.frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw environment background based on scene type
        if self.scene_type == "traffic":
            self._render_traffic_bg(frame)
        elif self.scene_type == "pedestrian":
            self._render_pedestrian_bg(frame)
        else:
            self._render_perimeter_bg(frame)

        # Update and render entities
        for ent in self.entities:
            ent.update(self.width, self.height)
            self._render_entity(frame, ent)

        # Randomly maintain entity population
        if len(self.entities) < 8 and random.random() < 0.05:
            if self.scene_type == "traffic":
                self._spawn_vehicle()
            elif self.scene_type == "pedestrian":
                self._spawn_pedestrian()
            else:
                if not any(e.type == "intruder" for e in self.entities):
                    self._spawn_intruder()
                else:
                    self._spawn_guard()

        # Add subtle CCTV camera noise and timestamp watermark
        noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Subtle scanline effect
        frame[::4, :, :] = (frame[::4, :, :] * 0.94).astype(np.uint8)

        return frame

    def _render_traffic_bg(self, frame: np.ndarray):
        # Asphalt background
        frame[:] = (42, 45, 48)
        
        # Sidewalks / curbs
        cv2.rectangle(frame, (0, 0), (self.width, int(self.height * 0.22)), (90, 95, 100), -1)
        cv2.rectangle(frame, (0, int(self.height * 0.78)), (self.width, self.height), (90, 95, 100), -1)
        
        # Grass edges
        cv2.rectangle(frame, (0, 0), (self.width, int(self.height * 0.12)), (40, 75, 45), -1)
        cv2.rectangle(frame, (0, int(self.height * 0.88)), (self.width, self.height), (40, 75, 45), -1)

        # Road dashed center lines
        center_y = int(self.height * 0.5)
        dash_len = 40
        gap_len = 30
        for x in range(0, self.width, dash_len + gap_len):
            cv2.line(frame, (x, center_y), (x + dash_len, center_y), (60, 210, 240), 3)

        # Lane divider solid lines
        cv2.line(frame, (0, int(self.height * 0.23)), (self.width, int(self.height * 0.23)), (220, 220, 225), 2)
        cv2.line(frame, (0, int(self.height * 0.77)), (self.width, int(self.height * 0.77)), (220, 220, 225), 2)

    def _render_pedestrian_bg(self, frame: np.ndarray):
        # Tiled stone plaza background
        frame[:] = (75, 78, 85)
        # Grid paving pattern
        step = 60
        for x in range(0, self.width, step):
            cv2.line(frame, (x, 0), (x, self.height), (65, 68, 75), 1)
        for y in range(0, self.height, step):
            cv2.line(frame, (0, y), (self.width, y), (65, 68, 75), 1)

        # Central fountain / planter
        cv2.circle(frame, (int(self.width * 0.5), int(self.height * 0.5)), 70, (50, 100, 60), -1)
        cv2.circle(frame, (int(self.width * 0.5), int(self.height * 0.5)), 65, (160, 120, 60), -1)
        cv2.circle(frame, (int(self.width * 0.5), int(self.height * 0.5)), 35, (190, 160, 90), -1)

    def _render_perimeter_bg(self, frame: np.ndarray):
        # High security compound
        frame[:] = (55, 60, 65)

        # Security perimeter fence line (dashed)
        cv2.rectangle(frame, (int(self.width * 0.35), int(self.height * 0.25)), (int(self.width * 0.75), int(self.height * 0.75)), (75, 80, 85), -1)
        
        # High Security Building
        cv2.rectangle(frame, (int(self.width * 0.42), int(self.height * 0.32)), (int(self.width * 0.68), int(self.height * 0.68)), (35, 40, 45), -1)
        cv2.putText(frame, "RESTRICTED FACILITY", (int(self.width * 0.44), int(self.height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 105, 115), 1)

        # Hazard stripes around loading bay
        bay_y = int(self.height * 0.68)
        for x in range(int(self.width * 0.45), int(self.width * 0.65), 20):
            cv2.line(frame, (x, bay_y), (x + 10, bay_y + 12), (30, 200, 240), 2)

    def _render_entity(self, frame: np.ndarray, ent: SimulatedEntity):
        ix, iy = int(ent.x), int(ent.y)
        w, h = ent.width, ent.height

        # Soft shadow
        shadow_poly = np.array([
            [ix - w//2 + 4, iy + h//2 + 4],
            [ix + w//2 + 8, iy + h//2 + 4],
            [ix + w//2 + 4, iy - h//2 + 8],
            [ix - w//2 + 2, iy - h//2 + 6]
        ], dtype=np.int32)
        cv2.fillConvexPoly(frame, shadow_poly, (20, 22, 25))

        if ent.type in ["car", "truck"]:
            # Rotated vehicle body
            rect = ((ent.x, ent.y), (ent.width, ent.height), math.degrees(ent.angle))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillPoly(frame, [box], ent.color)
            cv2.polylines(frame, [box], True, (20, 20, 20), 2)
            
            # Windshield / Roof
            roof_rect = ((ent.x - ent.vx * 1.5, ent.y - ent.vy * 1.5), (ent.width * 0.45, ent.height * 0.75), math.degrees(ent.angle))
            roof_box = np.int32(cv2.boxPoints(roof_rect))
            cv2.fillPoly(frame, [roof_box], (40, 45, 50))

            # Headlights
            hl_dist = ent.width * 0.48
            hl_spread = ent.height * 0.35
            front_x = ent.x + math.cos(ent.angle) * hl_dist
            front_y = ent.y + math.sin(ent.angle) * hl_dist
            perp_x = -math.sin(ent.angle) * hl_spread
            perp_y = math.cos(ent.angle) * hl_spread
            cv2.circle(frame, (int(front_x + perp_x), int(front_y + perp_y)), 3, (180, 240, 255), -1)
            cv2.circle(frame, (int(front_x - perp_x), int(front_y - perp_y)), 3, (180, 240, 255), -1)

        else:  # person, guard, intruder
            # Pedestrian body (torso + head)
            cv2.circle(frame, (ix, iy), 10, ent.color, -1)
            cv2.circle(frame, (ix, iy), 10, (20, 20, 20), 1)
            # Head
            cv2.circle(frame, (ix, iy - 2), 5, (210, 180, 150), -1)
            # Directional facing marker
            facing_x = int(ix + math.cos(ent.angle) * 8)
            facing_y = int(iy + math.sin(ent.angle) * 8)
            cv2.line(frame, (ix, iy), (facing_x, facing_y), (255, 255, 255), 2)


class VideoSourceManager:
    """Manages input video streams (Webcam, File, RTSP, Synthetic Simulation)."""
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.source_type: str = "simulation"
        self.source_param: str = "perimeter"
        self.cap: Optional[cv2.VideoCapture] = None
        self.sim: Optional[SyntheticSceneGenerator] = None
        self.lock = threading.Lock()
        self.is_running = True
        self.last_frame: Optional[np.ndarray] = None
        self.last_timestamp: float = time.time()
        self.fps_actual: float = 0.0
        self.frame_counter: int = 0
        self.fps_timer: float = time.time()

        # Initialize default simulation
        self.set_source("simulation", "perimeter")

    def set_source(self, source_type: str, param: str = ""):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.source_type = source_type
            self.source_param = param

            if source_type == "webcam":
                cam_index = int(param) if param.isdigit() else 0
                self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.sim = None
            elif source_type in ["file", "rtsp"]:
                self.cap = cv2.VideoCapture(param)
                self.sim = None
            else:  # simulation
                scene = param if param in ["traffic", "pedestrian", "perimeter"] else "perimeter"
                self.sim = SyntheticSceneGenerator(width=960, height=540, scene_type=scene)
                self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            now = time.time()
            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret and self.source_type == "file":
                    # Loop video file back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame = frame
                else:
                    return False, self.last_frame
            elif self.sim is not None:
                self.last_frame = self.sim.generate_frame()
                ret = True
            else:
                return False, None

            # Calculate actual FPS
            self.frame_counter += 1
            if now - self.fps_timer >= 1.0:
                self.fps_actual = self.frame_counter / (now - self.fps_timer)
                self.frame_counter = 0
                self.fps_timer = now

            return True, self.last_frame.copy() if self.last_frame is not None else None

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
