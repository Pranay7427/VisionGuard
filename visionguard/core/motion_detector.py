"""
VisionGuard - CPU-Optimized Motion Detection & Heatmap Engine
Implements MOG2/KNN Background Subtraction, Shadow Suppression, Contour Analysis, and Thermal Heatmap.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class MotionDetector:
    def __init__(self, history: int = 500, var_threshold: int = 25, detect_shadows: bool = True):
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self.heatmap_matrix: Optional[np.ndarray] = None
        self.heatmap_decay: float = 0.94
        self.motion_intensity: float = 0.0
        self.motion_percentage: float = 0.0

    def update_parameters(self, sensitivity: int, min_area: int):
        # sensitivity 1-100 maps to varThreshold 60 down to 10
        self.var_threshold = int(60 - (sensitivity / 100.0) * 50)
        self.bg_subtractor.setVarThreshold(self.var_threshold)

    def process_frame(self, frame: np.ndarray, min_area: int = 600) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Processes a frame for motion detection.
        Returns:
            - fg_mask: cleaned binary foreground mask
            - motion_boxes: list of dicts with {'box': (x, y, w, h), 'centroid': (cx, cy), 'area': area}
            - colored_heatmap: 3-channel visual thermal heatmap
        """
        h, w = frame.shape[:2]

        # Initialize or resize heatmap buffer
        if self.heatmap_matrix is None or self.heatmap_matrix.shape != (h, w):
            self.heatmap_matrix = np.zeros((h, w), dtype=np.float32)

        # 1. Apply Gaussian blur to reduce high-frequency sensor noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. Compute background subtraction mask
        fg_mask_raw = self.bg_subtractor.apply(blurred)

        # 3. Suppress shadows (shadow pixels have value 127 in OpenCV MOG2)
        if self.detect_shadows:
            _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)
        else:
            _, fg_mask = cv2.threshold(fg_mask_raw, 127, 255, cv2.THRESH_BINARY)

        # 4. Morphological operations to remove salt-and-pepper noise and connect broken blobs
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        fg_mask = cv2.dilate(fg_mask, self.kernel_dilate, iterations=2)

        # 5. Extract contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        total_motion_pixels = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + bw // 2
                cy = y + bh // 2
                motion_boxes.append({
                    "box": (x, y, bw, bh),
                    "centroid": (cx, cy),
                    "area": float(area),
                    "contour": cnt
                })
                total_motion_pixels += area

        self.motion_percentage = min(100.0, (total_motion_pixels / (w * h)) * 100.0)
        self.motion_intensity = len(motion_boxes)

        # 6. Update cumulative motion heatmap
        # Add current binary mask to accumulation matrix with decay
        normalized_mask = (fg_mask > 0).astype(np.float32)
        self.heatmap_matrix = self.heatmap_matrix * self.heatmap_decay + normalized_mask * 0.25
        np.clip(self.heatmap_matrix, 0.0, 1.0, out=self.heatmap_matrix)

        # Convert heatmap to colored representation
        heat_u8 = (self.heatmap_matrix * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

        return fg_mask, motion_boxes, colored_heatmap

    def reset_history(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        if self.heatmap_matrix is not None:
            self.heatmap_matrix.fill(0)
