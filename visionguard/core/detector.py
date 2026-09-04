"""
VisionGuard - CPU-Optimized Object Detector & Classifier
Combines OpenCV DNN MobileNet-SSD, HOG+SVM Person Detector, and Geometric Heuristic Classifiers.
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Dict, Tuple, Any, Optional

# COCO / MobileNet-SSD standard classes of interest
MOBILENET_CLASSES = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
    10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
    15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train",
    20: "tvmonitor"
}

PERSON_CLASSES = {"person", "pedestrian", "guard", "intruder"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle", "vehicle"}


class ObjectDetector:
    def __init__(self, model_type: str = "contour_heuristic", conf_threshold: float = 0.45):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.net = None
        self.hog = None
        self._init_models()

    def _init_models(self):
        # Initialize OpenCV HOG Person Detector
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            print(f"[Detector] HOG init warning: {e}")

        # Check if MobileNet-SSD model files exist
        model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
        os.makedirs(model_dir, exist_ok=True)
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")

        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"[Detector] DNN load error: {e}")
                self.net = None

    def detect(self, frame: np.ndarray, motion_boxes: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detects objects in frame.
        Returns list of dicts: {'box': (x,y,w,h), 'centroid': (cx, cy), 'label': str, 'confidence': float}
        """
        if self.model_type == "mobilenet_ssd" and self.net is not None:
            return self._detect_dnn(frame)
        elif self.model_type == "hog_svm" and self.hog is not None:
            return self._detect_hog(frame, motion_boxes)
        else:
            return self._detect_heuristic(frame, motion_boxes)

    def _detect_dnn(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])
                label = MOBILENET_CLASSES.get(idx, "object")
                
                # Filter security-relevant classes
                if label in PERSON_CLASSES or label in VEHICLE_CLASSES:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    bw = max(10, min(w - x1, x2 - x1))
                    bh = max(10, min(h - y1, y2 - y1))
                    cx = x1 + bw // 2
                    cy = y1 + bh // 2
                    results.append({
                        "box": (x1, y1, bw, bh),
                        "centroid": (cx, cy),
                        "label": label,
                        "confidence": round(confidence, 2)
                    })
        return results

    def _detect_hog(self, frame: np.ndarray, motion_boxes: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        results = []
        # Downscale for CPU speed
        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)

        for i, (rx, ry, rw, rh) in enumerate(rects):
            weight = float(weights[i]) if i < len(weights) else 0.7
            x = int(rx / scale)
            y = int(ry / scale)
            bw = int(rw / scale)
            bh = int(rh / scale)
            cx = x + bw // 2
            cy = y + bh // 2
            results.append({
                "box": (x, y, bw, bh),
                "centroid": (cx, cy),
                "label": "person",
                "confidence": min(0.99, max(0.5, round(weight, 2)))
            })

        # Augment with motion boxes if HOG missed vehicles
        if motion_boxes:
            for mb in motion_boxes:
                bx, by, bw, bh = mb["box"]
                # If not overlapping with a detected person, classify geometrically
                has_overlap = any(
                    abs(mb["centroid"][0] - r["centroid"][0]) < (bw + r["box"][2]) // 3 and
                    abs(mb["centroid"][1] - r["centroid"][1]) < (bh + r["box"][3]) // 3
                    for r in results
                )
                if not has_overlap:
                    label, conf = self._classify_box_geometry(bw, bh)
                    results.append({
                        "box": (bx, by, bw, bh),
                        "centroid": mb["centroid"],
                        "label": label,
                        "confidence": conf
                    })
        return results

    def _detect_heuristic(self, frame: np.ndarray, motion_boxes: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        results = []
        if not motion_boxes:
            return results

        for mb in motion_boxes:
            bx, by, bw, bh = mb["box"]
            cx, cy = mb["centroid"]
            label, conf = self._classify_box_geometry(bw, bh)
            results.append({
                "box": (bx, by, bw, bh),
                "centroid": (cx, cy),
                "label": label,
                "confidence": conf
            })
        return results

    def _classify_box_geometry(self, bw: int, bh: int) -> Tuple[str, float]:
        aspect_ratio = bh / float(bw) if bw > 0 else 1.0
        area = bw * bh

        if aspect_ratio >= 1.3:
            # Tall aspect ratio -> Person / Pedestrian
            return "person", 0.85
        elif aspect_ratio <= 0.85 and area > 1400:
            # Wide aspect ratio and large area -> Vehicle / Car
            return "car", 0.88
        elif area > 3500:
            # Large moving entity -> Vehicle
            return "vehicle", 0.82
        elif area < 600:
            return "motion", 0.65
        else:
            return "object", 0.75
