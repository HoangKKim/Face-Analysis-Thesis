import cv2
import numpy as np
import torch 

from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.detection.src.detector import YOLO_Detector
from cfg.tracker_cfg import *

class Tracker:
    def __init__(self, conf_threshold=0.4, max_age=50, n_init=3, detection_interval = 3, resize_factor = 0.5):
        self.n_init = n_init
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=0.2, nn_budget=200, embedder="mobilenet")
        self.conf_threshold = conf_threshold
        self.person_id_map = {}
        self.next_person_id = 0
        self.detector = YOLO_Detector()

        self.detection_interval = detection_interval  # Run detection every N frames
        self.resize_factor = resize_factor  # Resize frame for detection
        self.last_detections = []  # Cache last detections
        self.frame_count = 0

    def _resize_frame(self, frame):
        """Resize frame for faster detection"""
        if self.resize_factor == 1.0:
            return frame, 1.0, 1.0
            
        h, w = frame.shape[:2]
        new_w = int(w * self.resize_factor)
        new_h = int(h * self.resize_factor)
        resized = cv2.resize(frame, (new_w, new_h))
        
        scale_x = w / new_w
        scale_y = h / new_h
        return resized, scale_x, scale_y
    
    def _scale_detections(self, detections, scale_x, scale_y):
        """Scale detections back to original frame size"""
        scaled_detections = []
        for det in detections:
            scaled_det = det.copy()
            x1, y1, x2, y2 = det['bb_body']
            scaled_det['bb_body'] = [
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            ]
            scaled_detections.append(scaled_det)
        return scaled_detections

    def track_objects(self, frame: np.ndarray, frame_id: int):
        tracked_objects = []
        self.frame_count += 1
        
        # Step 1: Run detection only every N frames
        if self.frame_count % self.detection_interval == 1:
            # Resize frame for faster detection
            resized_frame, scale_x, scale_y = self._resize_frame(frame)
            
            # Detect persons on resized frame
            detections = self.detector.detect_person(resized_frame, conf_thres =  0.65, iou_thres = 0.35)
            
            # Scale detections back to original size
            self.last_detections = self._scale_detections(detections, scale_x, scale_y)
        
        # Use cached detections for intermediate frames
        detections = self.last_detections

        # Step 2: Format for DeepSORT - convert to [x, y, w, h] format
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bb_body']
            score = det['body_score']
            if score >= self.conf_threshold:
                w, h = x2 - x1, y2 - y1
                # Ensure positive width/height
                if w > 0 and h > 0:
                    formatted_detections.append([[x1, y1, w, h], score, 0])

        # Step 3: Update tracker with current frame
        trackings = self.tracker.update_tracks(formatted_detections, frame=frame)

        # Step 4: Process confirmed tracks
        
        for trk in trackings:
            # Skip unconfirmed tracks in early frames
            if not trk.is_confirmed() and frame_id > self.n_init:
                continue    

            track_id = trk.track_id
            bbox = list(map(int, trk.to_ltrb()))
            x1, y1, x2, y2 = bbox

            # Assign stable person_id
            if track_id not in self.person_id_map:
                self.person_id_map[track_id] = self.next_person_id
                self.next_person_id += 1

            person_id = self.person_id_map[track_id]

            tracked_objects.append({
                'label': 'body',
                'track_id': track_id,
                'person_id': person_id,
                'bbox': [x1, y1, x2, y2],
            })

        return {
            "frame_id": frame_id,
            "objects": tracked_objects
        }