import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.detection.src.detector import Detector
from cfg.tracker_cfg import *

class Tracker:
    def __init__(self, conf_threshold = CONF_THRESHOLD, max_age = MAX_AGE , n_init = N_INIT, class_names = CLASS_NAMES, class_id = CLASS_ID ):
        self.tracker = DeepSort(max_age, n_init)
        self.conf_threshold = conf_threshold
        self.class_names = class_names
        self.class_id = class_id
    
    def track_objects(self, frame, detector: Detector, frame_id):
        tracked_objects = []

        # Step 1: Detect objects
        detected_results = detector.detect_objects(frame)
        
        # Step 2: Prepare detections for DeepSORT
        detection = []
        for detected_obj in detected_results:
            (x1, y1, x2, y2) = detected_obj['bb_body']
            (fx1, fy1, fx2, fy2) = detected_obj['bb_face']
            body_score = detected_obj['body_score']
            face_score = detected_obj['face_score']
        
            w, h = x2-x1, y2-y1
            fw, fh = fx2-fx1, fy2-fy1

            if body_score >= self.conf_threshold:
                detection.append([[x1, y1, w, h], body_score, self.class_id[1]])  # body = 1

            if face_score >= self.conf_threshold:
                detection.append([[fx1, fy1, fw, fh], face_score, self.class_id[0]])  # face = 0

        # Step 3: DeepSORT tracking
        trackings = self.tracker.update_tracks(detection, frame = frame)

        # Step 4: Parse tracked objects
        for tracked_obj in trackings:
            # make sure it is the real object and need to be tracked (after n_init frames)
            if(frame_id > N_INIT):
                if not tracked_obj.is_confirmed():
                    continue

            track_id = tracked_obj.track_id
            ltrb = tracked_obj.to_ltrb()
            class_id = tracked_obj.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)

            tracked_objects.append({
                'track_id': track_id,
                'label': self.class_names[class_id],
                'bbox': [x1, y1, x2, y2]
            })
        
        return {
            "frame_id": frame_id,
            "objects": tracked_objects
        }




    








