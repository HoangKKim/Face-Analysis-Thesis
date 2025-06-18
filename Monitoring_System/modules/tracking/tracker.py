import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.detection.src.detector import Detector
from cfg.tracker_cfg import *

def is_the_same_person(self, bbox_face, bbox_body, iou_thres=0.1):
    # Kiểm tra nếu bbox_face nằm trong bbox_body
    fx1, fy1, fx2, fy2 = bbox_face
    bx1, by1, bx2, by2 = bbox_body

    if fx1 >= bx1 and fy1 >= by1 and fx2 <= bx2 and fy2 <= by2:
        return True  # mặt nằm trong thân → khả năng rất cao là cùng người

    # fallback: dùng IoU
    xA = max(fx1, bx1)
    yA = max(fy1, by1)
    xB = min(fx2, bx2)
    yB = min(fy2, by2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False

    faceArea = (fx2 - fx1) * (fy2 - fy1)
    bodyArea = (bx2 - bx1) * (by2 - by1)

    iou = interArea / float(faceArea + bodyArea - interArea)
    return iou >= iou_thres

class Tracker:
    def __init__(self, conf_threshold = CONF_THRESHOLD, max_age = MAX_AGE , n_init = N_INIT, class_names = CLASS_NAMES, class_id = CLASS_ID ):
        self.tracker = DeepSort(max_age, n_init)
        self.conf_threshold = conf_threshold
        self.class_names = class_names
        self.class_id = class_id
        self.person_id_map = {}
        self.next_person_id = 0

    def is_the_same_person(self, bbox_face, bbox_body, iou_thres=0.1):
        # Kiểm tra nếu bbox_face nằm trong bbox_body
        fx1, fy1, fx2, fy2 = bbox_face
        bx1, by1, bx2, by2 = bbox_body

        if fx1 >= bx1 and fy1 >= by1 and fx2 <= bx2 and fy2 <= by2:
            return True  # mặt nằm trong thân → khả năng rất cao là cùng người

        # fallback: dùng IoU
        xA = max(fx1, bx1)
        yA = max(fy1, by1)
        xB = min(fx2, bx2)
        yB = min(fy2, by2)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return False

        faceArea = (fx2 - fx1) * (fy2 - fy1)
        bodyArea = (bx2 - bx1) * (by2 - by1)

        iou = interArea / float(faceArea + bodyArea - interArea)
        return iou >= iou_thres

    def track_objects(self, frame, detector: Detector, frame_id):
        tracked_objects = []

        # Step 1: Detect objects
        detected_results = detector.detect_objects(frame)

        # Step 2: Prepare detections for DeepSORT
        detection = []
        for detected_obj in detected_results:
            (x1, y1, x2, y2) = detected_obj['bb_body']
            body_score = detected_obj['body_score']
            face_score = detected_obj['face_score']

            w, h = x2 - x1, y2 - y1

            if body_score >= self.conf_threshold:
                detection.append([[x1, y1, w, h], body_score, self.class_id[1]])  # body = 1
                
            if face_score >= self.conf_threshold:
                (fx1, fy1, fx2, fy2) = detected_obj['bb_face']
                fw, fh = fx2 - fx1, fy2 - fy1
                detection.append([[fx1, fy1, fw, fh], face_score, self.class_id[0]])  # face = 0

        # Step 3: DeepSORT tracking
        trackings = self.tracker.update_tracks(detection, frame=frame)

        # Step 4: Ghép face–body
        face_tracks = []
        body_tracks = []

        for tracked_obj in trackings:
            if frame_id > N_INIT and not tracked_obj.is_confirmed():
                continue
            track_id = tracked_obj.track_id
            bbox = list(map(int, tracked_obj.to_ltrb()))
            class_id = tracked_obj.get_det_class()

            if class_id == self.class_id[0]:  # face
                face_tracks.append((track_id, bbox))
            elif class_id == self.class_id[1]:  # body
                body_tracks.append((track_id, bbox))

        # Map track_id → person_id
        for face_id, face_bbox in face_tracks:
            matched = False
            for body_id, body_bbox in body_tracks:
                if self.is_the_same_person(face_bbox, body_bbox):
                    # Nếu 1 trong 2 đã có person_id → gán chung
                    pid = None
                    if face_id in self.person_id_map:
                        pid = self.person_id_map[face_id]
                    elif body_id in self.person_id_map:
                        pid = self.person_id_map[body_id]
                    else:
                        pid = self.next_person_id
                        self.next_person_id += 1

                    self.person_id_map[face_id] = pid
                    self.person_id_map[body_id] = pid
                    matched = True
                    break

            # Nếu không match body nào → vẫn gán person_id riêng cho face
            if not matched and face_id not in self.person_id_map:
                self.person_id_map[face_id] = self.next_person_id
                self.next_person_id += 1

        # Gán person_id cho body không match face
        for body_id, _ in body_tracks:
            if body_id not in self.person_id_map:
                self.person_id_map[body_id] = self.next_person_id
                self.next_person_id += 1

        # Step 5: Trả kết quả
        for tracked_obj in trackings:
            if frame_id > N_INIT and not tracked_obj.is_confirmed():
                continue

            track_id = tracked_obj.track_id
            bbox = list(map(int, tracked_obj.to_ltrb()))
            class_id = tracked_obj.get_det_class()
            person_id = self.person_id_map.get(track_id)

            tracked_objects.append({
                'track_id': track_id,
                'label': self.class_names[class_id],
                'bbox': bbox,
                'person_id': person_id
            })

        return {
            "frame_id": frame_id,
            "objects": tracked_objects
        }
