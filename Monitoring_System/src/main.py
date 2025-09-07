import cv2
import numpy as np
import json
import os
import shutil
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import List

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from cfg.keyframes_extractor_cfg import *
from cfg.recognizer_cfg import *
from cfg.expression_cfg import *

from modules.tracking.tracker import Tracker
from modules.detection.src.detector import YOLO_Detector
from modules.recognizer.recognizer import FaceRecognizer 
from modules.pose_estimation.src.inference import Inference
from modules.expression.fer_classifier import *

from utils.logger import *
from src.evaluate_result import * 
from src.supported_functions import *

MIN_WIDTH = 150
MIN_HEIGHT = 150
max_width_ratio = 0.15

import os
import cv2
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def run_tracking_and_detection(
    cap,
    total_frames,
    jumping_step,
    frame_width,
    tracker,
    detector,
    output_tracking_dir,
    min_width=30,
    min_height=30,
    max_width_ratio=0.6,
    logger=None
):

    frame_id = 0
    person_tracking_info = defaultdict(dict)

    logger.info("Starting to process video") if logger else print("Processing video...")

    pbar = tqdm(total=total_frames, desc="🔄 Processing video", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_for_crop = frame.copy()

        tracked_data = tracker.track_objects(frame, frame_id)

        if (frame_id % jumping_step == 0):
            for obj in tracked_data["objects"]:
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Bỏ các bbox không hợp lệ
                if w < min_width or h < min_height or w > max_width_ratio * frame_width:
                    continue

                label = obj['label']  # "face" or "body"
                person_id = obj.get('person_id', obj['track_id'])

                folder_name = f"person_{int(person_id):02d}"
                save_dir = os.path.join(output_tracking_dir, folder_name, label)
                save_dir_face = os.path.join(output_tracking_dir, folder_name, 'face')

                Path(save_dir).mkdir(parents=True, exist_ok=True)
                Path(save_dir_face).mkdir(parents=True, exist_ok=True)

                # Clip bbox về trong ảnh
                img_h, img_w = frame_for_crop.shape[:2]
                x1_clip, x2_clip = max(0, x1), min(img_w, x2)
                y1_clip, y2_clip = max(0, y1), min(img_h, y2)

                if x2_clip > x1_clip and y2_clip > y1_clip:
                    crop_img = frame_for_crop[y1_clip:y2_clip, x1_clip:x2_clip]
                else:
                    crop_img = None

                if crop_img is not None and crop_img.size > 0:
                    # Lưu ảnh body
                    save_path = os.path.join(save_dir, f"frame_{frame_id:05d}.jpg")
                    cv2.imwrite(save_path, crop_img)

                    bbox_int = [int(x1_clip), int(y1_clip), int(x2_clip), int(y2_clip)]
                    if frame_id not in person_tracking_info[person_id]:
                        person_tracking_info[person_id][frame_id] = {}
                    person_tracking_info[person_id][frame_id][label] = bbox_int

                    # Dò và lưu face (nếu có)
                    try:
                        face_bboxes = detector.detect_face(crop_img)
                        if face_bboxes:
                            fx1, fy1, fx2, fy2 = face_bboxes[0]
                            face_x1 = fx1 + x1_clip
                            face_y1 = fy1 + y1_clip
                            face_x2 = fx2 + x1_clip
                            face_y2 = fy2 + y1_clip

                            face_x1 = max(0, min(img_w, face_x1))
                            face_x2 = max(0, min(img_w, face_x2))
                            face_y1 = max(0, min(img_h, face_y1))
                            face_y2 = max(0, min(img_h, face_y2))

                            crop_face = frame_for_crop[face_y1:face_y2, face_x1:face_x2]
                            if crop_face is not None and crop_face.size > 0:
                                save_path_face = os.path.join(save_dir_face, f"frame_{frame_id:05d}.jpg")
                                cv2.imwrite(save_path_face, crop_face)

                                bbox_face_int = [int(face_x1), int(face_y1), int(face_x2), int(face_y2)]
                                person_tracking_info[person_id][frame_id]['face_bbox'] = bbox_face_int
                            else:
                                person_tracking_info[person_id][frame_id]['face_bbox'] = 'Unknown'
                        else:
                            person_tracking_info[person_id][frame_id]['face_bbox'] = 'Unknown'
                    except Exception as e:
                        print(f"[ERROR] Face detection failed at frame {frame_id}: {e}")
                        person_tracking_info[person_id][frame_id]['face_bbox'] = 'Error'

        pbar.update(1)
        frame_id += 1

    cap.release()
    pbar.close()

    # Ghi thông tin tracking ra json
    for person_id, frames_dict in person_tracking_info.items():
        folder_name = f"person_{int(person_id):02d}"
        person_dir = os.path.join(output_tracking_dir, folder_name)
        json_path = os.path.join(person_dir, "infor.json")

        frames_list = []
        for fid in sorted(frames_dict.keys()):
            frame_info = {"frame_id": fid}
            frame_info.update(frames_dict[fid])
            frames_list.append(frame_info)

        with open(json_path, "w") as f:
            json.dump(frames_list, f, indent=2)

    if logger:
        logger.info("Tracking done, results saved.")


import os
import shutil
from typing import List

def recognize_and_merge_output(
    tracking_dir: str,
    recognizing_dir: str,
    database_dir: str,
    face_recognizer,
    logger,
    default_threshold = 0.8,
    default_margin = 0.08,
) -> None:
    
    logger.info("Start face recognition and output classification")

    existed_person_dir: List[str] = []
    unknown_dir: List[str] = []

    for folder in os.listdir(tracking_dir):
        person_dir = os.path.join(tracking_dir, folder)
        face_dir = os.path.join(person_dir, 'face')

        if not os.path.exists(face_dir) or not os.listdir(face_dir):
            logger.info(f'{person_dir} is skipped (no faces found)')
            continue

        label = face_recognizer.recognize(face_dir)   
        logger.debug(f"Recognition result for {folder} to label: {label}")

        if label != 'Unknown':
            identified_folder = os.path.join(recognizing_dir, label)
            logger.info(f"{person_dir} to {label}")

            if identified_folder not in existed_person_dir:
                existed_person_dir.append(identified_folder)
                shutil.copytree(person_dir, identified_folder, dirs_exist_ok=True)
                logger.info(f"Created new folder {identified_folder}")
            else:
                move_images_and_merge_json(person_dir, identified_folder, logger)
                logger.info(f"Merged into existing folder {identified_folder}")
        else:
            unknown_name = f"unknown_{(len(unknown_dir) + 1):02d}"
            unknown_folder = os.path.join(recognizing_dir, unknown_name)
            unknown_dir.append(unknown_name)
            shutil.copytree(person_dir, unknown_folder, dirs_exist_ok=True)
            logger.warning(f"{person_dir} is unrecognized to {unknown_folder}")

def analyze_emotion_behavior(
    output_recognizing_dir: str,
    fer_predictor,
    fer_label: dict,
    pose_predictor,
    evaluator,
    fps: int,
    output_csv_path: str,
    logger
):
    final_dataframe = init_dataframes()

    for person_dir in os.listdir(output_recognizing_dir):
        if person_dir.startswith("unknown_"):
            continue
        
        person_path = os.path.join(output_recognizing_dir, person_dir)
        face_folder = os.path.join(person_path, 'face')
        body_folder = os.path.join(person_path, 'body')
        
        result = []
        logger.info(f"Processing {person_dir}")

        for body_file in os.listdir(body_folder):
            face_img_path = os.path.join(face_folder, body_file)
            body_img_path = os.path.join(body_folder, body_file)

                     
            body_img = cv2.imread(body_img_path)
            
            # Dự đoán cảm xúc
            if os.path.exists(face_img_path):
                predicted_emotion, score = fer_predictor.inference(face_img_path)
                emotion = fer_label[predicted_emotion]
                if (emotion == 'Sad' and score < 0.85):
                    emotion = 'Neutral'
            else:
                emotion = 'Unknown'

            # Dự đoán hành vi
            behavior, score = pose_predictor.predict(body_img)

            result.append({
                'body_image': body_img_path,
                "predicted_behaviour": behavior,
                'face_image': face_img_path,
                'predicted_emotion': emotion
            })
        
        concentration_path = os.path.join(person_path, 'concentration.json')
        # with open(concentration_path, 'w') as f:
        #     json.dump(result, f, indent=4)

        evaluator.update_output_dir(person_path)
        evaluator.load_report(concentration_path)
        evaluator.classify()

        aggregated_result = evaluator.aggregate_per_second_and_save(concentration_path, fps)
        mean_emotion = [sec['mean_emotion'] for sec in aggregated_result.values()]
        mean_behavior = [sec['mean_behavior'] for sec in aggregated_result.values()]
        mean_fusion = [sec['mean_fusion'] for sec in aggregated_result.values()]


        focus_score_result = evaluator.compute_focus_score(mean_fusion)
        focus_score_result['student_id'] = person_dir

        evaluator.draw_single_curve(mean_emotion, 'emotion') 
        evaluator.draw_single_curve(mean_behavior, 'behavior')
        evaluator.draw_single_curve(mean_fusion, 'concentration', focus_score_result['mean'])

        if focus_score_result['focus_score'] < -0.15:
            focus_score_result['result'] = 'NEGATIVE'
        elif focus_score_result['focus_score'] > 0.15:
            focus_score_result['result'] = 'POSITIVE'
        else:
            focus_score_result['result'] = 'NEUTRAL'
            
        final_dataframe = pd.concat([final_dataframe, pd.DataFrame([focus_score_result])], ignore_index=True)
        merge_final_info_for_person(person_path, os.path.join(person_path, 'infor.json'), concentration_path)

    final_dataframe = final_dataframe.sort_values(by="student_id", ascending=True)
    final_dataframe.to_csv(output_csv_path, index=False)

    merge_concentration_and_mode_summary(output_recognizing_dir)
    return final_dataframe


def main(video_path, log_callback = None):
    
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    VIDEO_PATH = video_path
    VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_DIR = os.path.join('output', VIDEO_NAME)
    
    # create logger
    logger = setup_logger(log_path=f'{OUTPUT_DIR}/process_log.txt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Create {OUTPUT_DIR} successfully")

    DATABASE_DIR = ['database/feature/feature_vectors.npy', 'database/feature/labels.txt']

    # -------------- define modulers ---------------
    TRACKER = Tracker()
    DETECTOR = YOLO_Detector()
    FACE_RECOGNIZER = FaceRecognizer()
    POSE_PREDICTOR = Inference()
    FER_PREDICTOR = FERClassifier(root_dir = '')
    FER_PREDICTOR.load_model(FER_MODEL)
    EVALUATOR = FocusEvaluator(logger)

    # -------------- define global files/ dir --------------

    # Define Output's dirs
    OUTPUT_TRACKING_DIR = os.path.join(OUTPUT_DIR,'detection_and_tracking')
    OUTPUT_RECOGNIZING_DIR = os.path.join(OUTPUT_DIR, 'students_result')
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'total_result.csv')

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    os.makedirs(OUTPUT_TRACKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RECOGNIZING_DIR, exist_ok=True)

    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)    

    jumping_step = int(fps / NUM_KEYFRAMES_GET_PER_FRAME)

    if not cap.isOpened():
        logger.error("Cannot open video:", VIDEO_PATH)
        return
    # ------------------------------------------------------------- Step 1: detect & tracking ----------------------------------------------------------------
    log("Detecting and Tracking ...")
    run_tracking_and_detection( cap=cap, total_frames=total_frames, 
                               jumping_step=jumping_step, frame_width=frame_width, 
                               tracker=TRACKER, detector=DETECTOR, 
                               output_tracking_dir=OUTPUT_TRACKING_DIR, logger=logger
    )
    logger.info("Tracking done, results saved at {OUTPUT_TRACKING_DIR}.")

    # ------------------------------------------ recognized face id & refractor output -------------------------------------------------------
    log("Recognizing Student ID ...")
    recognize_and_merge_output(OUTPUT_TRACKING_DIR, OUTPUT_RECOGNIZING_DIR, 
                               DATABASE_DIR, FACE_RECOGNIZER, 
                               logger)

    shutil.rmtree(OUTPUT_TRACKING_DIR)

    log("Process Unknown Folder")    
    # process_unknown_folder(OUTPUT_RECOGNIZING_DIR, logger)
    show_unknown_folder_ui_toplevel(OUTPUT_RECOGNIZING_DIR, logger)  # <- BLOCKING
    # # ----------------------------------------------- behavior -----------------------------------------------------------
    log("Predicting emotions and behaviours ...")
    
    logger.info("Predicting emotions and behaviours")

    final_df = analyze_emotion_behavior(OUTPUT_RECOGNIZING_DIR, FER_PREDICTOR, 
                                        FER_LABEL, POSE_PREDICTOR, 
                                        EVALUATOR, fps, OUTPUT_CSV_PATH, logger)
    merge_concentration_and_mode_summary(OUTPUT_RECOGNIZING_DIR)

    generate_final_charts(final_df, OUTPUT_RECOGNIZING_DIR)

    # plot on video 
    logger.info(f"Outputing video")
    log("Exporting output video ...")

    person_json = [os.path.join(OUTPUT_RECOGNIZING_DIR, person_dir, 'record.json') 
                   for person_dir in os.listdir(OUTPUT_RECOGNIZING_DIR)]
    visualize_person_annotations_on_video(
        VIDEO_PATH,
        person_json,
        os.path.join(OUTPUT_DIR, 'final_video.mp4'),
        [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    )
    return OUTPUT_DIR
        
if __name__ == "__main__":
    import argparse
    from src.prepocess_vid import * 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, required=True, help="path of input video"
    )
    
    args = parser.parse_args()

    # downsample_video('input/input_video/backup_demo.mov', 'input/input_video/backup_demo_fps60.mp4', 60)    

    # main('input/input_video/backup_demo_fps60.mp4')
    output_folder = main(args.video)    

    
