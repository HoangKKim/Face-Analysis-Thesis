import cv2
import numpy as np
import json
import os
import shutil
import pandas as pd
import tkinter as tk
import re
import os

from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from zipfile import ZipFile

from pathlib import Path
from tqdm import tqdm

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from cfg.keyframes_extractor_cfg import *
from cfg.recognizer_cfg import *
from cfg.expression_cfg import *

from modules.expression.fer_classifier import *

from utils.logger import *
from src.evaluate_result import * 

from tqdm import tqdm


def move_images(src_folder, dst_folder, logger):
    for subfolder in ['face', 'body']:
        src_subfolder = os.path.join(src_folder, subfolder)
        dst_subfolder = os.path.join(dst_folder, subfolder)

        if not os.path.exists(src_subfolder):
            print(f"Cannot find {src_subfolder}.")
            continue

        os.makedirs(dst_subfolder, exist_ok=True)

        for filename in os.listdir(src_subfolder):
            src_path = os.path.join(src_subfolder, filename)
            dst_path = os.path.join(dst_subfolder, filename)
            shutil.copy(src_path, dst_path)

        logger.info(f"Move all image from {src_folder} to {dst_folder}")

def move_images_and_merge_json(src_dir, dst_dir, logger):
    move_images(src_dir, dst_dir, logger)

    # merge info.json (if exist)
    src_json = os.path.join(src_dir, 'infor.json')
    dst_json = os.path.join(dst_dir, 'infor.json')

    if os.path.exists(src_json):
        with open(src_json, 'r') as f:
            src_data = json.load(f)
    else:
        src_data = []

    if os.path.exists(dst_json):
        with open(dst_json, 'r') as f:
            dst_data = json.load(f)
    else:
        dst_data = []

    merged_data = dst_data + src_data
    # Loại bỏ frame_id trùng lặp nếu cần (ưu tiên cái cũ / mới)
    merged_data = {item['frame_id']: item for item in merged_data}  # unique theo frame_id
    merged_data = [v for _, v in sorted(merged_data.items())]

    with open(dst_json, 'w') as f:
        json.dump(merged_data, f, indent=2)


def init_dataframes():
    standard_columns = ['student_id', 'ratio_positive', 'ratio_neutral', 'ratio_negative', 'mean', 'std', 'focus_score', 'result']
    df = pd.DataFrame(columns=standard_columns)
    return df

def extract_frame_id_from_filename(filename):
    try:
        return int(filename.replace("frame_", "").replace(".jpg", ""))
    except:
        return None


def merge_final_info_for_person(root_dir, infor_path, concentration_path):
    if not os.path.exists(infor_path) or not os.path.exists(concentration_path):
        raise FileNotFoundError("Error")
    
    with open(infor_path, 'r') as f:
        bbox_list = json.load(f)
        bbox_map = {item['frame_id']: item for item in bbox_list}

    with open(concentration_path, 'r') as f:
        concentration_list = json.load(f)

    merged_result = []

    for item in concentration_list:
        frame_id = extract_frame_id_from_filename(os.path.basename(item.get("body_image", "")))
        if frame_id is None:
            continue

        merged_item = {
            "frame_id": frame_id,
            "emotion": item.get("predicted_emotion", ""),
            "behavior": item.get("predicted_behaviour", "")
        }

        bbox_info = bbox_map.get(frame_id)
        if bbox_info:
            merged_item["face"] = bbox_info.get("face_bbox", "")
            merged_item["body"] = bbox_info.get("body", "")

        merged_result.append(merged_item)
    output_path = os.path.join(root_dir, "combined.json")
    with open(output_path, 'w') as f:
        json.dump(merged_result, f, indent=2)

def draw_annotation(frame, annotation, color=(0, 255, 0)):
    face_box = annotation.get("face")
    body_box = annotation.get("body")
    emotion = annotation.get("mode_emotion", "")
    behavior = annotation.get("mode_behavior", "")
    person_id = annotation.get("person_id", "")

    if face_box and face_box != 'Unknown':
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"E: {emotion}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if body_box:
        x1, y1, x2, y2 = body_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"B: {behavior}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def visualize_person_annotations_on_video(
    input_video_path: str,
    person_json_paths: list,        # concentration.json
    output_video_path: str,
    color_map=None
):
    # ====== 1. Chuẩn bị thông tin color ======
    if color_map is None:
        color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    # ====== 2. Load annotation cho từng người ======
    annotations_per_person = {}
    max_frame_id = 0

    for idx, json_path in enumerate(person_json_paths):
        with open(json_path, 'r') as f:
            person_data = json.load(f)

        person_id = os.path.basename(os.path.dirname(json_path))
        color = color_map[idx % len(color_map)]

        person_track = {}
        for entry in person_data:
            frame_id = entry["frame_id"]
            entry["person_id"] = person_id
            entry["color"] = color
            person_track[frame_id] = entry
            max_frame_id = max(max_frame_id, frame_id)

        annotations_per_person[person_id] = {
            "color": color,
            "track": person_track,
            "last_seen": None  # giữ thông tin frame trước đó
        }

    # ====== 3. Mở video để ghi lại ======
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    with tqdm(total=total_frames, desc="Visualizing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for person_id, data in annotations_per_person.items():
                track = data["track"]
                color = data["color"]

                if frame_id in track:
                    annotations_per_person[person_id]["last_seen"] = track[frame_id]

                ann = annotations_per_person[person_id]["last_seen"]
                if ann:
                    draw_annotation(frame, ann, color)

            out.write(frame)
            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()

def process_unknown_folder(root_folder, logger):
    count = 0
    for file in os.listdir(root_folder):
        # print(file)
        if file.startswith("unknown_"):
            # continue
            unknow_dir = os.path.join(root_folder, file)
            # allow user define unknow dir
            belonged_to = input(f'Review the {unknow_dir} and enter the student id which it is belonged to <Ex: Student_XX>: ')
            belonged_to = belonged_to.strip()

            identified_dir = os.path.join(root_folder, belonged_to)

            if not os.path.exists(identified_dir):
                os.makedirs(identified_dir)
                logger.info(f"[CREATE] Created new directory: {identified_dir}")
                
            # rename the unknow folder
            move_images_and_merge_json(unknow_dir, identified_dir, logger)
            shutil.rmtree(unknow_dir)
            logger.info(f"Deleted directory: {unknow_dir}")

            logger.info(f"Merged images into {identified_dir}")
            count +=1
    logger.info(f"Done to process {count} unknow dir")

import os
import tkinter as tk
from PIL import Image, ImageTk
import shutil

def show_unknown_folder_ui_toplevel(folder_path, logger):
    top = tk.Toplevel()
    top.title("🕵️ Recheck Unknown Folders")

    label = tk.Label(top, text="Review unknown folders and assign to correct students", font=("Arial", 12))
    label.pack(pady=10)

    # ===== Scrollable Canvas Setup =====
    canvas = tk.Canvas(top, width=600, height=400)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # ===== Render Unknown Items =====
    for file in os.listdir(folder_path):
        if file.startswith("unknown_"):
            sub_frame = tk.Frame(scrollable_frame)
            sub_frame.pack(fill='x', pady=5)

            img_path = None
            folder_full_path = os.path.join(folder_path, file, 'body')
            for f in os.listdir(folder_full_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_full_path, f)
                    break

            if img_path:
                try:
                    img = Image.open(img_path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)

                    img_label = tk.Label(sub_frame, image=photo)
                    img_label.image = photo
                    img_label.pack(side='left', padx=5)
                except Exception as e:
                    logger.warning(f"Could not load image from {img_path}: {e}")

            tk.Label(sub_frame, text=file, width=20).pack(side='left', padx=5)

            entry = tk.Entry(sub_frame)
            entry.pack(side='left', padx=5)

            def assign_and_rename(entry=entry, file=file, frame=sub_frame, btn=None):
                student_id = entry.get().strip()
                if student_id:
                    dst = os.path.join(folder_path, student_id)
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                        logger.info(f"[CREATE] Created {dst}")
                    move_images_and_merge_json(os.path.join(folder_path, file), dst, logger)
                    shutil.rmtree(os.path.join(folder_path, file))
                    logger.info(f"[MERGE] Merged {file} to {student_id}")
                    if btn:
                        btn.config(state=tk.DISABLED)
                    frame.destroy()

            assign_btn = tk.Button(sub_frame, text="Assign")
            assign_btn.config(command=lambda e=entry, f=file, fr=sub_frame, b=assign_btn: assign_and_rename(e, f, fr, b))
            assign_btn.pack(side='left', padx=5)

    # ===== Done Button =====
    def on_done():
        top.destroy()

    tk.Button(top, text="✅ Done", command=on_done).pack(pady=10)

    top.grab_set()
    top.wait_window()


def merge_concentration_and_mode_summary(person_folders: list):
    for folder in os.listdir(person_folders):
        folder_path = os.path.join(person_folders, folder)
        concentration_path = os.path.join(folder_path, "combined.json")
        mode_summary_path = os.path.join(folder_path, "concentration_mode_summary.json")

        output_path = os.path.join(folder_path, "record.json")
        print(output_path)

        if not os.path.exists(concentration_path) or not os.path.exists(mode_summary_path):
            print(f"[SKIP] Missing file in {folder_path}")
            continue

        # Load files
        with open(concentration_path, "r") as f:
            frame_data = json.load(f)

        with open(mode_summary_path, "r") as f:
            mode_summary = json.load(f)

        # Map frame_id to behavior/emotion
        frame_mode_map = {}
        for _, data in mode_summary.items():
            behavior = data.get("mode_behavior")
            emotion = data.get("mode_emotion")
            for fid in data.get("frame_ids", []):
                match = re.search(r"frame_(\d+)", fid)
                if match:
                    frame_id = int(match.group(1))
                    frame_mode_map[frame_id] = {
                        "mode_behavior": behavior,
                        "mode_emotion": emotion
                    }

        # Merge
        merged = []
        for entry in frame_data:
            fid = int(entry["frame_id"])
            if fid in frame_mode_map:
                merged_entry = {
                    "frame_id": fid,
                    "face": entry.get("face"),
                    "body": entry.get("body"),
                    **frame_mode_map[fid]
                }
                merged.append(merged_entry)
            else:
                print(f"[WARN] Frame {fid} in {folder} not found in mode summary")

        # Save output
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"[OK] Wrote {len(merged)} entries to {output_path}")

def generate_final_charts(df: pd.DataFrame, root_dir: str):
    os.makedirs(root_dir, exist_ok=True)

    for _, row in df.iterrows():
        student_id = row['student_id']
        pos = row['ratio_positive']
        neu = row['ratio_neutral']
        neg = row['ratio_negative']
        mean = row['mean']
        focus = row['focus_score']
        result = row['result']

        fig, ax = plt.subplots(figsize=(8, 5))

        # Bar chart
        bar_colors = ['green', 'orange', 'red']
        bar_labels = ['Positive', 'Neutral', 'Negative']
        bar_values = [pos, neu, neg]
        bars = ax.bar(bar_labels, bar_values, color=bar_colors)
        ax.set_ylabel('Concentration Ratios')
        ax.set_ylim(0, 1)

        # Thêm phần trăm trên từng cột
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.0%}",
                ha='center', va='bottom', fontsize=10
            )

        # Tiêu đề và chú thích
        ax.set_title(f"{student_id}")
        ax.legend(bars, bar_labels, loc='upper left')

        # Hiển thị mean và focus dưới dạng text box
        textstr = f"Mean: {mean:.2f}\nFocus: {focus:.2f}\n{result}"
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)


        # Lưu ảnh
        output_path = os.path.join(root_dir, student_id, "figure", "statistic_chart.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__ == '__main__':
    df = pd.read_csv('output/backup_demo_fps60/total_result.csv')

    generate_final_charts(df, 'output/backup_demo_fps60/recognition')


