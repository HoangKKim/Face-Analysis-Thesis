# import json 
# import os
# from collections import defaultdict
# import re

# from cfg.evaluate_cfg import *


# def get_minute_from_frame(frame_id, fps=60):
#     return (frame_id // fps) // 60 + 1

# # open json 
# def get_person_json(output_folder):
#     student_json_paths = []
#     for student_folder in os.listdir(output_folder):
#         if student_folder.startswith("unknown_"):
#             continue
#         student_json_path = os.path.join(output_folder, student_folder, 'concentration_mode_summary.json')
#         student_json_paths.append(student_json_path)
#     return student_json_paths

# def extract_frame_id_from_path(path):
#     match = re.search(r"frame_(\d+)\.jpg", path)
#     return int(match.group(1)) if match else -1

# def check_predictions_against_gt(predicted_list, ground_truth_dict, student_id, fps=60):
#     results = []
    
#     if student_id not in ground_truth_dict:
#         raise ValueError(f"Student {student_id} not found in ground truth")
    
#     gt_records = ground_truth_dict[student_id]

#     # Biến đếm để tính precision
#     total_behavior = 0
#     correct_behavior = 0
    
#     total_emotion = 0
#     correct_emotion = 0

#     total_fusion = 0
#     correct_fusion = 0
    
#     for sec, item in predicted_list.items():
#         # print(item)
#         # frame_id = extract_frame_id_from_path(item['body_image'])
#         minute_belonged_to = int(float(sec)) // 60 + 1
        
#         # Lấy record ground truth của phút đó (nếu có)
#         gt_minute_record = next((r for r in gt_records if r['minute'] == minute_belonged_to), None)
        
#         # Lấy predicted behavior và emotion
#         pred_behavior = item.get('mode_behavior', None)
#         pred_emotion = item.get('mode_emotion', None)
        
#         # Kiểm tra tồn tại behavior trong danh sách behavior của ground truth phút đó
#         behavior_check = False
#         emotion_check = False
#         fusion_check = False
        
#         if gt_minute_record:
#             # kiểm tra behavior có trong danh sách behavior không
#             total_behavior += 1
#             total_fusion +=1
#             if pred_behavior in gt_minute_record.get('Behavior', []):
#                 behavior_check = True
#                 correct_behavior += 1
            
#             if pred_emotion != 'Unknown':  # Bỏ qua nếu Unknown
#                 total_emotion += 1
#                 if pred_emotion in gt_minute_record.get('Emotion', []):
#                     emotion_check = True
#                     correct_emotion += 1

#             if pred_behavior in POSITIVE:
#                 fusion_result = 'Positive'
#             elif pred_behavior in NEGATIVE:
#                 fusion_result = 'Negative'
#             else:
#                 if pred_emotion in POSITIVE:
#                     fusion_result = 'Positive'
#                 elif pred_emotion in NEGATIVE:
#                     fusion_result = 'Negative'
#                 else:
#                     fusion_result = 'Neutral'
#             if fusion_result in gt_minute_record.get('Final', []):
#                 fusion_check = True
#                 correct_fusion +=1
        
#         results.append({
#             "second": sec,
#             "behavior": behavior_check,
#             "emotion": emotion_check if pred_emotion != 'Unknown' else None,
#             "fusion": fusion_check
#         })

#     # compute precision
#     behavior_precision = correct_behavior / total_behavior if total_behavior > 0 else 0
#     emotion_precision = correct_emotion / total_emotion if total_emotion > 0 else 0
#     fusion_precision = correct_fusion / total_fusion if total_fusion > 0 else 0
    
#     return results, behavior_precision, emotion_precision, fusion_precision

# def read_json_file(path):
#     with open(path, 'r' , encoding='utf-8') as f:
#         return json.load(f)


# if __name__ ==  "__main__":
#     import os
#     import pandas as pd
    
#     output_dir = "output/backup_demo_fps60/students_result"
#     # read all concentration json file
#     predicted_paths = get_person_json(output_dir)

#     print(predicted_paths)
#     # read ground truth
    
#     ground_truth_data = read_json_file('ground_truth.json')

#     os.makedirs('output/backup_demo_fps60/ground_truth/per_student', exist_ok=True)
#     precision_summary = []

#     for path in predicted_paths:
#         predicted_data = read_json_file(path)
#         student_name = os.path.basename(os.path.dirname(path))

#         result, behavior_precision, emotion_precision, fusion_precision = check_predictions_against_gt(
#             predicted_data, ground_truth_data, student_name
#         )

#         # Lưu chi tiết per student
#         result_df = pd.DataFrame(result)  # result: list of dicts per frame
#         result_df.to_csv(f"output/backup_demo_fps60/ground_truth/per_student/{student_name}.csv", index=False)

#         # Lưu tổng kết precision
#         precision_summary.append({
#             "student": student_name,
#             "precision_behavior": behavior_precision,
#             "precision_emotion": emotion_precision,
#             "precision_fusion": fusion_precision
#         })

#     # Lưu precision_summary vào file CSV tổng
#     precision_df = pd.DataFrame(precision_summary)
#     precision_df.to_csv("output/backup_demo_fps60/ground_truth/precision_summary.csv", index=False)

#     # Tính trung bình precision cho từng loại
#     mean_behavior = precision_df["precision_behavior"].mean()
#     mean_emotion = precision_df["precision_emotion"].mean()
#     mean_fusion = precision_df["precision_fusion"].mean()

#     # In ra kết quả
#     print(f"Mean of precision for behavior: {mean_behavior:.2f}")
#     print(f"Mean of precision for emotion: {mean_emotion:.2f}")
#     print(f"Mean of precision for concentration: {mean_fusion:.2f}")


import json
from sklearn.metrics import precision_score, recall_score, f1_score
import os


def evaluate_result_from_lists(y_true_behavior, y_pred_behavior, y_true_emotion, y_pred_emotion):
    # Accuracy
    behavior_accuracy = sum(p == t for p, t in zip(y_pred_behavior, y_true_behavior)) / len(y_true_behavior)
    emotion_accuracy = sum(p == t for p, t in zip(y_pred_emotion, y_true_emotion)) / len(y_true_emotion)

    # Precision, Recall, F1 (macro cho nhiều class)
    behavior_precision = precision_score(y_true_behavior, y_pred_behavior, average='macro', zero_division=0)
    behavior_recall = recall_score(y_true_behavior, y_pred_behavior, average='macro', zero_division=0)
    behavior_f1 = f1_score(y_true_behavior, y_pred_behavior, average='macro', zero_division=0)

    emotion_precision = precision_score(y_true_emotion, y_pred_emotion, average='macro', zero_division=0)
    emotion_recall = recall_score(y_true_emotion, y_pred_emotion, average='macro', zero_division=0)
    emotion_f1 = f1_score(y_true_emotion, y_pred_emotion, average='macro', zero_division=0)

    return {
        "behavior": {
            "accuracy": behavior_accuracy,
            "precision": behavior_precision,
            "recall": behavior_recall,
            "f1": behavior_f1
        },
        "emotion": {
            "accuracy": emotion_accuracy,
            "precision": emotion_precision,
            "recall": emotion_recall,
            "f1": emotion_f1
        }
    }

def load_data(predicted_path, groundtruth_path):
    with open(predicted_path, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)

    with open(groundtruth_path, "r", encoding="utf-8") as f:
        groundtruth_data = json.load(f)

    if len(predicted_data) != len(groundtruth_data):
        raise ValueError(f"No matching length in {predicted_path} and {groundtruth_path}")

    y_pred_behavior = [pre['predicted_behaviour'] for pre in predicted_data]
    y_true_behavior = [gt['body_label'] for gt in groundtruth_data]

    y_pred_emotion = [pre['predicted_emotion'] for pre in predicted_data]
    y_true_emotion = [gt['face_label'] for gt in groundtruth_data]

    return y_true_behavior, y_pred_behavior, y_true_emotion, y_pred_emotion

if __name__ == '__main__':
    root_dir = 'output/backup_demo_fps60/students_result'

    all_y_true_behavior = []
    all_y_pred_behavior = []
    all_y_true_emotion = []
    all_y_pred_emotion = []

    for student in os.listdir(root_dir):
        student_path = os.path.join(root_dir, student)
        if not os.path.isdir(student_path):
            continue

        predicted_path = os.path.join(student_path, 'concentration.json')
        groundtruth_path = os.path.join('groundtruth_manual', f'{student}.json')

        y_true_b, y_pred_b, y_true_e, y_pred_e = load_data(predicted_path, groundtruth_path)

        all_y_true_behavior.extend(y_true_b)
        all_y_pred_behavior.extend(y_pred_b)
        all_y_true_emotion.extend(y_true_e)
        all_y_pred_emotion.extend(y_pred_e)

    # Tính metric trên toàn bộ dữ liệu gộp lại
    results = evaluate_result_from_lists(all_y_true_behavior, all_y_pred_behavior,
                                        all_y_true_emotion, all_y_pred_emotion)

    print("Overall results on all students combined:")
    print(results)

def evaluate_result_acc(predicted_path, groundtruth_path):
    with open(predicted_path, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)

    with open(groundtruth_path, "r", encoding="utf-8") as f:
        groundtruth_data = json.load(f)

    if len(predicted_data) != len(groundtruth_data):
        raise ValueError("No matching length")
    
    behavior_correct = 0
    emotion_correct = 0
    total = len(predicted_data)

    for pre, gt in zip(predicted_data, groundtruth_data):
        if pre['predicted_behaviour'] == gt['body_label']:
            behavior_correct +=1
        if pre['predicted_emotion'] == gt['face_label']:
            emotion_correct +=1

    behavior_accuracy = behavior_correct / total
    emotion_accuracy = emotion_correct / total

    return behavior_accuracy, emotion_accuracy

def compare_mean_fusion(predicted_path, label_path):
    # Đọc file JSON
    with open(predicted_path, "r", encoding="utf-8") as f:
        predicted = json.load(f)
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    total = 0
    correct = 0

    for (key_pred, val_pred), (key_label, val_label) in zip(predicted.items(), label.items()):
        total += 1
        if val_pred["mean_behavior"] == val_label["mean_behavior"]:
            correct += 1

    return correct / total if total else 0


import os

# if __name__ == '__main__':
#     root_dir = 'output/backup_demo_fps60/students_result'
#     avg = 0 
#     # avg_behavior = 0
#     for file in os.listdir(root_dir):
#         student = os.path.splitext(file)[0]
#         acc = compare_mean_fusion(f'output/backup_demo_fps60/students_result/{student}/concentration_frames_per_second.json', 
#                                             f'groundtruth_manual/mode/{student}_frames_per_second.json')
#         avg += acc
#         # avg_behavior += behavior
#         print(f'{student}: + Acc: {acc:.2f}')
    
#     print(f'Avg: {avg/8}')
#     # print(f'Avg_Behavior: {avg_behavior/8}')


