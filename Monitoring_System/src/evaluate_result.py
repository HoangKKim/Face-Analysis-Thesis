import numpy as np
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from collections import defaultdict, Counter

from cfg.evaluate_cfg import *
from utils.logger import *
class FocusEvaluator:
    def __init__(self, logger=None):
        # self.output_dir = os.path.join(output_dir, "figure")

        # create dir for figures
        # os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logger or setup_logger(log_path=os.path.join(output_dir, 'focus_eval.log'))
        self.fusion_result = None
        self.behavior_result = None
        self.emotion_result = None

    def load_report(self, report_path):
        try:
            with open(report_path, 'r') as f:
                self.student_data = json.load(f)
            self.logger.info(f"Loaded report: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to load report: {e}")
            self.student_data = []

    def classify(self):
        fusion_result = []
        emotion_result = []
        behavior_result = []

        for element in self.student_data:
            behaviour = element['predicted_behaviour']
            emotion = element['predicted_emotion']

            # behavior
            if behaviour in POSITIVE:
                behavior_result.append(1)
            elif behaviour in NEGATIVE: 
                behavior_result.append(-1)
            else:
                behavior_result.append(0)

            # emotion
            if emotion in POSITIVE:
                emotion_result.append(1)
            elif emotion in NEGATIVE: 
                emotion_result.append(-1)
            else:
                emotion_result.append(0)

            # fusion 
            if behaviour in POSITIVE:
                fusion_result.append(1)
            elif behaviour in NEGATIVE:
                fusion_result.append(-1)
            else:
                if emotion in POSITIVE:
                    fusion_result.append(1)
                elif emotion in NEGATIVE:
                    fusion_result.append(-1)
                else:
                    fusion_result.append(0)

        self.fusion_result = np.array(fusion_result)
        self.behavior_result = np.array(behavior_result)
        self.emotion_result = np.array(emotion_result)

        self.logger.info("Extract info completed.")
        # return self.fusion_result, self.behavior_result, self.emotion_result
    
    def map_mean_to_label(self, mean_val):
        if mean_val > 0.3:
            return 1
        elif mean_val < -0.3:
            return -1
        else:
            return 0

    def aggregate_per_second_and_save(self, report_path, fps):
        grouped = defaultdict(list)

        for i, element in enumerate(self.student_data):
            base_name = os.path.splitext(os.path.basename(element['body_image']))[0]
            frame_id_str = base_name.split('_')[-1]
            try:
                frame_id = int(frame_id_str)
            except ValueError:
                self.logger.warning(f"Cannot parse frame_id from {base_name}")
                continue

            second = frame_id // fps
            grouped[second].append(i)

        result_per_second = {}
        mode_summary = {}

        for sec, indices in sorted(grouped.items()):
            fusion_vals = self.fusion_result[indices].tolist()
            behavior_vals = self.behavior_result[indices].tolist()
            emotion_vals = self.emotion_result[indices].tolist()

            mean_fusion = self.map_mean_to_label(np.mean(fusion_vals))
            mean_behavior = self.map_mean_to_label(np.mean(behavior_vals))
            mean_emotion = self.map_mean_to_label(np.mean(emotion_vals))

            predicted_behaviours = [self.student_data[idx]["predicted_behaviour"] for idx in indices]
            predicted_emotions = [self.student_data[idx]["predicted_emotion"] for idx in indices]

            most_common_behaviour = Counter(predicted_behaviours).most_common(1)[0][0] if predicted_behaviours else None
            most_common_emotion = Counter(predicted_emotions).most_common(1)[0][0] if predicted_emotions else None

            # Lưu mode vào dictionary riêng
            mode_summary[sec] = {
                "mode_behavior": most_common_behaviour,
                "mode_emotion": most_common_emotion,
                "frame_ids": [
                    os.path.splitext(os.path.basename(self.student_data[idx]['body_image']))[0]
                    for idx in indices
                ]
            }

            result_per_second[sec] = {
                "fusion_result": fusion_vals,
                "behavior_result": behavior_vals,
                "emotion_result": emotion_vals,
                "num_frames": len(indices),
                "frame_ids": [
                    os.path.splitext(os.path.basename(self.student_data[idx]['body_image']))[0]
                    for idx in indices
                ],
                "mean_fusion": mean_fusion,
                "mean_behavior": mean_behavior,
                "mean_emotion": mean_emotion,
            }

        # Save detailed results
        output_path = report_path.replace(".json", "_frames_per_second.json")
        with open(output_path, "w") as f:
            json.dump(result_per_second, f, indent=4)

        # Save mode behavior/emotion summary
        summary_path = report_path.replace(".json", "_mode_summary.json")
        with open(summary_path, "w") as f:
            json.dump(mode_summary, f, indent=4)

        # self.logger.info(f"Saved detailed frame results per second to {output_path}")
        self.logger.info(f"Saved morde behavio/emotion summary to {summary_path}")
        return result_per_second


    def draw_single_curve(self, factor_data, type_curve, mean_value=None):
        time_steps = list(range(len(factor_data)))

        df = pd.DataFrame({
            'time': time_steps,
            'value': factor_data
        })

        color_map = {
            "emotion": "#1f77b4",
            "behavior": "#2ca02c",
            "fusion": "#d62728",
            "mean": "#ffcc00"
        }
        color = color_map.get(type_curve.lower(), 'black')

        plt.figure(figsize=(20, 6))
        sns.lineplot(data=df, x='time', y='value', marker='o', linewidth=1, color=color, dashes=False)

        if mean_value is not None:
            plt.axhline(mean_value, color=color_map['mean'], linestyle='--', linewidth=1.5, label=f'Mean = {mean_value:.2f}')
            plt.legend()

        plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
        plt.title(f"{type_curve.title()} Curve")
        plt.xlabel("Seconds")
        plt.ylabel("State")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f'{type_curve}_curve.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Saved {type_curve} curve at {save_path}")

    def compute_focus_score(self, data, alpha = 0.0):
        data = data
        total = len(data)
        count_pos = data.count(1)
        count_neu = data.count(0)
        count_neg = data.count(-1)

        ratio_pos = count_pos / total
        ratio_neu = count_neu / total
        ratio_neg = count_neg / total

        mu = np.mean(data)
        sigma = np.std(data, ddof=0)
        score = mu - alpha * sigma
        # normalized_score = (score + 1) / 2

        self.logger.info(f"Ratio of positive: {(ratio_pos * 100):.2f}%")
        self.logger.info(f"Ratio of neutral : {(ratio_neu * 100):.2f}%")
        self.logger.info(f"Ratio of negative: {(ratio_neg * 100):.2f}%")

        self.logger.info(f"Mean: {mu:.3f}")
        self.logger.info(f"Std: {sigma:.3f}")
        self.logger.info(f"Focus score: {score:.3f}")
        # self.logger.info(f"Normalized score: {normalized_score:.2f}")

        return {
            "ratio_positive": round(ratio_pos, 2),
            "ratio_neutral": round(ratio_neu, 2),
            "ratio_negative": round(ratio_neg, 2),
            "mean": round(mu, 2),
            "std": round(sigma, 2),
            "focus_score": round(score, 2),
        }
    
    def update_output_dir(self, output_dir):
        self.output_dir = os.path.join(output_dir, 'figure')
        os.makedirs(self.output_dir, exist_ok = True)


# class FocusEvaluator:
#     def __init__(self, logger=None):
#         # self.output_dir = os.path.join(output_dir, "figure")

#         # create dir for figures
#         # os.makedirs(self.output_dir, exist_ok=True)

#         self.logger = logger or setup_logger(log_path=os.path.join(output_dir, 'focus_eval.log'))
#         self.fusion_result = None
#         self.behavior_result = None
#         self.emotion_result = None

#     def load_report(self, report_path):
#         try:
#             with open(report_path, 'r') as f:
#                 self.student_data = json.load(f)
#             self.logger.info(f"Loaded report: {report_path}")
#         except Exception as e:
#             self.logger.error(f"Failed to load report: {e}")
#             self.student_data = []

#     def classify(self):
#         fusion_result = []
#         emotion_result = []
#         behavior_result = []

#         for element in self.student_data:
#             behaviour = element['body_label']
#             emotion = element['face_label']

#             # behavior
#             if behaviour in POSITIVE:
#                 behavior_result.append(1)
#             elif behaviour in NEGATIVE: 
#                 behavior_result.append(-1)
#             else:
#                 behavior_result.append(0)

#             # emotion
#             if emotion in POSITIVE:
#                 emotion_result.append(1)
#             elif emotion in NEGATIVE: 
#                 emotion_result.append(-1)
#             else:
#                 emotion_result.append(0)

#             # fusion 
#             if behaviour in POSITIVE:
#                 fusion_result.append(1)
#             elif behaviour in NEGATIVE:
#                 fusion_result.append(-1)
#             else:
#                 if emotion in POSITIVE:
#                     fusion_result.append(1)
#                 elif emotion in NEGATIVE:
#                     fusion_result.append(-1)
#                 else:
#                     fusion_result.append(0)

#         self.fusion_result = np.array(fusion_result)
#         self.behavior_result = np.array(behavior_result)
#         self.emotion_result = np.array(emotion_result)

#         self.logger.info("Extract info completed.")
#         # return self.fusion_result, self.behavior_result, self.emotion_result
    
#     def map_mean_to_label(self, mean_val):
#         if mean_val > 0.3:
#             return 1
#         elif mean_val < -0.3:
#             return -1
#         else:
#             return 0

#     def aggregate_per_second_and_save(self, report_path, fps):
#         grouped = defaultdict(list)

#         for i, element in enumerate(self.student_data):
#             base_name = os.path.splitext(os.path.basename(element['body_image']))[0]
#             frame_id_str = base_name.split('_')[-1]
#             try:
#                 frame_id = int(frame_id_str)
#             except ValueError:
#                 self.logger.warning(f"Cannot parse frame_id from {base_name}")
#                 continue

#             second = frame_id // fps
#             grouped[second].append(i)

#         result_per_second = {}
#         mode_summary = {}

#         for sec, indices in sorted(grouped.items()):
#             fusion_vals = self.fusion_result[indices].tolist()
#             behavior_vals = self.behavior_result[indices].tolist()
#             emotion_vals = self.emotion_result[indices].tolist()

#             mean_fusion = self.map_mean_to_label(np.mean(fusion_vals))
#             mean_behavior = self.map_mean_to_label(np.mean(behavior_vals))
#             mean_emotion = self.map_mean_to_label(np.mean(emotion_vals))

#             predicted_behaviours = [self.student_data[idx]["body_label"] for idx in indices]
#             predicted_emotions = [self.student_data[idx]["face_label"] for idx in indices]

#             most_common_behaviour = Counter(predicted_behaviours).most_common(1)[0][0] if predicted_behaviours else None
#             most_common_emotion = Counter(predicted_emotions).most_common(1)[0][0] if predicted_emotions else None

#             # Lưu mode vào dictionary riêng
#             mode_summary[sec] = {
#                 "mode_behavior": most_common_behaviour,
#                 "mode_emotion": most_common_emotion,
#                 "frame_ids": [
#                     os.path.splitext(os.path.basename(self.student_data[idx]['body_image']))[0]
#                     for idx in indices
#                 ]
#             }

#             result_per_second[sec] = {
#                 "fusion_result": fusion_vals,
#                 "behavior_result": behavior_vals,
#                 "emotion_result": emotion_vals,
#                 "num_frames": len(indices),
#                 "frame_ids": [
#                     os.path.splitext(os.path.basename(self.student_data[idx]['body_image']))[0]
#                     for idx in indices
#                 ],
#                 "mean_fusion": mean_fusion,
#                 "mean_behavior": mean_behavior,
#                 "mean_emotion": mean_emotion,
#             }

#         # Save detailed results
#         output_path = report_path.replace(".json", "_frames_per_second.json")
#         with open(output_path, "w") as f:
#             json.dump(result_per_second, f, indent=4)

#         # Save mode behavior/emotion summary
#         summary_path = report_path.replace(".json", "_mode_summary.json")
#         with open(summary_path, "w") as f:
#             json.dump(mode_summary, f, indent=4)

#         # self.logger.info(f"Saved detailed frame results per second to {output_path}")
#         self.logger.info(f"Saved morde behavio/emotion summary to {summary_path}")
#         return result_per_second


#     def draw_single_curve(self, factor_data, type_curve, mean_value=None):
#         time_steps = list(range(len(factor_data)))

#         df = pd.DataFrame({
#             'time': time_steps,
#             'value': factor_data
#         })

#         color_map = {
#             "emotion": "#1f77b4",
#             "behavior": "#2ca02c",
#             "fusion": "#d62728",
#             "mean": "#ffcc00"
#         }
#         color = color_map.get(type_curve.lower(), 'black')

#         plt.figure(figsize=(20, 6))
#         sns.lineplot(data=df, x='time', y='value', marker='o', linewidth=1, color=color, dashes=False)

#         if mean_value is not None:
#             plt.axhline(mean_value, color=color_map['mean'], linestyle='--', linewidth=1.5, label=f'Mean = {mean_value:.2f}')
#             plt.legend()

#         plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
#         plt.title(f"{type_curve.title()} Curve")
#         plt.xlabel("Seconds")
#         plt.ylabel("State")
#         plt.grid(True)
#         plt.tight_layout()

#         save_path = os.path.join(self.output_dir, f'{type_curve}_curve.png')
#         plt.savefig(save_path)
#         plt.close()
#         self.logger.info(f"Saved {type_curve} curve at {save_path}")

#     def compute_focus_score(self, data, alpha = 0.0):
#         data = data
#         total = len(data)
#         count_pos = data.count(1)
#         count_neu = data.count(0)
#         count_neg = data.count(-1)

#         ratio_pos = count_pos / total
#         ratio_neu = count_neu / total
#         ratio_neg = count_neg / total

#         mu = np.mean(data)
#         sigma = np.std(data, ddof=0)
#         score = mu - alpha * sigma
#         # normalized_score = (score + 1) / 2

#         self.logger.info(f"Ratio of positive: {(ratio_pos * 100):.2f}%")
#         self.logger.info(f"Ratio of neutral : {(ratio_neu * 100):.2f}%")
#         self.logger.info(f"Ratio of negative: {(ratio_neg * 100):.2f}%")

#         self.logger.info(f"Mean: {mu:.3f}")
#         self.logger.info(f"Std: {sigma:.3f}")
#         self.logger.info(f"Focus score: {score:.3f}")
#         # self.logger.info(f"Normalized score: {normalized_score:.2f}")

#         return {
#             "ratio_positive": round(ratio_pos, 2),
#             "ratio_neutral": round(ratio_neu, 2),
#             "ratio_negative": round(ratio_neg, 2),
#             "mean": round(mu, 2),
#             "std": round(sigma, 2),
#             "focus_score": round(score, 2),
#         }
    
#     def update_output_dir(self, output_dir):
#         self.output_dir = os.path.join(output_dir, 'figure')
#         os.makedirs(self.output_dir, exist_ok = True)

if __name__ == '__main__':
    input_dir = 'groundtruth_manual'

    output_dir = 'groundtruth_manual'

    # os.makedirs(output_dir, exist_ok=True)
    # logger = setup_logger(log_path="process_log.txt")

    evaluator = FocusEvaluator()
    evaluator.update_output_dir(input_dir)
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)

        if file not in ['Student_01.json', 'Student_02.json', 'Student_03.json', 'Student_04.json', 'Student_05.json', 'Student_06.json', 'Student_07.json', 'Student_08.json']:
            continue

        evaluator.load_report(file_path)
        print(file_path)
        evaluator.classify()

        aggregated_result = evaluator.aggregate_per_second_and_save(file_path, 60)

        mean_fusion = [sec['mean_fusion'] for sec in aggregated_result.values()]
        # mean_behavior = [sec['mean_behavior'] for sec in aggregated_result.values()]
        # mean_emotion = [sec['mean_emotion'] for sec in aggregated_result.values()]


        focus_score_result = evaluator.compute_focus_score(mean_fusion)
        score = focus_score_result['focus_score']
        print(f'{file}: {score}')
        

        




        

