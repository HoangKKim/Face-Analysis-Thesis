import numpy as np
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from cfg.evaluate_cfg import *
from utils.logger import *


class FocusEvaluator:
    def __init__(self, output_dir, logger=None):
        self.output_dir = os.path.join(output_dir, "figure")
        self.logger = logger or setup_logger(log_path=os.path.join(output_dir, 'focus_eval.log'))

        # create dir for figures
        os.makedirs(self.output_dir, exist_ok=True)

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
        return self.fusion_result, self.behavior_result, self.emotion_result

    def draw_single_curve(self, factor_data, type_curve):
        time_steps = list(range(len(factor_data)))

        df = pd.DataFrame({
            'time': time_steps,
            'value': factor_data
        })

        color_map = {
            "emotion": "#1f77b4",
            "behavior": "#2ca02c",
            "fusion": "#d62728",
        }
        color = color_map.get(type_curve.lower(), 'black')

        plt.figure(figsize=(20, 6))
        sns.lineplot(data=df, x='time', y='value', marker='x', linewidth=1, color=color)

        plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
        plt.title(f"{type_curve.title()} Curve")
        plt.xlabel("Time (frames)")
        plt.ylabel("State")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f'{type_curve}_curve.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Saved {type_curve} curve at {save_path}")

    def draw_three_curves(self):
        n = len(self.fusion_result)
        frames = np.arange(n)

        df = pd.DataFrame({
            "Frame": np.tile(frames, 3),
            "Value": np.concatenate([self.emotion_result, self.behavior_result, self.fusion_result]),
            "Signal": np.repeat(["Emotion", "Behavior", "Fusion"], n),
        })

        palette = {
            "Emotion": "#1f77b4",
            "Behavior": "#2ca02c",
            "Fusion": "#d62728",
        }

        dashes = {
            "Emotion": (2, 2),
            "Behavior": (4, 1),
            "Fusion": "",
        }

        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.figure(figsize=(20, 6))

        sns.lineplot(
            data=df,
            x="Frame",
            y="Value",
            hue="Signal",
            palette=palette,
            linewidth=1.0,
            style="Signal",
            dashes=dashes
        )

        plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        plt.ylim([-1.2, 1.2])
        plt.xlabel("Time (frames)")
        plt.ylabel("State")
        plt.title("Concentration Curve", fontsize=15)
        plt.legend(title="Signal Type", loc="upper right")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'concentration_curve.png')
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Saved 3-curve figure to {save_path}")

    def compute_focus_score(self, alpha = 0.3):
        data = self.fusion_result
        ratio_pos = np.mean(data == 1)
        ratio_neu = np.mean(data == 0)
        ratio_neg = 1 - ratio_pos - ratio_neu

        mu = np.mean(data)
        sigma = np.std(data, ddof=0)
        score = mu - alpha * sigma
        normalized_score = (score + 1) / 2

        self.logger.info(f"Ratio of positive: {(ratio_pos * 100):.2f}%")
        self.logger.info(f"Ratio of neutral : {(ratio_neu * 100):.2f}%")
        self.logger.info(f"Ratio of negative: {(ratio_neg * 100):.2f}%")

        self.logger.info(f"Mean: {mu:.3f}")
        self.logger.info(f"Std: {sigma:.3f}")
        self.logger.info(f"Focus score: {score:.3f}")
        self.logger.info(f"Normalized score: {normalized_score:.2f}")

        return {
            "ratio_positive": round(ratio_pos, 2),
            "ratio_neutral": round(ratio_neu, 2),
            "ratio_negative": round(ratio_neg, 2),
            "mean": round(mu, 3),
            "std": round(sigma, 3),
            "focus_score": round(score, 3),
            "normalized_focus_score": round(normalized_score, 3)}
    
    def update_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok = True)

    def get_output_dir(self):
        return self.output_dir

if __name__ == '__main__':
    # # file path
    input_dir = 'output/output_pose/'
    output_dir = 'output/figure'
    # root_dir = 'output/figure/Ngoc'
    # os.makedirs(root_dir, exist_ok = True)

    # # get factors result
    # all, behavior, emotion = result_evalutator_by_report(file_path)

    # # draw_part_factor_curve(emotion, 'emotion', root_dir)
    # # draw_part_factor_curve(behavior, 'behavior', root_dir)
    # # draw_part_factor_curve(all, 'total', root_dir)

    # draw_three_curves(emotion, behavior, all, root_dir)
    # final_result = compute_focus_score(all, 0.5)

    # # print(f"Concentration score: {final_result:.2f}")

    EVALUATOR = FocusEvaluator(output_dir)
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0])

        # get inf
        EVALUATOR.update_output_dir(output_path)
        EVALUATOR.load_report(file_path)
        fusion, behavior, emotion = EVALUATOR.classify()

        # draw each figure
        EVALUATOR.draw_single_curve(emotion, 'emotion')
        EVALUATOR.draw_single_curve(behavior, 'behavior')
        EVALUATOR.draw_single_curve(fusion, 'fusion')

        # draw all in one
        EVALUATOR.draw_three_curves()
        



        

