import numpy as np
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from cfg.evaluate_cfg import *

# step1 : Nhận vào 1 file json

def result_evalutator_by_report(student_report):
    # load json file
    try:
        with open(student_report, 'r') as f:
            student_data = json.load(f)
    except Exception as e:
        print(f"Failed to load report {e}")

    moment_result = []
    emotion_result = []
    behavior_result = []

    # process 
    for element in student_data:
        behaviour =  element['predicted_behaviour']
        emotion = element['predicted_emotion']

        # for behavior
        if behaviour in POSITIVE:
            behavior_result.append(1)
        elif behaviour in NEGATIVE:
            behavior_result.append(-1)
        else:
            behavior_result.append(0)

        # for emotion
        if emotion in POSITIVE:
            emotion_result.append(1)
        elif emotion in NEGATIVE:
            emotion_result.append(-1)
        else:
            emotion_result.append(0)

        # final moment
        if behaviour in POSITIVE:
            moment_result.append(1)
        elif behaviour in NEGATIVE:
            moment_result.append(-1)
        else:
            if emotion in POSITIVE:
                moment_result.append(1)
            elif emotion in NEGATIVE:
                moment_result.append(-1)
            else:
                moment_result.append(0)

    return moment_result, behavior_result, emotion_result

def draw_part_factor_curve(factor_data, figure_name, root_dir = 'output/figure'):
    time_steps = list(range(len(factor_data)))

    # create dataframe
    df = pd.DataFrame({
        'time': time_steps,
        'value': factor_data})
    
    # draw line chart
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x= 'time', y = 'value', markers='x', linewidth = 1)

    plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
    plt.title(f'{figure_name} Curve Over Time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, figure_name))
    plt.show() 


if __name__ == '__main__':
    # file path
    file_path = 'output/output_pose/Ngoc.json'
    root_dir = 'output/figure/Ngoc'
    os.makedirs(root_dir, exist_ok = True)

    # get factors result
    all, behavior, emotion = result_evalutator_by_report(file_path)

    draw_part_factor_curve(emotion, 'emotion', root_dir)
    draw_part_factor_curve(behavior, 'behavior', root_dir)
    draw_part_factor_curve(all, 'total', root_dir)
