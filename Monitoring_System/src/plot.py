# import matplotlib.pyplot as plt
# import numpy as np

# def plot_student_report(report_path):
#     """
#     data: dict hoặc pandas row chứa các trường: pos, neu, neg, mean, normalized_score
#     student_id: tên/số ID sinh viên
#     """


#     categories = ['Positive', 'Neutral', 'Negative']
#     values = [data['pos'], data['neu'], data['neg']]
    
#     # Chuyển về % để vẽ bar
#     values_percent = [v * 100 for v in values]

#     fig, ax1 = plt.subplots(figsize=(8,5))
    
#     # Bar plot cho pos, neu, neg
#     bars = ax1.bar(categories, values_percent, color=['#4caf50', '#2196f3', '#f44336'], alpha=0.7)
#     ax1.set_ylabel('Percentage (%)')
#     ax1.set_ylim(0, 100)
#     ax1.set_title(f'Behavior and Emotion Summary for {student_id}')
    
#     # Thêm text % trên mỗi bar
#     for bar in bars:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}%', ha='center', va='bottom')
    
#     # Tạo axis thứ 2 để vẽ line plot mean hoặc normalized_score
#     ax2 = ax1.twinx()
#     ax2.plot(categories, [data['mean']] * 3, color='black', linestyle='--', label='Mean score')
#     ax2.plot(categories, [data['normalized_score']] * 3, color='orange', linestyle='-', label='Normalized score')
#     ax2.set_ylabel('Score')
#     ax2.set_ylim(0, 1)
    
#     # Thêm legend
#     ax2.legend(loc='upper right')
    
#     # Thêm kết luận ở trên cùng
#     conclusion = f"Focus score: {data['normalized_score']:.2f} - "
#     conclusion += "Good" if data['normalized_score'] > 0.7 else "Needs improvement"
#     plt.text(0.5, 105, conclusion, ha='center', va='bottom', fontsize=12, transform=ax1.get_xaxis_transform())
    
#     plt.tight_layout()
#     plt.show()

# # Ví dụ dùng hàm với dữ liệu giả
# example_data = {
#     'pos': 0.55,
#     'neu': 0.30,
#     'neg': 0.15,
#     'mean': 0.6,
#     # 'normalized_score': 0.65
# }

# plot_student_report(example_data, 'Student_01')


import pandas as pd
import matplotlib.pyplot as plt
import os

