# import numpy as np

# # List tọa độ keypoint theo COCO (17 điểm): shape (17, 2)
# keypoints = np.array([
#     [365.83,  83.33], [374.16,  75.00], [357.5 ,  75.00],
#     [390.83,  75.00], [357.5 ,  83.33], [407.5 , 108.33],
#     [365.83, 116.66], [440.83, 150.00], [349.16, 158.33],
#     [449.16, 166.66], [307.5 , 175.00], [440.83, 208.33],
#     [399.16, 216.66], [432.5 , 283.33], [374.16, 275.00],
#     [474.16, 366.66], [407.5 , 341.66]
# ], dtype=np.float32)

# # Độ tin cậy của từng keypoint (nếu có)
# scores = np.array([
#     0.93, 0.93, 0.90, 0.95, 0.91, 0.92, 0.86,
#     0.85, 0.91, 0.93, 0.95, 0.88, 0.89, 0.92,
#     0.88, 0.85, 0.90
# ], dtype=np.float32)

# import cv2
# import matplotlib.pyplot as plt

# # Load ảnh RGB
# img = cv2.imread('demo.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Skeleton connections theo chuẩn COCO (có thể custom nếu muốn)
# skeleton = [
#     (0, 1), (0, 2), (1, 3), (2, 4),
#     (3, 5), (4, 6), (5, 7), (7, 9),
#     (6, 8), (8, 10), (5, 6),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15),
#     (12, 14), (14, 16)
# ]

# # Threshold để chỉ vẽ điểm tin cậy
# threshold = 0.3

# # Vẽ keypoint
# for i, (x, y) in enumerate(keypoints):
#     if scores[i] > threshold:
#         cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # màu xanh

# # Vẽ nối các keypoint theo skeleton
# for (i, j) in skeleton:
#     if scores[i] > threshold and scores[j] > threshold:
#         pt1 = tuple(keypoints[i].astype(int))
#         pt2 = tuple(keypoints[j].astype(int))
#         cv2.line(img, pt1, pt2, (255, 0, 0), 1)  # màu xanh dương

# # Hiển thị ảnh kết quả
# plt.figure(figsize=(8, 8))
# plt.imshow(img)
# plt.axis('off')
# plt.title("Keypoints + Skeleton (COCO Format)")
# plt.show()


from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from cfg.pose_cfg import *

register_all_modules()

# config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
# checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(MMPOSE_MODEL_CFG, MMPOSE_CKPT_PATH, device='cuda:0')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'modules/pose_estimation/demo_1.jpg')
# 
keypoints = results[0].pred_instances.keypoints

keypoints = keypoints[0]

# print(keypoints[0])

import cv2
import matplotlib.pyplot as plt

# Load ảnh RGB
img = cv2.imread('modules/pose_estimation/demo_1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Skeleton connections theo chuẩn COCO (có thể custom nếu muốn)
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Vẽ keypoint
for (x, y) in (keypoints):
    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # màu xanh

# Vẽ nối các keypoint theo skeleton
for (i, j) in skeleton:
    pt1 = tuple(keypoints[i].astype(int))
    pt2 = tuple(keypoints[j].astype(int))
    cv2.line(img, pt1, pt2, (255, 0, 0), 1)  # màu xanh dương

cv2.imwrite('modules/pose_estimation/Keypoints_plot.jpg', img)


