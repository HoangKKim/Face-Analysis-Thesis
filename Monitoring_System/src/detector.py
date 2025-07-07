from modules.detection.src.detector import Detector
from cfg.detector_cfg import *

import argparse
import json
import os
import cv2

# for i in range(510, 575):
path = 'output/output_tracking/person_06/body/frame_06300.jpg'
img = cv2.imread(path)
if img is not None:
    print('read image successfully')
    detector = Detector()

    detected_result = detector.detect_objects(img)
    print(f"{detected_result}")
else:
    print('Failed to read img')





# img_path = 'input/input_video/Kim_Oanh_test_video.mp4'

# # init detector
# cap = cv2.VideoCapture(img_path)
# output_video_path = 'output/detect_result.mp4'

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec cho .mp4
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
# total_result = []

# while True:
#     rect, frame = cap.read()

#     if not rect:
#         break

#     result = detector.detect_objects(frame)
#     for obj in result:
#         [x1, y1, x2, y2] = obj['bb_body']
#         bd_score = obj['body_score']
#         face_score = obj['face_score']

#         if face_score != 0:
#             [fx1, fy1, fx2, fy2] = obj['bb_face']
#             # draw bbox
#             cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0,0,255), 1)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,00), 1)
#     out.write(frame)
#     total_result.append(result)

# cap.release()
# out.release()

# with open('output/detect.json', 'w') as f:
#     json.dump(total_result, f, indent=4)

        
        
# write in json file
# with open(f'output/{os.path.splitext(os.path.basename(args.img_path))[0]}_output.json', 'w') as f:
#     json.dump(detected_objs, f, indent = 4)
# print(f"Detected done - The result is written in output/{os.path.splitext(os.path.basename(args.img_path))[0]}_output.json")

    

