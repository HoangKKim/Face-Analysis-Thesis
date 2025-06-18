import os
import cv2
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from glob import glob
import os
from pathlib import Path

# giả sử: tốc độ video là 30fps (30 frames per second) -> 
# số frames muốn lấy cho 1 giây: k 
# steps = fps / k

def extract_keyframes(input_video, num_frames_get_per_sec = 5, saving_dir = 'output/output_keyframes/whole_video'):
    # read video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    jumping_step = int(fps / num_frames_get_per_sec)
    index = 0

    # create saving dir 
    Path(saving_dir).mkdir(parents=True, exist_ok=True)

    while True:
        print(index, end='...')
        rec, frame = cap.read()
        if not rec:
            break
        # getting frame
        if (index % jumping_step == 0):
            # save frame in folder
            frame_path = os.path.join(saving_dir, f'frame_{index}.jpg')
            cv2.imwrite(frame_path, frame)
        index +=1
    print(f"\nExtract video from {input_video} successfully")
                
# def calc_histogram(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0, 180, 0, 256])
#     return cv2.normalize(hist, hist).flatten()

# def extract_keyframes(frame_folder, window = 5):
#     # assure the ordered
#     frame_paths = sorted(glob(os.path.join(frame_folder, '*.jpg')))
#     histograms = []
#     for path in frame_paths:
#         img = cv2.imread(path)
#         if img is None:
#             continue
#         hist = calc_histogram(img)
#         histograms.append(hist)
    
#     # compute diff among seq frames
#     diffs = [cv2.compareHist(histograms[i], histograms[i+1], cv2.HISTCMP_BHATTACHARYYA)
#              for i in range(len(histograms) - 1)]
#     diffs = np.array(diffs)

#     # find local minima and maxima
#     min_idx = argrelextrema(diffs, np.less, order=window)[0]
#     max_idx = argrelextrema(diffs, np.greater, order=window)[0]
#     keyframe_indices = sorted(set(min_idx).union(set(max_idx)))

#     keyframe_paths = [frame_paths[i] for i in keyframe_indices]

#     # Hiển thị đồ thị (tuỳ chọn)
#     plt.plot(diffs)
#     plt.scatter(min_idx, diffs[min_idx], c='g', label='Minima')
#     plt.scatter(max_idx, diffs[max_idx], c='r', label='Maxima')
#     plt.legend()
#     plt.title('Histogram Distance Curve')
#     plt.show()

#     return keyframe_paths

# keyframes = extract_keyframes('output/output_tracking/person_01/face', window = 10)
# output_dir = 'output/output_keyframes/person_01/face'
# os.makedirs(output_dir, exist_ok=True)
# print("keyframes selected: ",len(keyframes))
# for i, path in enumerate(keyframes):
#     img = cv2.imread(path)
#     cv2.imwrite(os.path.join(output_dir, f'keyframe_{i:03d}.jpg'), img)

if __name__ == '__main__':
    extract_keyframes('input/input_video/Kim_Oanh_test_video.mp4', 5)