import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.detector.demos.image import detector
import numpy as np

video_path = './input/input_video/215475_class.mp4'
conf_threshold = 0.5
tracker = DeepSort(max_age = 30, n_init = 3)
class_names = ['face', 'body']
class_id = [0, 1]

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

cap = cv2.VideoCapture(video_path)

if(not cap.isOpened()):
    print("Error: Could not open video.")
    exit()

# Lấy thông tin video
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output/output_video/output_result.mp4', fourcc, fps, (width, height))

count_frame = 0

track_colors = {}  # dict lưu màu cho từng track_id

while True:
    rec, frame = cap.read()
    if not rec:
        break
    
    results = detector(frame, img_path = count_frame, data = './src/detector/data/JointBP_CrowdHuman_face.yaml', weights = './src/detector/best.pt', 
                imgsz = 1024, device = 0, conf_thres = 0.7, iou_thres = 0.5, match_iou = 0.6, 
                scales = [1], line_thick = 2, counting = 0, num_offsets = 2)
    
    detect = []
    for detected_obj in results:
        (x1, y1, x2, y2) = detected_obj['bb_body']
        (fx1, fy1, fx2, fy2) = detected_obj['bb_face']

        w, h = x2-x1, y2-y1
        fw, fh = fx2-fx1, fy2-fy1

        detect.append([[x1, y1, w, h], detected_obj['body_score'], 1])
        detect.append([[fx1, fy1, fw, fh], detected_obj['face_score'], 0])


    tracks = tracker.update_tracks(detect, frame=frame)

    for track in tracks:
        track_id = track.track_id

        # Nếu track_id chưa có màu thì random 1 màu mới
        if track_id not in track_colors:
            track_colors[track_id] = (np.random.randint(0, 255),
                                      np.random.randint(0, 255),
                                      np.random.randint(0, 255))

        color = track_colors[track_id]

        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        label = f'{class_names[class_id]} - {track_id}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(f'./output/output_tracker/frame_{count_frame}.jpg', frame)
    print(f'Frame processing: {count_frame}')
    count_frame += 1

    # cv2.imshow("OT", frame)
    out.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

    








