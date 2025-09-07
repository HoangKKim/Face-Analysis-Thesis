import cv2
from tqdm import tqdm

def downsample_video(input_path, output_path, target_fps=30):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Không thể mở video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_interval = int(original_fps / target_fps)
    print(original_fps, frame_interval)
    frame_id = 0

    with tqdm(total=total_frames, desc="Đang giảm FPS", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # if frame_id % frame_interval == 0:
            out.write(frame)

            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Video đã được lưu tại: {output_path}")

# detect 
from modules.detection.src.detector import *

def plot_detection_result(image, detections, color=(0, 255, 0)):
    """
    Vẽ bounding boxes từ danh sách detection lên ảnh.
    - image: ảnh gốc (numpy array)
    - detections: output từ detect_person (list các dict chứa 'bb_body' và 'person_id')
    """
    for det in detections:
        x1, y1, x2, y2 = det['bb_body']
        person_id = det['person_id']
        score = det['body_score']

        label = f"ID:{person_id} {score:.2f}"

        # Vẽ bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Vẽ label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return image

def run_video_detection(video_path, detector):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Không mở được video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_person(frame, conf_thres=0.4, iou_thres=0.5)
        frame_vis = plot_detection_result(frame.copy(), detections)

        cv2.imwrite("frame_0.jpg", frame_vis)


        break

    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = YOLO_Detector()
    run_video_detection("input/input_video/backup_demo_fps60.mp4", detector)

    # downsample_video('input/input_video/backup_demo.mov', 'input/input_video/backup_demo_fps60.mp4', 60)    