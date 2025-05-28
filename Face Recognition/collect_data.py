import cv2
from detection.src.detector import Detector


# CNN_FACE_DETECTOR = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# detector = Detector(img_path = None)
# detect objects in image
# detected_result = detector.detect_objects(cv2.imread(img_path))
# print(detected_result)

cap = cv2.VideoCapture('./test.mkv')
frame_id = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imwrite(f'./database_3/frame_{frame_id+1}.jpg', frame)
    frame_id+=1
    
    cv2.imshow("camera", frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_id >= 100:
        break


cap.release()
cv2.destroyAllWindows()


