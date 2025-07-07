import cv2
import os
cap = cv2.VideoCapture('./input/recognition/2025-06-22 16-17-00.mkv')
os.makedirs('database/image/Minh', exist_ok=True)


count_frame = 0
while count_frame < 600:
    ret, frame = cap.read()

    if ret is None: 
        break
    
    cv2.imwrite(f"database/image/Minh/frame_{count_frame+1}.jpg", frame)
    count_frame +=1
cap.release()

    
    