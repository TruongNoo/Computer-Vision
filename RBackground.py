import cv2
import numpy as np

cap = cv2.VideoCapture(0)

prev_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = gray
        continue

    frame_diff = cv2.absdiff(prev_frame, gray)

    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('Motion Detection', thresh)

    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()