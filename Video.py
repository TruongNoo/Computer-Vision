import cv2
import numpy as np

frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(0)

squares = [
    {'x': frame_width // 4, 'y': 0, 'size': 50, 'color': (0, 255, 0), 'speed': 2, 'moving_up': False},
    {'x': frame_width // 2, 'y': 0, 'size': 50, 'color': (0, 0, 255), 'speed': 3, 'moving_up': False},
    {'x': 3 * frame_width // 4, 'y': 0, 'size': 50, 'color': (255, 0, 0), 'speed': 4, 'moving_up': False}
]

while True:
    ret, frame = cap.read()
    
    for square in squares:
        if square['moving_up']:
            square['y'] -= square['speed']
            if square['y'] <= 0:
                square['moving_up'] = False
        else:
            square['y'] += square['speed']
            if square['y'] >= frame_height-square['size']:
                square['moving_up'] = True
                square['speed'] = np.random.randint(1, 5)

        cv2.rectangle(frame, (square['x'], square['y']),
                      (square['x'] + square['size'], square['y'] + square['size']),
                      square['color'], -1)

    cv2.imshow('Falling Squares', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()