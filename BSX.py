import cv2
import numpy as np

image = cv2.imread('bsx.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 50
license_plate_contours = [c for c in contours if cv2.contourArea(c) > min_area]

cv2.drawContours(image, license_plate_contours, -1, (0, 255, 0), 2)

cv2.imshow('Nhan dang bien so xe', image)
cv2.waitKey(0)
cv2.destroyAllWindows()