import cv2
import numpy as np
from matplotlib import pyplot as plt
# Bai tap 1: Can bang anh

img = cv2.imread("C:/Users/thoidaipc/Downloads/apc.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input", img)
cv2.waitKey(33)
rows, cols = img.shape
new_level = 256
equalized_img = cv2.equalizeHist(img)
cv2.imshow("Thu vien", equalized_img)
cv2.waitKey(20)
matrix = np.zeros((3,256))
h_g = matrix[0]
t_g = matrix[1]
f_g = matrix[2]

for i in range(rows):
    for j in range(cols):
        h_g[img[i,j]] += 1

t_g[0] = h_g[0]
for i in range(1,256):
    t_g[i] = t_g[i-1]+h_g[i]

average = rows*cols/new_level

for i in range(256):
    f_g[i] = max(0, round(t_g[i]/average)-1)

for i in range(rows):
    for j in range(cols):
        img[i,j] = f_g[img[i,j]]

cv2.imshow("Output", img)
cv2.waitKey(0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()
