import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
    Xét một cột
    v = 
    Hiển thị Histogram trước và sau
    Cách sinh ảnh Histogram
    Bước 1: Tạo ảnh np.zeros((H,W,3), np.uint8) W = 256, H \in (200-400)
        + W = 256 => 1 cột thể hiện giá trị Histogram
        + Xét một cột: H tương ứng với max{h}
    Bước 2: Tính v = H*h_i/max(h_i)
        + Tô từ (i,H-v) đến (i,H -1)
'''
def create_histogram(his):
    H = 300
    W = 256
    his_img = np.zeros((H,W,3))
    h_max = max(his)
    for i in range(256):
        v = int(H*his[i]/h_max)
        for j in range(H-v,H-1):
            his_img[j,i] = (0,0,256)
    return his_img


img = cv2.imread("BG.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Input", img)
# cv2.waitKey(33)

rows, cols = img.shape
new_level = 256

matrix = np.zeros((3,256))
h_g = matrix[0]
t_g = matrix[1]
f_g = matrix[2]
for i in range(rows):
    for j in range(cols):
        h_g[img[i,j]] += 1

H = 300
W = 256
his = np.zeros((H,W,3))
h_max = max(h_g)
for i in range(256):
    v = int(H*h_g[i]//h_max)
    for j in range(H-v,H-1):
        his[j,i] = (0,0,255)



t_g[0] = h_g[0]
for i in range(1,256):
    t_g[i] = t_g[i-1]+h_g[i]

average = rows*cols/new_level

for i in range(256):
    f_g[i] = max(0, round(t_g[i]/average)-1)

for i in range(rows):
    for j in range(cols):
        img[i,j] = f_g[img[i,j]]

his_1 = create_histogram(h_g)
cv2.imshow("Input", his_1)
cv2.waitKey(33)
for i in range(256):
    h_g[i] = 0
for i in range(rows):
    for j in range(cols):
        h_g[img[i,j]] += 1
his_2 = create_histogram(h_g)
cv2.imshow("Output", his_2)
cv2.waitKey(0)
# cv2.imshow("Output", img)
# cv2.waitKey(0)

histr = cv2.calcHist([img],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()

