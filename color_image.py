import cv2
import numpy as np

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
def create_histogram(his, channel):
    H = 300
    W = 256
    his_img = np.zeros((H,W,3))
    h_max = max(his)
    for i in range(256):
        v = int(H*his[i]/h_max)
        for j in range(H-v,H-1):
            if channel == 0:
                his_img[j,i] = (255,0,0)
            elif channel == 1:
                his_img[j,i] = (0,255,0)
            else:
                his_img[j,i] = (0,0,255)
    return his_img


img = cv2.imread("Trungthu.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("Input", img)
# cv2.waitKey(0)
# blue_channel = [i[3][0] for i in img]
# green_channel = [i[3][1] for i in img]
# red_channel = [i[3][2] for i in img]
# blue_his = create_histogram(blue_channel, 0)
# cv2.imshow("Blue", blue_his)
# cv2.waitKey(50)
# green_his = create_histogram(green_channel, 1)
# cv2.imshow("Green", green_his)
# cv2.waitKey(33)
# red_his = create_histogram(red_channel, 2)
# cv2.imshow("Red", red_his)
# cv2.waitKey(0)
cv2.imshow("Input", img)
cv2.waitKey(33)
rows, cols, height = img.shape
new_level = 256
for channel in range(2):
    matrix = np.zeros((3,256))
    h_g = matrix[0]
    t_g = matrix[1]
    f_g = matrix[2]

    for i in range(rows):
        for j in range(cols):
            h_g[img[i,j,channel]] += 1

    t_g[0] = h_g[0]
    for i in range(1,256):
        t_g[i] = t_g[i-1]+h_g[i]

    average = rows*cols/new_level

    for i in range(256):
        f_g[i] = max(0, round(t_g[i]/average)-1)

    for i in range(rows):
        for j in range(cols):
            img[i,j,channel] = f_g[img[i,j,channel]]

cv2.imshow("Output", img)
cv2.waitKey(0)