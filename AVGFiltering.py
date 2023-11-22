import cv2
import numpy as np
import math

def avgFiltering(I, size, theta):
    rows, cols = I.shape
    d = size // 2
    
    I_kq = np.zeros((rows, cols))
    
    for i in range(d):
        for j in range(cols):
            I_kq[i][j] = I[i][j]
            I_kq[rows-i-1][j]=I[rows-i-1][j]
            
    for i in range(d):
        for j in range(rows):
            I_kq[j][i] = I[j][i]
            I_kq[j][cols-i-1]=I[j][cols-i-1]
                
    for i in range(rows-size+1):
        for j in range(cols - size + 1):
            AVG = 0
            for h in range(i, i + size, 1):
                for k in range(j, j + size, 1):
                    AVG += I[h][k]
            I_p = I[i+d][j+d]
            AVG = round(AVG / (size**2))
            if abs(I_p - AVG) <= theta:
                I_kq[i+d][j+d] = I_p
            else:
                I_kq[i+d][j+d] = AVG
            
    return I_kq.astype(np.uint8)
img = cv2.imread("Trungthu.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input", img)
cv2.waitKey(33)
print(img.shape)
# I = np.array([[1, 0, 3, 5,7,2],
#               [0,1,2,1,2,4],
#               [3,3,4,6,3,7],
#               [1,2,5,3,4,6],
#               [1,5,2,4,2,4],
#               [4,2,1,0,0,2]])
# print(avgFiltering(I,3,2))
I_1 = avgFiltering(img,3,2)
cv2.imshow("Output",I_1)
cv2.waitKey(0)

print(I_1)