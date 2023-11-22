import cv2
import numpy as np
import math

def ndf(x, mu, sigma):
    


def convolution(I,T):
    M, N = I.shape
    m, n = T.shape
    
    rows = M - m + 1
    cols = N - n + 1
    
    I_k = np.zeros((rows,cols), dtype=np.uint8)
    for x in range(rows):
        for y in range(cols):
            result = 0
            for i in range(m):
                for j in range(n):
                    result += I[x+i][y+j] * T[i][j]
            I_k[x][y] = result
            
    return I_k

img = cv2.imread("Trungthu.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Input", img)
cv2.waitKey(33)