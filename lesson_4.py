import cv2
import numpy as np

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
            

# img = cv2.imread("BG.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Input", img)
# cv2.waitKey(77)

# rows, cols = img.shape

# T_1 = 1/9 * np.ones((3,3))
# T_2 = 1/25 * np.ones((5,5))
# T_3 = 1/49 * np.ones((7,7))
# I_1 = convolution(img,T_1)
# I_2 = convolution(img,T_2)
# I_3 = convolution(img,T_3)
# cv2.imshow("Output_1", I_1)
# cv2.waitKey(66)
# cv2.imshow("Output_2", I_2)
# cv2.waitKey(55)
# cv2.imshow("Output_3", I_3)
# cv2.waitKey(44)

img1 = cv2.imread("Anhmo.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input_1", img1)
cv2.waitKey(33)

T_4 = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]])
T_5 = np.array([[-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]])
T_6 = np.array([[-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1],
                [-1,-1,25,-1,-1],
                [-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1]])
I_4 = convolution(img1,T_4)
I_5 = convolution(img1,T_5)
I_6 = convolution(img1,T_6)

cv2.imshow("Output_4", I_4)
cv2.waitKey(22)
cv2.imshow("Output_5", I_5)
cv2.waitKey(11)
cv2.imshow("Output_6", I_6)
cv2.waitKey(0)