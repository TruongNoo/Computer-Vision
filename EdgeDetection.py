import numpy as np
import cv2
import math

def gradientEdgeDetection(I):
    m, n = I.shape
    H_x = np.array([[-1, 1]])
    H_y = np.array([[-1],
                    [1]])
    I_k = np.zeros((m-1,n-1), dtype=np.int32)
    rows = m - 1
    cols = n - 1 
    for x in range(rows):
        for y in range(cols):
            I_k[x][y] = I[x+1][y] - 2 * I[x][y] + I[x][y+1]           
    return I_k.astype(np.int32)

def prewittEdgeDetection(I):
    m, n = I.shape
    H_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])
    H_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])
    I_k = np.zeros((m-2,n-2), dtype=np.int32)
    rows = m - 2
    cols = n - 2 
    for x in range(rows):
        for y in range(cols):
            result_1 = 0
            result_2 = 0
            for i in range(3):
                for j in range(3):
                    result_1 += I[x+i][y+j] * H_x[i][j]
                    result_2 += I[x+i][y+j] * H_y[i][j]
            G = math.sqrt(result_1**2 + result_2**2)
            I_k[x,y] = G if G < 255 else 255
            
    return I_k.astype(np.int32)

def sobelEdgeDetection(I):
    m, n = I.shape
    H_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    H_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
    I_k = np.zeros((m-2,n-2), dtype=np.int32)
    rows = m - 2
    cols = n - 2 
    for x in range(rows):
        for y in range(cols):
            result_1 = 0
            result_2 = 0
            for i in range(3):
                for j in range(3):
                    result_1 += I[x+i][y+j] * H_x[i][j]
                    result_2 += I[x+i][y+j] * H_y[i][j]
            G = math.sqrt(result_1**2 + result_2**2)
            I_k[x,y] = G if G < 255 else 255
            
    return I_k.astype(np.int32)

def sobelEdgeDetectionMod(I):
    m, n = I.shape
    H_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    H_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
    I_k = np.zeros((m-2,n-2), dtype=np.uint8)
    rows = m - 2
    cols = n - 2 
    for x in range(rows):
        for y in range(cols):
            result_1 = 0
            result_2 = 0
            for i in range(3):
                for j in range(3):
                    result_1 += I[x+i][y+j] * H_x[i][j]
                    result_2 += I[x+i][y+j] * H_y[i][j]
            G = math.sqrt(result_1**2 + result_2**2)
            I_k[x,y] = G if G < 255 else 255
    return I_k.astype(np.uint8)

img = cv2.imread("HHA.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input", img)
cv2.waitKey(33)

T = np.array([[0,0,0,0,0,0],
            [5,5,5,5,0,0],
            [5,5,5,5,0,0],
            [5,5,5,5,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])
Test = np.array([[0,0,0,0],
                 [0,3,3,3],
                 [0,3,3,3],
                 [0,3,3,3]])
print(gradientEdgeDetection(Test))
print(gradientEdgeDetection(T))
print(prewittEdgeDetection(T))
print(sobelEdgeDetection(T))
image = sobelEdgeDetectionMod(img)
print(image)
if image.dtype == np.float32:
    image = cv2.convertScaleAbs(image)
elif image.dtype == np.int32:
    image = cv2.convertScaleAbs(image)
cv2.imshow('Output', image)
cv2.waitKey(0)