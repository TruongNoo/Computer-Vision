import cv2
import numpy as np

def medFiltering(I, size, theta):
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
            Med = 0
            arr = []
            for h in range(i, i + size, 1):
                for k in range(j, j + size, 1):
                    arr.append(I[h][k])
            arr.sort()
            Med = arr[len(arr)//2]
            I_p = I[i+d][j+d]
            if abs(I_p - Med) <= theta:
                I_kq[i+d][j+d] = I_p
            else:
                I_kq[i+d][j+d] = Med
            
    return I_kq.astype(np.uint8)
img = cv2.imread("Trungthu.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input", img)
cv2.waitKey(33)
# print(img.shape)
# I = np.array([[6,7,1,1,0,3],
#               [4,0,2,5,6,5],
#               [4,7,2,7,6,3],
#               [6,6,7,1,3,6],
#               [5,0,0,7,5,5],
#               [1,1,5,3,6,2]])
# print(medFiltering(I,3,1))
I_1 = medFiltering(img,3,2)
cv2.imshow("Output",I_1)
cv2.waitKey(0)

# print(I_1)