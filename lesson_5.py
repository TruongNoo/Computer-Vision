import cv2
import math
import numpy as np

def Correlation(H):
       ts = 0
       ms = 0
       H_1 = []
       H_2 = []
       for i in range(len(H[0])):
              H_1.append(H[0][i] - 1/len(H[0])*sum(H[0]))
              H_2.append(H[1][i] - 1/len(H[0])*sum(H[1]))
       for i in range(len(H[0])):
              ts = ts + H_1[i]*H_2[i]
              ms = ms + H_1[i]**2 * H_2[i]**2
       ms = math.sqrt(ms)
       return ts / ms

def Chi_Square(H):
       d = 0
       for i in range(len(H[0])):
              if H[0][i] > H[1][i]:
                     d = d + ((H[0][i] - H[1][i])**2)/(H[0][i]+H[1][i])
              else:
                     d = d + ((H[1][i] - H[0][i])**2)/(H[0][i]+H[1][i])
       return d

def Intersection(H):
       d = 0
       for i in range(len(H[0])):
              if H[0][i] > H[1][i]:
                     d = d + H[1][i]
              else:
                     d = d + H[0][i]
       return d

def Bhattacharyya(H):
       d = 0
       for i in range(len(H[0])):
              d += (math.sqrt(H[0][i] * H[1][i]))/(math.sqrt(sum(H[0])*sum(H[1])))     
       return 1 - d

def getHistogram(I_1, I_2, n):
       I_k = np.zeros((2,n), dtype=np.uint8)       
       t = 256//n
       h_1, w_1 = I_1.shape
       h_2, w_2 = I_2.shape
       for i in range(h_1):
              for j in range(w_1):
                     I_k[0][I_1[i][j] // t] += 1
       for i in range(h_2):
              for j in range(w_2):
                     I_k[1][I_2[i][j] // t] += 1
       return I_k.astype(np.uint8)

def backProjection(I_1, I_2, n):
       H = getHistogram(I_1, I_2, n) 
       t = 256//n      
       h_1, w_1 = I_1.shape
       h_2, w_2 = I_2.shape
       I_k = np.zeros((h_2,w_2), dtype=np.uint8)
       for i in range(h_2):
              for j in range(w_2):
                     I_k[i][j] = H[0][I_2[i][j] // t] 
       return I_k.astype(np.uint8)
n = 8
I_1 = np.array([[1,78,222,54,199,57],
       [144,244,64,34,155,130],
       [67,104,210,248,138,61],
       [248,164,220,45,59,67],
       [205,47,42,70,126,33],
       [225,235,95,123,48,159]])
I_2 = np.array([[7,27,118,141,3,220],
       [228,134,3,14,86,112],
       [184,21,67,85,250,105],
       [251,218,151,131,197,164],
       [7,243,35,167,13,119],
       [139,99,181,228,214,175]])
print("Histogram của 2 ảnh là: ",getHistogram(I_1,I_2,n))
print("Correlation = ", Correlation(getHistogram(I_1,I_2,n)))
print("Chi_Square = ", Chi_Square(getHistogram(I_1,I_2,n)))
print("Intersection = ", Intersection(getHistogram(I_1,I_2,n)))
print("Bhattacharyya = ", Bhattacharyya(getHistogram(I_1,I_2,n)))
print(backProjection(I_2,I_1,n))

img1 = cv2.imread("Test_1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input_1", img1)
cv2.waitKey(33)
img2 = cv2.imread("Test_2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input_2", img2)
cv2.waitKey(22)
print("Histogram của 2 ảnh là: ",getHistogram(img1,img2,n))
print("Correlation = ", Correlation(getHistogram(img1,img2,n)))
print("Chi_Square = ", Chi_Square(getHistogram(img1,img2,n)))
print("Intersection = ", Intersection(getHistogram(img1,img2,n)))
print("Bhattacharyya = ", Bhattacharyya(getHistogram(img1,img2,n)))
cv2.imshow("Output_2", backProjection(img2,img1,n))
cv2.waitKey(0)