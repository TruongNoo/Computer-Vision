import cv2
import os
import numpy as np
import math

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
#Ý 1:
# image_folder = "D:\Code\Python\Computer Vision\Picture\House"

# image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
# i = 0
# for image_file in image_files:
#     i += 1
#     image_path = os.path.join(image_folder, image_file) 
#     img = cv2.imread(image_path)
#     cv2.imshow(image_file, img)
#     cv2.waitKey(1000-10*i)
# cv2.waitKey(0)

#Ý 2:
n = 8
CS = 1000000000
I = 0
B = 1000000000
image_input = cv2.imread("Cat.jpg", cv2.IMREAD_GRAYSCALE)
path = "D:\Code\Python\Computer Vision\Picture\\"
dir_list = os.listdir(path)
N = ""
N1 = ""
N2 = ""
N3 = ""
for name in dir_list:
    name_path = path + name
    image_files = [f for f in os.listdir(name_path) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    Clt = 0
    for image_file in image_files:
        image_path = os.path.join(name_path, image_file) 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        H = getHistogram(image_input,img,n)
        # if Clt < Correlation(H):
        #     Clt = Correlation(H)
        #     N = name
        if CS > Chi_Square(H):
            CS = Chi_Square(H)
            N1 = name
            cv2.imshow("Output", img)
            cv2.waitKey(22)
        # if I < Intersection(H):
        #     I = Intersection(H)
        #     N2 = name
        # if B > Bhattacharyya(H):
        #     B = Bhattacharyya(H)
        #     N3 = name
print("The image mentions", N1)
cv2.waitKey(0)