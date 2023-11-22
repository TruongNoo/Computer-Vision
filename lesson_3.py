import cv2
import itertools
import collections
import matplotlib.pyplot as plt
import numpy as np

# Đọc hình ảnh
image = cv2.imread('Trungthu.jpg', cv2.IMREAD_GRAYSCALE)
plt.hist(image.ravel(),256,[0,256]); plt.show()
cv2.imshow('Input', image)
cv2.waitKey(33)
newLevel = 256

print(image)

rows = image.shape[0]
cols = image.shape[1]

avg = rows * cols / newLevel
t = []
f = []
hg = []
flat_list = list(itertools.chain.from_iterable(image))
res = collections.Counter(flat_list)
x = sorted(res.items())
for i in range(256):
    hg.append(x[i][1])
    
print(hg)
sum = 0
for i in range(256):
    t.append(sum+x[i][1])
    sum = sum + x[i][1]
    f.append(max(0, round(t[i]/avg)-1))
for i in range(rows):
    for j in range(cols):
        image[i,j] = f[image[i,j]]
        
print(image)

# Sinh ảnh Histogram
#Vẽ đồ thị histogram
plt.hist(image.ravel(),256,[0,256]); plt.show()

cv2.imshow('Output', image)
cv2.waitKey(0)