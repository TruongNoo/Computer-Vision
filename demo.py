import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh màu
image = cv2.imread('Trungthu.jpg')

# Tách các kênh màu R, G, B
b, g, r = cv2.split(image)

# Tạo histogram ban đầu
hist_b = np.zeros(256, dtype=np.int)
hist_g = np.zeros(256, dtype=np.int)
hist_r = np.zeros(256, dtype=np.int)

# Tính toán histogram
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        hist_b[b[i, j]] += 1
        hist_g[g[i, j]] += 1
        hist_r[r[i, j]] += 1

# Cân bằng histogram
equalized_b = cv2.equalizeHist(b)
equalized_g = cv2.equalizeHist(g)
equalized_r = cv2.equalizeHist(r)

# Hiển thị histogram và ảnh gốc
plt.figure(figsize=(12, 6))

plt.subplot(231)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(232)
plt.title('Histogram (Blue Channel)')
plt.plot(hist_b, color='blue')

plt.subplot(233)
plt.title('Equalized Histogram (Blue Channel)')
plt.plot(cv2.calcHist([equalized_b], [0], None, [256], [0, 256]), color='blue')

plt.subplot(234)
plt.title('Histogram (Green Channel)')
plt.plot(hist_g, color='green')

plt.subplot(235)
plt.title('Equalized Histogram (Green Channel)')
plt.plot(cv2.calcHist([equalized_g], [0], None, [256], [0, 256]), color='green')

plt.subplot(236)
plt.title('Histogram (Red Channel)')
plt.plot(hist_r, color='red')

plt.tight_layout()
plt.show()