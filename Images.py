import cv2
import numpy as np

# Đọc hình ảnh
image = cv2.imread('BG.jpg')

# height, width, _ = image.shape

# x_center = width // 2
# y_center = height // 2

# square_size = 500

# x1 = x_center - square_size // 2
# y1 = y_center - square_size // 2
# x2 = x_center + square_size // 2
# y2 = y_center + square_size // 2

# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 30)

# Lấy kích thước của hình ảnh
height, width, _ = image.shape

# Kích thước của hình vuông (vd: 100x100 pixels)
square_size = 500

# Tính toán tọa độ góc trên bên trái và góc dưới bên phải của hình vuông
x1 = (width - square_size) // 2
y1 = (height - square_size) // 2
x2 = x1 + square_size
y2 = y1 + square_size

# Màu sắc cho hình vuông (vd: màu đỏ, xanh lá cây, và xanh da trời)
color = (0, 255, 0)  # Màu đỏ trong BGR
# Fill hình vuông với màu
image[y1:y2, x1:x2] = color

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
