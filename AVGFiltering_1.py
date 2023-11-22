import numpy as np
import cv2

# Đường dẫn đến tệp ảnh
path = "Filtering.png"
# Định nghĩa ảnh dưới dạng một mảng NumPy
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Inputs", img)
cv2.waitKey(3)  # Chờ bất kỳ phím nào được nhấn để tiếp tục

I = np.array([[1,0,3,5,7,2],\
     [0,1,2,1,2,4],
     [3,3,4,6,3,7],
     [1,2,5,3,4,6],
     [1,5,2,4,2,4],
     [4,2,1,0,0,2]])

I_2 = np.array([[5,0,6,7,7,0],
     [4,7,6,5,1,1],
     [7,0,3,7,5,5],
     [4,6,3,4,0,4],
     [6,5,4,0,7,6],
     [7,4,3,0,3,5]])

def convolution(img, kernel_size, limit):
    # Lấy kích thước của ảnh và kernel
    kernel = (1 / kernel_size**2) * np.ones((kernel_size, kernel_size))
    M, N = img.shape
    m, n = kernel.shape

    # Khởi tạo ma trận đầu ra Y
    Y = np.zeros((M - m + 1, N - n + 1), dtype=np.float32)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            img_patch = img[i: i + m, j: j + n]
            Y[i, j] = np.sum(img_patch * kernel)

    X = np.zeros((M, N), dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (i == 0 or i == X.shape[0] - 1 or j == 0 or j == X.shape[1] - 1) : X[i, j] = img [i, j]
            else:
                if abs(round(Y[i-1, j-1]) - img[i, j]) <= limit:
                    X[i, j] = img[i,j]
                else:
                    X[i, j] = Y[i-1, j-1]
    return X.astype(np.uint8)


# # Kernel Gaussian
T0 = convolution(I, 3, 2)
print(T0)

T1 = convolution(I_2, 3, 2)
print(T1)
# T1 = convolution(img, (1 / 9) * np.ones((3, 3)))
T2 = convolution(img, 3, 0.5)
# T4 = convolution(img, (1 / 49) * np.ones((7, 7)))
#
# # Kernel Laplace
# k = np.array([[0, 1, 0],
#               [1, -4, 1],
#               [0,  1, 0]])
#
# T5 = convolution(img, k)
#
# # Nếu tổng = 1, ta làm rõ các đặc trưng
# k = np.array([[0, -1, 0],
#               [-1, 5, -1],
#               [0, -1, 0]])
# T6 = convolution(T2, k)
#
# # Nếu tổng = 0 thì ta chiết xuất các đặc trưng
# k = np.array([[0, -1, 0],
#               [-1, 4, -1],
#               [0, -1, 0]])
# T7 = convolution(T2, k)
#
#
# # làm nét ảnh
# k = np.array([[-1, -1, -1],
#               [-1, 9, -1],
#               [-1, -1, -1]])
#
# T8 = convolution(img, k)
#
# # Hiển thị kết quả
# cv2.imshow("T0", T0)
# cv2.imshow("T1", T1)
cv2.imshow("T2", T2)
print(T2)
# cv2.imshow("T4", T4)
# cv2.imshow("T5", T5)
# cv2.imshow("T6", T6)
# cv2.imshow("T7", T7)
# cv2.imshow("T8", T8)
cv2.waitKey(0)
cv2.destroyAllWindows()