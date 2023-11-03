from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math


# get the image
root = mpimg.imread('cute-cat.jpg')
imgplot = plt.imshow(root)
plt.show()
print("----------------------------------------")

gaussian_kernel = [
[0, 0, 1, 2, 1, 0, 0],
[0, 3, 13, 22, 13, 3, 0],
[1, 13, 59, 97, 59, 13, 1],
[2, 22, 97, 159, 97, 22, 2],
[1, 13, 59, 97, 59, 13, 1],
[0, 3, 13, 22, 13, 3, 0],
[0, 0, 1, 2, 1, 0, 0]]

# convert gaussian_kernel to numpy array
gaussian_kernel = np.array(gaussian_kernel)

gaussian_kernel_value = 1003

# convolve RGB image with gaussian_kernel function using CPU 
def convolution(img, kernel):
    h, w, c = img.shape
    kernel_size = kernel.shape[0]
    kernel_offset = kernel_size // 2
    img_conv = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                sum = 0
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        mm = kernel_size - 1 - m
                        nn = kernel_size - 1 - n
                        ii = i + (m - kernel_offset)
                        jj = j + (n - kernel_offset)
                        if (ii >= 0 and ii < h and jj >= 0 and jj < w):
                            sum += img[ii][jj][k] * kernel[mm][nn]
                img_conv[i][j][k] = np.uint8(sum / gaussian_kernel_value)
    return img_conv


t1 = time.time()
cpu_img = convolution(root, gaussian_kernel)
t2 = time.time()

# display the image
print("Execute time of CPU (s): " + str(t2-t1))
imgplot = plt.imshow(cpu_img)
plt.show()


print("----------------------------------------")

# convolve RGB image with gaussian_kernel function using GPU 