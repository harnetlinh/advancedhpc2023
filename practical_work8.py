from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math


# get the image
root = mpimg.imread('cute-cat.jpg')
blending_image1 = mpimg.imread('clown.jpg')
blending_image2 = mpimg.imread('khoile.jpg')
imgplot = plt.imshow(root)
plt.show()
print("----------------------------------------")

# convert RGB image to HSV image
@cuda.jit
def rgb_to_hsv(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    r = src[tidx, tidy, 0]
    g = src[tidx, tidy, 1]
    b = src[tidx, tidy, 2]
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    elif cmax == b:
        h = 60 * (((r - g) / delta) + 4)
    if cmax == 0:
        s = 0
    else:
        s = delta / cmax
    v = cmax
    dst[tidx, tidy, 0] = h
    dst[tidx, tidy, 1] = s
    dst[tidx, tidy, 2] = v


# display the image after converting RGB to HSV image
h, w, c = root.shape
pixelCount = h * w
gpu_img = root.copy()
cuda_image_data = cuda.to_device(gpu_img)
output_cuda_image_data = cuda.device_array(np.shape(gpu_img), np.uint8)
blockSize = (8, 8)
gridSize = (math.ceil(gpu_img.shape[0] / blockSize[0]), math.ceil(gpu_img.shape[1] / blockSize[1]))
rgb_to_hsv[gridSize, blockSize](cuda_image_data, output_cuda_image_data)
gpu_img = output_cuda_image_data.copy_to_host()
imgplot = plt.imshow(gpu_img)
plt.show()

