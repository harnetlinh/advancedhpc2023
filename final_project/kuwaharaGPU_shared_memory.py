
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import numba

# get the image
root = mpimg.imread('cute-cat.jpg')
imgplot = plt.imshow(root)
plt.show()
print("----------------------------------------")

# convert RGB image to HSV image with shared memory
@cuda.jit
def rgb_to_hsv_with_shared_memory(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    b = src[tidx, tidy, 0] / 255
    g = src[tidx, tidy, 1] / 255
    r = src[tidx, tidy, 2] / 255
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        h = 0
    elif cmax == r:
        h = ((((g - b) / delta) % 6) * 60)  % 360
    elif cmax == g:
        h = ((((b - r) / delta) + 2) * 60) % 360
    elif cmax == b:
        h = ((((r - g) / delta) + 4) * 60) % 360
    if cmax == 0:
        s = 0
    else:
        s = delta / cmax
    v = cmax

    dst[tidx, tidy, 0] = h % 360
    dst[tidx, tidy, 1] = s * 100
    dst[tidx, tidy, 2] = v * 100

# create kuwahara filter with window size variable as input and v in hsv color space as input with shared memory
@cuda.jit
def kuwahara_filter_hsv_with_shared_memory(src, dst, v, window_size):
    # convert image to hsv color space with shared memory
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w = src.shape[0], src.shape[1]   
    # shared memory
    s = cuda.shared.array(shape=(0, 0), dtype=numba.float32)
    s[tidx, tidy] = v[tidx, tidy]
    cuda.syncthreads()

    

