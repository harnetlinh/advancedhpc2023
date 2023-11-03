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

# convert RGB image to HSV image
@cuda.jit
def rgb_to_hsv(src, dst):
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


# create kuwahara filter with window size variable as input and v in hsv color space as input
@cuda.jit
def kuwahara_filter_hsv(src, dst, v, window_size):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w = src.shape[0], src.shape[1]

    # first region  rx1 belongs to [tidx - window_size, tidx] and ry1 belongs to [tidy - window_size, tidy]
    sum1 = 0
    for rx1 in range(tidx - window_size, tidx): 
        for ry1 in range(tidy - window_size, tidy): 
            if rx1 >= 0 and ry1 >= 0 and rx1 < h and ry1 < w:
                sum1 += v[rx1, ry1]
    # find standard deviation of first region
    mean1 = sum1 / ((window_size) * (window_size))
    sum1 = 0
    for rx1 in range(tidx - window_size, tidx): 
        for ry1 in range(tidy - window_size, tidy): 
            if rx1 >= 0 and ry1 >= 0 and rx1 < h and ry1 < w:
                sum1 += (v[rx1, ry1] - mean1) ** 2
    std1 = (math.sqrt(sum1 / ((window_size) * (window_size))))

    # second region  rx2 belongs to [tidx, tidx + window_size] and ry2 belongs to [tidy - window_size, tidy]
    sum2 = 0
    for rx2 in range(tidx, tidx + window_size): 
        for ry2 in range(tidy - window_size, tidy): 
            if rx2 >= 0 and ry2 >= 0 and rx2 < h and ry2 < w:
                sum2 += v[rx2, ry2]
    # find standard deviation of second region
    mean2 = sum2 / ((window_size) * (window_size))
    sum2 = 0
    for rx2 in range(tidx, tidx + window_size): 
        for ry2 in range(tidy - window_size, tidy): 
            if rx2 >= 0 and ry2 >= 0 and rx2 < h and ry2 < w:
                sum2 += (v[rx2, ry2] - mean2) ** 2
    std2 = (math.sqrt(sum2 / ((window_size) * (window_size))))

    # third region  rx3 belongs to [tidx - window_size, tidx] and ry3 belongs to [tidy, tidy + window_size]
    sum3 = 0
    for rx3 in range(tidx - window_size, tidx): 
        for ry3 in range(tidy, tidy + window_size): 
            if rx3 >= 0 and ry3 >= 0 and rx3 < h and ry3 < w:
                sum3 += v[rx3, ry3]
    # find standard deviation of third region
    mean3 = sum3 / ((window_size) * (window_size))
    sum3 = 0
    for rx3 in range(tidx - window_size, tidx): 
        for ry3 in range(tidy, tidy + window_size): 
            if rx3 >= 0 and ry3 >= 0 and rx3 < h and ry3 < w:
                sum3 += (v[rx3, ry3] - mean3) ** 2
    std3 = (math.sqrt(sum3 / ((window_size) * (window_size))))

    # fourth region  rx4 belongs to [tidx, tidx + window_size] and ry4 belongs to [tidy, tidy + window_size]
    sum4 = 0
    for rx4 in range(tidx, tidx + window_size): 
        for ry4 in range(tidy, tidy + window_size): 
            if rx4 >= 0 and ry4 >= 0 and rx4 < h and ry4 < w:
                sum4 += v[rx4, ry4]
    # find standard deviation of fourth region
    mean4 = sum4 / ((window_size) * (window_size))
    sum4 = 0
    for rx4 in range(tidx, tidx + window_size): 
        for ry4 in range(tidy, tidy + window_size): 
            if rx4 >= 0 and ry4 >= 0 and rx4 < h and ry4 < w:
                sum4 += (v[rx4, ry4] - mean4) ** 2
    std4 = (math.sqrt(sum4 / ((window_size) * (window_size))))
    
    # # find the region with the lowest standard deviation
    min_std = min(std1, std2, std3, std4)

    # assign the mean of the region with the lowest standard deviation to the pixel
    avg_r = 0
    avg_g = 0
    avg_b = 0

    if min_std == std1:
        for rx1 in range(tidx - window_size, tidx): 
            for ry1 in range(tidy - window_size, tidy): 
                if rx1 >= 0 and ry1 >= 0 and rx1 < h and ry1 < w:
                    avg_r += src[rx1, ry1, 2]
                    avg_g += src[rx1, ry1, 1]
                    avg_b += src[rx1, ry1, 0]
    elif min_std == std2:
        for rx2 in range(tidx, tidx + window_size): 
            for ry2 in range(tidy - window_size, tidy): 
                if rx2 >= 0 and ry2 >= 0 and rx2 < h and ry2 < w:
                    avg_r += src[rx2, ry2, 2]
                    avg_g += src[rx2, ry2, 1]
                    avg_b += src[rx2, ry2, 0]
    elif min_std == std3:
        for rx3 in range(tidx - window_size, tidx): 
            for ry3 in range(tidy, tidy + window_size): 
                if rx3 >= 0 and ry3 >= 0 and rx3 < h and ry3 < w:
                    avg_r += src[rx3, ry3, 2]
                    avg_g += src[rx3, ry3, 1]
                    avg_b += src[rx3, ry3, 0]
    else:
        for rx4 in range(tidx, tidx + window_size): 
            for ry4 in range(tidy, tidy + window_size): 
                if rx4 >= 0 and ry4 >= 0 and rx4 < h and ry4 < w:
                    avg_r += src[rx4, ry4, 2]
                    avg_g += src[rx4, ry4, 1]
                    avg_b += src[rx4, ry4, 0]

    avg_r = avg_r / ((window_size) * (window_size))
    avg_g = avg_g / ((window_size) * (window_size))
    avg_b = avg_b / ((window_size) * (window_size))
    dst[tidx, tidy, 2] = avg_r
    dst[tidx, tidy, 1] = avg_g
    dst[tidx, tidy, 0] = avg_b


h, w, c = root.shape
pixelCount = h * w
gpu_img = root.copy()
cuda_image_data = cuda.to_device(gpu_img)

output_cuda_image_data_hsv = cuda.device_array(np.shape(gpu_img), np.uint8)
blockSize = (16, 16)
gridSize = (math.ceil(gpu_img.shape[0] / blockSize[0]), math.ceil(gpu_img.shape[1] / blockSize[1]))
rgb_to_hsv[gridSize, blockSize](cuda_image_data, output_cuda_image_data_hsv)
hsv_image = output_cuda_image_data_hsv.copy_to_host()
# imgplot = plt.imshow(gpu_img)
plt.imsave('cute-cat-hsv.jpg', hsv_image)

# get the V channel
v = hsv_image[:, :, 2]

# apply the kuwahara filter
output_cuda_image_data = cuda.device_array(np.shape(gpu_img), np.uint8)

v = np.ascontiguousarray(v)
# start timer
start = time.time()
kuwahara_filter_hsv[gridSize, blockSize](gpu_img, output_cuda_image_data, v, 8)
# end timer
end = time.time()
print("Time taken for kuwahara in GPU: ", end - start)
dst = output_cuda_image_data.copy_to_host()
plt.imsave('cute-cat-kuwahara_gpu.jpg', dst)
# display the image
imgplot = plt.imshow(dst)