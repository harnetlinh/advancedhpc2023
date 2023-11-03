# create function RGB image to SHV image with numpy and matplotlib

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

# convert RGB image to HSV image with numpy and matplotlib
def rgb_to_hsv(src):
    height, width = src.shape[0], src.shape[1]
    dst = np.zeros((height, width, 3), dtype=np.uint8)
    for tidx in range(height):
        for tidy in range(width):
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

    return dst

# display HSV image
hsv = rgb_to_hsv(root)
imgplot = plt.imshow(hsv)
plt.show()
print("----------------------------------------")

# create kuwahara filter with window size variable as input and v in hsv color space as input
def kuwahara_filter_hsv(src, v, window_size):
    height, width = src.shape[0], src.shape[1]
    dst = np.zeros((height, width, 3), dtype=np.uint8)
    for tidx in range(height):
        for tidy in range(width):
            # first region  rx1 belongs to [tidx - window_size, tidx] and ry1 belongs to [tidy - window_size, tidy]
            sum1 = np.float32(0)
            for rx1 in range(tidx - window_size, tidx): 
                for ry1 in range(tidy - window_size, tidy): 
                    if rx1 >= 0 and ry1 >= 0 and rx1 < height and ry1 < width:
                        sum1 += v[rx1, ry1]
            # find standard deviation of first region
            mean1 = sum1 / ((window_size) * (window_size))
            sum1 = np.float32(0)
            for rx1 in range(tidx - window_size, tidx): 
                for ry1 in range(tidy - window_size, tidy): 
                    if rx1 >= 0 and ry1 >= 0 and rx1 < height and ry1 < width:
                        sum1 += (v[rx1, ry1] - mean1) ** 2
            std1 = math.sqrt(sum1 / ((window_size) * (window_size)))

            # second region  rx2 belongs to [tidx, tidx + window_size] and ry2 belongs to [tidy - window_size, tidy]
            sum2 = np.float32(0)
            for rx2 in range(tidx, tidx + window_size): 
                for ry2 in range(tidy - window_size, tidy): 
                    if rx2 >= 0 and ry2 >= 0 and rx2 < height and ry2 < width:
                        sum2 += v[rx2, ry2]
            # find standard deviation of second region
            mean2 = sum2 / ((window_size) * (window_size))
            sum2 = np.float32(0)
            for rx2 in range(tidx, tidx + window_size): 
                for ry2 in range(tidy - window_size, tidy): 
                    if rx2 >= 0 and ry2 >= 0 and rx2 < height and ry2 < width:
                        sum2 += (v[rx2, ry2] - mean2) ** 2
            std2 = math.sqrt(sum2 / ((window_size) * (window_size)))

            # third region  rx3 belongs to [tidx - window_size, tidx] and ry3 belongs to [tidy, tidy + window_size]
            sum3 = np.float32(0)
            for rx3 in range(tidx - window_size, tidx):
                for ry3 in range(tidy, tidy + window_size):
                    if rx3 >= 0 and ry3 >= 0 and rx3 < height and ry3 < width:
                        sum3 += v[rx3, ry3]
            # find standard deviation of third region
            mean3 = sum3 / ((window_size) * (window_size))
            sum3 = np.float32(0)
            for rx3 in range(tidx - window_size, tidx):
                for ry3 in range(tidy, tidy + window_size):
                    if rx3 >= 0 and ry3 >= 0 and rx3 < height and ry3 < width:
                        sum3 += (v[rx3, ry3] - mean3) ** 2
            std3 = math.sqrt(sum3 / ((window_size) * (window_size)))

            # fourth region  rx4 belongs to [tidx, tidx + window_size] and ry4 belongs to [tidy, tidy + window_size]
            sum4 = np.float32(0)
            for rx4 in range(tidx, tidx + window_size):
                for ry4 in range(tidy, tidy + window_size):
                    if rx4 >= 0 and ry4 >= 0 and rx4 < height and ry4 < width:
                        sum4 += v[rx4, ry4]
            # find standard deviation of fourth region
            mean4 = sum4 / ((window_size) * (window_size))
            sum4 = np.float32(0)
            for rx4 in range(tidx, tidx + window_size):
                for ry4 in range(tidy, tidy + window_size):
                    if rx4 >= 0 and ry4 >= 0 and rx4 < height and ry4 < width:
                        sum4 += (v[rx4, ry4] - mean4) ** 2
            std4 = math.sqrt(sum4 / ((window_size) * (window_size)))

            # find minimum standard deviation
            min_std = min(std1, std2, std3, std4)
            
            # assign the mean of the region with minimum standard deviation to the pixel
            avg_r = np.float32(0)
            avg_g = np.float32(0)
            avg_b = np.float32(0)

            if min_std == std1:
                for rx1 in range(tidx - window_size, tidx): 
                    for ry1 in range(tidy - window_size, tidy): 
                        if rx1 >= 0 and ry1 >= 0 and rx1 < height and ry1 < width:
                            avg_r += src[rx1, ry1, 2]
                            avg_g += src[rx1, ry1, 1]
                            avg_b += src[rx1, ry1, 0]
            if min_std == std2:
                for rx2 in range(tidx, tidx + window_size): 
                    for ry2 in range(tidy - window_size, tidy): 
                        if rx2 >= 0 and ry2 >= 0 and rx2 < height and ry2 < width:
                            avg_r += src[rx2, ry2, 2]
                            avg_g += src[rx2, ry2, 1]
                            avg_b += src[rx2, ry2, 0]

            if min_std == std3:
                for rx3 in range(tidx - window_size, tidx): 
                    for ry3 in range(tidy, tidy + window_size): 
                        if rx3 >= 0 and ry3 >= 0 and rx3 < height and ry3 < width:
                            avg_r += src[rx3, ry3, 2]
                            avg_g += src[rx3, ry3, 1]
                            avg_b += src[rx3, ry3, 0]

            if min_std == std4:
                for rx4 in range(tidx, tidx + window_size): 
                    for ry4 in range(tidy, tidy + window_size): 
                        if rx4 >= 0 and ry4 >= 0 and rx4 < height and ry4 < width:
                            avg_r += src[rx4, ry4, 2]
                            avg_g += src[rx4, ry4, 1]
                            avg_b += src[rx4, ry4, 0]

            avg_r = avg_r / (window_size * window_size)
            avg_g = avg_g / (window_size * window_size)
            avg_b = avg_b / (window_size * window_size)

            dst[tidx, tidy, 2] = avg_r
            dst[tidx, tidy, 1] = avg_g
            dst[tidx, tidy, 0] = avg_b

    return dst

# display kuwahara filtered image
v = hsv[:, :, 2]
# start timer
start = time.time()
filtered = kuwahara_filter_hsv(root, v, 8)
# end timer
end = time.time()
print("Time taken for kuwahara filter in CPU: ", end - start)
imgplot = plt.imshow(filtered)
# save image
plt.imsave('cute-cat-kuwahara_cpu.jpg', filtered)
plt.show()