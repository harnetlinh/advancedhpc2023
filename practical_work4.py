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
h, w, c = root.shape
pixelCount = h*w

# reshape it
cpu_img = root.copy()

# convert image to BW image function using CPU
def gray_scale(img):
    h1, w1, c1 = cpu_img.shape
    for i in range(h1):
        for j in range(w1):
            gray = img[i][j][0]/3 + img[i][j][1]/3 + img[i][j][2]/3
            img[i][j][0] = gray
            img[i][j][1] = gray
            img[i][j][2] = gray
    return img

t1 = time.time()
cpu_img = gray_scale(cpu_img)
t2 = time.time()

print("Execute time of CPU (s): " + str(t2-t1))
imgplot = plt.imshow(cpu_img)
plt.show()

print("----------------------------------------")

gpu_img = root.copy()
cuda_image_data = cuda.to_device(gpu_img)
output_cuda_image_data = cuda.device_array(np.shape(gpu_img),np.uint8)

# convert image to BW image function using GPU
@cuda.jit 
def grayscale(src, dst): 
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  g = np.uint8(src[tidx, tidy, 0]/3 + src[tidx, tidy, 1]/3 + src[tidx, tidy, 2]/3)
  dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g


t3 = time.time()
blockSize = (8,8)
gridSize = (math.ceil(gpu_img.shape[0] / blockSize[0]), math.ceil(gpu_img.shape[1] / blockSize[1]))
grayscale[gridSize, blockSize](cuda_image_data, output_cuda_image_data)
t4 = time.time()

gpu_img = output_cuda_image_data.copy_to_host()

print("Execute time of GPU (s): " + str(t4-t3))
imgplot = plt.imshow(gpu_img)
plt.show()