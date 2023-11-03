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

# grayscale the image binarization function using GPU with if statement
@cuda.jit
def binarization(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if src[tidx, tidy, 0] > 200:
        dst[tidx, tidy, 0] = 255
        dst[tidx, tidy, 1] = 255
        dst[tidx, tidy, 2] = 255
    else:
        dst[tidx, tidy, 0] = 0
        dst[tidx, tidy, 1] = 0
        dst[tidx, tidy, 2] = 0


# display the image
h, w, c = root.shape
pixelCount = h * w
gpu_img = root.copy()
cuda_image_data = cuda.to_device(gpu_img)
output_cuda_image_data = cuda.device_array(np.shape(gpu_img), np.uint8)
blockSize = (8, 8)
gridSize = (math.ceil(gpu_img.shape[0] / blockSize[0]), math.ceil(gpu_img.shape[1] / blockSize[1]))
binarization[gridSize, blockSize](cuda_image_data, output_cuda_image_data)
gpu_img = output_cuda_image_data.copy_to_host()
imgplot = plt.imshow(gpu_img)
plt.show()

print("----------------------------------------")

# brightness the image function using GPU without using if statement
@cuda.jit
def brightness(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    dst[tidx, tidy, 0] = src[tidx, tidy, 0] + 50
    dst[tidx, tidy, 1] = src[tidx, tidy, 1] + 50
    dst[tidx, tidy, 2] = src[tidx, tidy, 2] + 50


# display the image
h, w, c = root.shape
pixelCount = h * w
gpu_img = root.copy()
cuda_image_data = cuda.to_device(gpu_img)
output_cuda_image_data = cuda.device_array(np.shape(gpu_img), np.uint8)
blockSize = (8, 8)
gridSize = (math.ceil(gpu_img.shape[0] / blockSize[0]), math.ceil(gpu_img.shape[1] / blockSize[1]))
brightness[gridSize, blockSize](cuda_image_data, output_cuda_image_data)
gpu_img = output_cuda_image_data.copy_to_host()
imgplot = plt.imshow(gpu_img)
plt.show()


print("----------------------------------------")
# blend 2 images into 1 image function using GPU without using if statement
@cuda.jit
def blend(src1, src2, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    dst[tidx, tidy, 0] = (src1[tidx, tidy, 0] + src2[tidx, tidy, 0]) / 2
    dst[tidx, tidy, 1] = (src1[tidx, tidy, 1] + src2[tidx, tidy, 1]) / 2
    dst[tidx, tidy, 2] = (src1[tidx, tidy, 2] + src2[tidx, tidy, 2]) / 2


# display the blended image from 2 images blending_image1 and blending_image2
h, w, c = blending_image1.shape
pixelCount = h * w
blending_gpu_img1 = blending_image1.copy()
blending_gpu_img2 = blending_image2.copy()
cuda_image_data1 = cuda.to_device(blending_gpu_img1)    
cuda_image_data2 = cuda.to_device(blending_gpu_img2)
output_cuda_image_data = cuda.device_array(np.shape(blending_gpu_img1), np.uint8)
blockSize = (8, 8)
gridSize = (math.ceil(blending_gpu_img1.shape[0] / blockSize[0]), math.ceil(blending_gpu_img1.shape[1] / blockSize[1]))
blend[gridSize, blockSize](cuda_image_data1, cuda_image_data2, output_cuda_image_data)
blending_gpu_img = output_cuda_image_data.copy_to_host()
imgplot = plt.imshow(blending_gpu_img)
plt.show()