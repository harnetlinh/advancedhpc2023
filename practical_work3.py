from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# get the image
root = mpimg.imread('cute-cat.jpg')
imgplot = plt.imshow(root)
plt.show()
print("----------------------------------------")
h, w, c = root.shape
pixelCount = h*w

# reshape it
cpu_img = root.copy()
cpu_img = cpu_img.reshape(pixelCount, 3)

# convert image to BW image function using CPU
def gray_scale(img):
  for i in img:
    gray = i[0]/3 + i[1]/3 + i[2]/3
    i[0] = gray
    i[1] = gray
    i[2] = gray
  return img

t1 = time.time()
cpu_img = gray_scale(cpu_img)
cpu_img = np.reshape(cpu_img, (h,w,3))
t2 = time.time()

print("Execute time of CPU (s): " + str(t2-t1))
imgplot = plt.imshow(cpu_img)
plt.show()

print("----------------------------------------")

gpu_img = root.copy().reshape(pixelCount, 3)
cuda_image_data = cuda.to_device(gpu_img)
output_cuda_image_data = cuda.device_array(np.shape(gpu_img),np.uint8)

# convert image to BW image function using GPU
@cuda.jit 
def grayscale(src, dst): 
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
  g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2])/ 3) 
  dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g


t3 = time.time()
blockSize = 64
gridSize = int(pixelCount / blockSize)
grayscale[gridSize, blockSize](cuda_image_data, output_cuda_image_data)
t4 = time.time()

gpu_img = output_cuda_image_data.copy_to_host().reshape(h,w,3)

print("Execute time of GPU (s): " + str(t4-t3))
imgplot = plt.imshow(gpu_img)
plt.show()