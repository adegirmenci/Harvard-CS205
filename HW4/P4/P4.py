# Alperen Degirmenci
# CS 205 HW4 P4

import numpy as np
import matplotlib.image as img

# Image files
in_file_name = "Harvard_Huge.png"
out_file_name = "Harvard_Sharpened_GPU.png"
# Sharpening constant
EPSILON = np.float32(0.005)

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
from pycuda.reduction import ReductionKernel
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA sharpening kernel as a string
sharpening_kernel_source = \
'''
__global__ void sharpening_kernel(const float* curr_im, float* next_im, const int width, const int height, const float eps)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   int uleft = (i_y-1)*width+i_x-1;
   int uctr = uleft+1;
   int uright = uctr+1;
   int left = i_y*width+i_x-1;
   int ctr = left+1;
   int right = ctr+1;
   int dleft = (i_y+1)*width+i_x-1;
   int dctr = dleft+1;
   int dright = dctr+1;

   float temp = 0.0;
   if((i_x > 0) && (i_x < width-1) && (i_y >0) && (i_y < height-1))
      temp = curr_im[ctr] + eps * (-1*curr_im[uleft] - 2*curr_im[uctr] - 1*curr_im[uright] - 2*curr_im[left] + 12*curr_im[ctr] - 2*curr_im[right] - 1*curr_im[dleft] - 2*curr_im[dctr] - 1*curr_im[dright]);

   //__syncthreads();

   if((i_x > 0) && (i_x < width-1) && (i_y >0) && (i_y < height-1))
      next_im[ctr] = temp;
}
'''

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def mean_variance(red_kernel, d_data, size):
  '''Return the mean and variance of a 2D array'''
  mean = gpu.sum(d_data,dtype=np.float32).get()/np.float32(size)
  #mean = sum_kernel(d_data,size).get()
  variance = red_kernel(d_data,mean).get()/np.float32(size)
  #print "Mean = %f,  Variance = %f" % (mean, variance)
  return mean, variance

if __name__ == '__main__':
  # Compile the CUDA kernel
  sharpening_kernel = cuda_compile(sharpening_kernel_source,"sharpening_kernel") 
  reduction_kernel = ReductionKernel(np.float32, neutral='0', reduce_expr='a+b', map_expr='std::pow(x[i]-mean,2)',arguments='const float* x, const float mean', name='reduction_kernel')
  #sum_kernel = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]/size",arguments="const float* x, const int size",name="sum_kernel")

  # Read image. BW images have R=G=B so extract the R-value
  original_image = img.imread(in_file_name)[:,:,0]

  # Get image data
  height, width = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)

  # On the host, define the kernel parameters
  blocksize = (128,2,1)     #128,8 The number of threads per block (x,y,z)
  gridx = int(np.ceil(width/(1.0*blocksize[0])))
  gridy = int(np.ceil(height/(1.0*blocksize[1])))
  gridsize  = (gridx,gridy)   # The number of thread blocks (x,y)

  # Initialize the GPU event trackers for timing
  start_gpu_time = cu.Event()
  end_gpu_time = cu.Event()
  gpu_transfer_time = 0.0
  gpu_meanvar_time = 0.0
  gpu_comp_time = 0.0

  # Allocate memory
  image = np.float32(np.array(original_image))
  temp = image
  size = width*height

 # Allocate device memory and copy host to device
  start_gpu_time.record()
  d_image = gpu.to_gpu(image.reshape(-1)) # image
  d_temp = gpu.to_gpu(temp.reshape(-1)) # temp image
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  # Compute the image's initial mean and variance
  start_gpu_time.record()
  init_mean, init_variance = mean_variance(reduction_kernel,d_image,size)
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_meanvar_time += start_gpu_time.time_till(end_gpu_time)*1e-3
  print "Mean = %f,  Variance = %f" % (init_mean, init_variance)
  variance = init_variance

  i = 0; 
  while variance < 1.1 * init_variance:
    # Compute Sharpening
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    sharpening_kernel(d_image, d_temp, width, height, EPSILON, block=blocksize, grid=gridsize)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_comp_time += start_gpu_time.time_till(end_gpu_time)*1e-3      
    # Swap images
    d_image, d_temp = d_temp, d_image
    # Compute the image's pixel mean and variance
    start_gpu_time.record()
    mean, variance = mean_variance(reduction_kernel, d_image, size)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_meanvar_time += start_gpu_time.time_till(end_gpu_time)*1e-3
    print "Mean = %f,  Variance = %f" % (mean, variance)
    # Increment counter
    i += 1

  # Copy from device to host
  start_gpu_time.record()
  h_image = d_image.get()
  h_image = h_image.reshape([height,width])
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  print 'Sharpening completed in %d steps' % i
  print "GPU Total Time: %1.10f" % (gpu_transfer_time+gpu_comp_time+gpu_meanvar_time)
  print "GPU Time Spent on Transfer of Data (to/from): %1.10f" % gpu_transfer_time
  print "GPU Time Spent on Calculating Mean/Variance: %1.10f" % gpu_meanvar_time
  print "GPU Time Spent on Sharpening the Image: %1.10f" % gpu_comp_time
  print '%d,%d Threads, %d,%d Blocks' % (blocksize[0],blocksize[1],gridsize[0],gridsize[1])

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, h_image, cmap='gray', vmin=0.0, vmax=1.0)
