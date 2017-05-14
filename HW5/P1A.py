# Alperen Degirmenci
# CS 205 HW5 P1A

import numpy as np
import cv2
from cv2 import cv

# Image files
in_file_name = "testimage.jpg"
out_file_name = "stretchdown.jpg"
carveSize = 1200 #pixels

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA sharpening kernel as a string
energy_kernel_source = \
'''
__global__ void energy_kernel(const uchar3* image, float* energy, const int width, const int height, const int iter)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   if((i_x < width-iter) && (i_y < height))
   {
      int ctr = i_y*width+i_x;
      int right = ctr+1;
      int down = (i_y+1)*width+i_x;

      float I_ctr = (float)(image[ctr].x + image[ctr].y + image[ctr].z);
      float I_right = 0.0;
      float I_down = 0.0;

      if(i_x < width-1-iter)
      {
         I_right = (float)(image[right].x + image[right].y + image[right].z);
      }
      if(i_y < height-1)
      {
         I_down = (float)(image[down].x + image[down].y + image[down].z);
      }
      energy[ctr] = abs(I_right - I_ctr) + abs(I_down - I_ctr);
   }
   __syncthreads();
}
'''
cmlEnergy_kernel_source = \
'''
__global__ void cmlEnergy_kernel(const float* energy, float* cmlEnergy, const int width, const int height, const int row, const int iter)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   //int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   int ctr = row*width+i_x;

   if((i_x < width-iter) && (0 < row) && (row < height))
   {
      int top = (row-1)*width+i_x;
      int left = top-1;
      int right = top+1;

      float C_top = cmlEnergy[top];
      float C_left = 999999999;//std::numeric_limits<float>::max();
      float C_right = 999999999;//std::numeric_limits<float>::max();

      if(0 < i_x)
      {
         C_left = cmlEnergy[left];
      }
      if(i_x < width-1-iter)
      {
         C_right = cmlEnergy[right];
      }
      cmlEnergy[ctr] = min(C_top, min(C_left, C_right)) + energy[ctr];
   }
   else if(row == 0)
      cmlEnergy[ctr] = energy[ctr];

   __syncthreads();
}
'''
min_kernel_source = \
'''
__global__ void min_kernel(const float* cmlEnergy, int* trace, const int width, const int height, const int row, const int iter)
{
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   if((tid == 0) && (row < height-1))
   {
      int ctr = trace[row+1]-width;
      int left = ctr-1;
      int right = ctr+1;
      int x = ctr - row*width;
      float C_ctr = cmlEnergy[ctr];
      float C_left = 999999999.0;
      float C_right = 999999999.0;

      if(0 < x)
         C_left = cmlEnergy[left];
      if(x < width-1-iter)
         C_right = cmlEnergy[right];

      int min_idx = left;
      float min_val = C_left;
      if(C_ctr <= min_val)
      {
         min_val = C_ctr;
         min_idx = ctr;
      }
      if(C_right < min_val)
      {
         min_val = C_right;
         min_idx = right;
      }

      trace[row] = min_idx;
   }
   else if((tid == 0) && (row == (height-1)))
   {
      int min_idx = row*width;
      float min_val = 999999999.0;
      for(int i = 0; i < width-iter; i++)
      {
         float cml = cmlEnergy[row*width+i];
         if(cml < min_val)
         {
            min_val = cml;
            min_idx = row*width+i;
         }
      }
      trace[row] = min_idx;
   }
}
'''
crop_kernel_source = \
'''
__global__ void crop_kernel(const uchar3* image, uchar3* cropped, int* trace, const int width, const int height, const int iter)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;
   int idx = i_y*width + i_x;

   if((i_x < width-1-iter) && (i_y < height))
   {
      int cropIdx = trace[i_y];
      if(idx < cropIdx)
         cropped[idx] = image[idx];
      else
         cropped[idx] = image[idx+1];
   }

   //__syncthreads();
}
'''

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

if __name__ == '__main__':
  # Compile the CUDA kernel
  energy_kernel = cuda_compile(energy_kernel_source,"energy_kernel")
  cmlEnergy_kernel = cuda_compile(cmlEnergy_kernel_source,"cmlEnergy_kernel")
  min_kernel = cuda_compile(min_kernel_source,"min_kernel")
  crop_kernel = cuda_compile(crop_kernel_source,"crop_kernel")

  # Read image using OpenCV / uint8 ndarray
  original_image = cv2.imread(in_file_name)

  # Get image data
  height, width, layers = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)

  # On the host, define the kernel parameters
  blocksize = (128,4,1)     #128,8 The number of threads per block (x,y,z)
  gridx = int(np.ceil(width/(1.0*blocksize[0])))
  gridy = int(np.ceil(height/(1.0*blocksize[1])))
  gridsize  = (gridx,gridy)   # The number of thread blocks (x,y)
  # Block and grid size for the cumulative energy calculation
  blocksize_cml = (128,1,1)
  gridsize_cml = (int(np.ceil(width/(1.0*blocksize_cml[0]))),1)

  # Initialize the GPU event trackers for timing
  start_gpu_time = cu.Event()
  end_gpu_time = cu.Event()
  gpu_transfer_time = 0.0
  gpu_energy_time = 0.0
  gpu_cml_time = 0.0
  gpu_trace_time = 0.0
  gpu_crop_time = 0.0

  # Make sure image is uint8
  image = np.uint8(original_image)
  size = width*height

 # Allocate device memory and copy host to device
  start_gpu_time.record()
  d_image = gpu.to_gpu(image.astype(np.uint8).view(gpu.vec.uchar3))
  d_energy = gpu.zeros(shape=int(height*width), dtype=np.float32)
  d_cmlEnergy = gpu.zeros_like(d_energy)
  d_trace = gpu.zeros(shape=int(height), dtype=np.int32)
  d_cropped = gpu.zeros(shape=int(height*width), dtype=gpu.vec.uchar3)
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  i = 0;
  # While the next front has pixels we need to process 
  while i < carveSize:
    # Region Growing
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    energy_kernel(d_image, d_energy, width, height, np.int32(i), block=blocksize, grid=gridsize) #energy
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_energy_time += start_gpu_time.time_till(end_gpu_time)*1e-3

    for row in xrange(0,height):
      start_gpu_time.record()
      cmlEnergy_kernel(d_energy, d_cmlEnergy, width, height, np.int32(row), np.int32(i), block=blocksize_cml, grid=gridsize_cml)
      end_gpu_time.record()
      end_gpu_time.synchronize()
      gpu_cml_time += start_gpu_time.time_till(end_gpu_time)*1e-3

    for row in reversed(xrange(0,height)):
      start_gpu_time.record()
      min_kernel(d_cmlEnergy, d_trace, width, height, np.int32(row), np.int32(i), block=(1,1,1), grid=(1,1))
      end_gpu_time.record()
      end_gpu_time.synchronize()
      gpu_trace_time += start_gpu_time.time_till(end_gpu_time)*1e-3

    start_gpu_time.record()
    crop_kernel(d_image, d_cropped, d_trace, width, height, np.int32(i), block=blocksize, grid=gridsize)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_crop_time += start_gpu_time.time_till(end_gpu_time)*1e-3

    start_gpu_time.record()
    d_image, d_cropped = d_cropped, d_image # cropped image becomes the input for the next iteration
    d_cropped.fill(np.uint8(0)) # reset to 0
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3
    
    # Increment counter
    i += 1

  # Copy from device to host
  start_gpu_time.record()
  h_image = d_image.get().view(np.uint8).reshape([height,width,3])
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3
  
  print 'Completed in %d steps' % i
  print "Total GPU Time: %1.10f" % (gpu_transfer_time+gpu_energy_time+gpu_cml_time+gpu_trace_time+gpu_crop_time)
  print "Energy Kernel Time: %1.10f" % (gpu_energy_time)
  print "Cumulative Energy Kernel Time: %1.10f" % (gpu_cml_time)
  print "Backtrace Kernel Time: %1.10f" % (gpu_trace_time) 
  print "Cropping Kernel Time: %1.10f" % (gpu_crop_time)
  print "Data Transfer Time: %1.10f" % (gpu_transfer_time)
  print '%d,%d Threads, %d,%d Blocks' % (blocksize[0],blocksize[1],gridsize[0],gridsize[1])

  # Save the current image
  cv2.imwrite(out_file_name, h_image[0:height,0:width-carveSize,:])
