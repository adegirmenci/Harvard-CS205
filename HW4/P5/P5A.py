# Alperen Degirmenci
# CS 205 HW4 P5A

import numpy as np
import matplotlib.image as img

# Image files
in_file_name = "Harvard_Huge.png"
out_file_name = "Harvard_RegionGrow_GPU_A.png"

# Region growing constants [min, max]
seed_threshold = [0, 0.08];
threshold      = [0, 0.27];

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA sharpening kernel as a string
regionGrow_kernel_source = \
'''
__global__ void regionGrow_kernel(const float* image, char* region, const char* this_front, char* next_front, const int width, const int height, const float* threshold)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   if((i_x < width) && (i_y < height))
   {
      int ctr = i_y*width+i_x;

      if(this_front[ctr] == 1)
      {
         float pixel = image[ctr];
         char reg = region[ctr];
         if((reg == 0) && (pixel >= threshold[0]) && (pixel <= threshold[1]))
         {
            region[ctr] = 1;
   
            int up = (i_y-1)*width+i_x;
            int left = ctr-1;
            int right = ctr+1;
            int down = (i_y+1)*width+i_x;

            //expand front
            if(i_x > 0)
               next_front[left] = 1;
            if(i_x < width-1)
               next_front[right] = 1;
            if(i_y > 0)
               next_front[up] = 1;
            if(i_y < height-1)
               next_front[down] = 1;
         }
      }
   }   
   __syncthreads();
}
'''

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

if __name__ == '__main__':
  # Compile the CUDA kernel
  regionGrow_kernel = cuda_compile(regionGrow_kernel_source,"regionGrow_kernel")

  # Read image. BW images have R=G=B so extract the R-value
  original_image = img.imread(in_file_name)[:,:,0]

  # Get image data
  height, width = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)

  # Initialize the image region as empty
  im_region = np.int8(np.zeros([height, width]))

  # On the host, define the kernel parameters
  blocksize = (128,4,1)     #128,8 The number of threads per block (x,y,z)
  gridx = int(np.ceil(width/(1.0*blocksize[0])))
  gridy = int(np.ceil(height/(1.0*blocksize[1])))
  gridsize  = (gridx,gridy)   # The number of thread blocks (x,y)

  # Initialize the GPU event trackers for timing
  start_gpu_time = cu.Event()
  end_gpu_time = cu.Event()
  gpu_transfer_time = 0.0
  gpu_comp_time = 0.0

  # Allocate memory
  image = np.float32(np.array(original_image))
  region = np.int8(np.zeros([height,width]))
  thisFront = np.int8(np.ones([height,width]))
  nextFront = np.int8(np.zeros([height,width]))
  size = width*height

 # Allocate device memory and copy host to device
  start_gpu_time.record()
  d_image = gpu.to_gpu(image.reshape(-1))
  d_region = gpu.to_gpu(region.reshape(-1))
  d_thisFront = gpu.to_gpu(thisFront.reshape(-1))
  d_nextFront = gpu.to_gpu(nextFront.reshape(-1))
  d_seed_threshold = gpu.to_gpu(np.float32(seed_threshold))
  d_threshold = gpu.to_gpu(np.float32(threshold))
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  # Find any seed points in the image and add them to the region
  regionGrow_kernel(d_image,d_region,d_thisFront,d_nextFront,width,height,d_seed_threshold,block=blocksize,grid=gridsize)

  contComp = True
  i = 0;
  # While the next front has pixels we need to process 
  while contComp:
    # Region Growing
    # Swap front sets, this_front <= next_front, next_front = zeros (in kernel)
    d_thisFront, d_nextFront = d_nextFront, d_thisFront
    d_nextFront.fill(0)
    #d_nextFront = gpu.zeros_like(d_thisFront)      
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    regionGrow_kernel(d_image, d_region, d_thisFront, d_nextFront, width, height, d_threshold,block=blocksize, grid=gridsize)
    # nextFront should have all zeroes if there are no more fronts
    moreFronts = gpu.max(d_nextFront).get()
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_comp_time += start_gpu_time.time_till(end_gpu_time)*1e-3
    
    # check if the max element in nextFront is a zero
    if(moreFronts == 0):
      contComp = False # terminate loop
    # Increment counter
    i += 1

  # Copy from device to host
  start_gpu_time.record()
  h_region = d_region.get()
  h_region = h_region.reshape([height,width])
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3
  
  print 'Completed in %d steps' % i
  print "GPU Time: %1.10f" % (gpu_transfer_time+gpu_comp_time)

  print '%d,%d Threads, %d,%d Blocks' % (blocksize[0],blocksize[1],gridsize[0],gridsize[1])

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, h_region, cmap='gray', vmin=0.0, vmax=1.0)
