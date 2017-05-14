# Alperen Degirmenci
# CS 205 HW4 P5B

import numpy as np
import matplotlib.image as img

# Image files
in_file_name = "Harvard_Huge.png"
out_file_name = "Harvard_RegionGrow_GPU_B.png"

# Region growing constants [min, max]
seed_threshold = [0, 0.08];
threshold      = [0, 0.27];

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.scan as scan
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA kernel as a string
queue_kernel_source = \
'''
__global__ void queue_kernel(const int* front, const int* scan, int* queue, const int width, const int height)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;
   int ctr = i_y*width+i_x;

   if((i_x < width) && (i_y < height))
   {
      if(front[ctr] == 1)
      {
         int scanIdx = scan[ctr];
         queue[scanIdx] = ctr;
      }
   }
   __syncthreads();
}
'''

regionGrow_kernel_source = \
'''
__global__ void regionGrow_kernel(const float* image, char* region, const int* queue, int* next_front, const int width, const int height, const int qLength, const float* threshold)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;
   int ctr = i_y*width+i_x;

   //if((i_x < width) && (i_y < height))
   if(ctr < qLength)
   {
      int queIdx = queue[ctr];
      int x = queIdx % width;
      int y = queIdx / width;
      //int x = queIdx - y*width;

      float pixel = image[queIdx];
      char reg = region[queIdx];
      if((reg == 0) && (pixel >= threshold[0]) && (pixel <= threshold[1]))
      {
         region[queIdx] = 1;
   
         int up = (y-1)*width+x;
         int left = queIdx-1;
         int right = queIdx+1;
         int down = (y+1)*width+x;

         //expand front
         if(x > 0)
            next_front[left] = 1;
         if(x < width-1)
            next_front[right] = 1;
         if(y > 0)
            next_front[up] = 1;
         if(y < height-1)
            next_front[down] = 1;
      }
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
  regionGrow_kernel = cuda_compile(regionGrow_kernel_source,"regionGrow_kernel")
  queue_kernel = cuda_compile(queue_kernel_source, "queue_kernel")
  scan_kernel = scan.ExclusiveScanKernel(np.int32, "a+b", "0")

  # Read image. BW images have R=G=B so extract the R-value
  original_image = img.imread(in_file_name)[:,:,0]

  # Get image data
  height, width = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)
  size = width*height

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
  cpu_compute_time = 0.0

  # Allocate memory
  image = np.float32(np.array(original_image))
  region = np.int8(np.zeros([height,width]))
  queue = np.int32(np.linspace(0,size-1,size))
  nextFront = np.int32(np.zeros([height,width]))
  scan = np.int32(np.zeros([height,width]))
  qLen = np.int32(queue.size)
 # Allocate device memory and copy host to device
  start_gpu_time.record()
  d_image = gpu.to_gpu(image.reshape(-1))
  d_region = gpu.to_gpu(region.reshape(-1))
  d_queue = gpu.to_gpu(queue.reshape(-1))
  d_nextFront = gpu.to_gpu(nextFront.reshape(-1))
  d_scan = gpu.to_gpu(scan.reshape(-1))
  d_seed_threshold = gpu.to_gpu(np.float32(seed_threshold))
  d_threshold = gpu.to_gpu(np.float32(threshold))
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  # Find any seed points in the image and add them to the region
  start_gpu_time.record() # start recording time
  # call regionGrow kernel with seed_threshold and a queue of indices from 0 to the size of the image
  regionGrow_kernel(d_image,d_region,d_queue,d_nextFront,width,height,qLen,d_seed_threshold,block=blocksize, grid=gridsize)
  # reset the queue for queue calculation
  d_queue.fill(0)
  cu.memcpy_dtod(d_scan.gpudata,d_nextFront.gpudata,d_nextFront.nbytes)
  scan_kernel(d_scan) # scan the front for queue index determination
  # call queue kernel in order to generate queue
  queue_kernel(d_nextFront, d_scan, d_queue, width, height, block=blocksize, grid=gridsize) # generate queue
  qLen = np.int32(gpu.max(d_scan).get()) # max value in scan must be the largest index of queue
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_comp_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  contComp = True
  if(qLen==0):
    contComp = False

  steps = 0;
  # While the next front has pixels we need to process 
  while contComp:
    # Region Growing
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    d_nextFront.fill(0) # set nextFront to zero
    regionGrow_kernel(d_image, d_region, d_queue, d_nextFront, width, height, qLen, d_threshold, block=blocksize, grid=gridsize)
    d_queue.fill(0)
    cu.memcpy_dtod(d_scan.gpudata,d_nextFront.gpudata,d_nextFront.nbytes)
    scan_kernel(d_scan) # scan the front for queue index determination
    queue_kernel(d_nextFront, d_scan, d_queue, width, height, block=blocksize, grid=gridsize) # generate queue
    qLen = np.int32(gpu.max(d_scan).get())
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_comp_time += start_gpu_time.time_till(end_gpu_time)*1e-3     

    if(qLen==0):
        contComp = False

    # Increment counter
    steps += 1
    
  # Copy from device to host
  start_gpu_time.record()
  h_region = d_region.get()
  h_region = h_region.reshape([height,width])
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_transfer_time += start_gpu_time.time_till(end_gpu_time)*1e-3

  print "Completed in %d steps" % steps
  print "GPU Time: %1.10f" % (gpu_transfer_time+gpu_comp_time+cpu_compute_time)
  print 'Time to transfer to/from GPU: %1.10f s' % gpu_transfer_time
  print 'Time to compute region grow on GPU: %1.10f' % gpu_comp_time

  print '%d,%d Threads, %d,%d Blocks' % (blocksize[0],blocksize[1],gridsize[0],gridsize[1])

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, h_region, cmap='gray', vmin=0.0, vmax=1.0)
