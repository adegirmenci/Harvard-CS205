# Alperen Degirmenci
# CS 205 HW 4 P3

import numpy as np

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA D2x kernel as a string.
min_kernel_source = \
'''
__global__ void min_kernel(float* d_data, float* d_min)
{
   // Allocate shared memory in the kernel call
   extern __shared__ float s_data[];
   int idx = threadIdx.x;

   // Copy data to shared memory
   s_data[idx] = d_data[idx];
   __syncthreads();

   // Make sure blockDim is divisible by 2
   int blockD = blockDim.x;
   int nThreads = blockD + (blockD%2);

   int half = nThreads / 2;

   while (half > 0 && idx < half) {
      int idx2 = idx + half;

      if(idx2 < blockD)
         if (s_data[idx] > s_data[idx2])
            s_data[idx] = s_data[idx2];
      
      if(half == 1)
         half = 0;
      else
         half = half/2 + (half%2);

      __syncthreads();
   }

   // Store the minimum value back to global memory
   if (idx == 0)
      d_min[0] = s_data[0];
}
'''
def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)


if __name__ == '__main__':
  # Compile the CUDA kernel
  min_kernel = cuda_compile(min_kernel_source,"min_kernel")

  # On the host, define the vectors, be careful about the types
  N = np.int32(1024)
  data = np.float32(np.random.random(N)) # Random array of length N
  dmin = np.float32(np.zeros(1)) # Container for min value
  smem_size = data.nbytes # Shared memory size

  # On the host, define the kernel parameters
  blocksize = (1024,1,1) # The number of threads per block (x,y,z)
  gridsize  = (1,1)   # The number of thread blocks     (x,y)

  correctResults = 0 # Initialize number of correct runs
  nTests = 1000 # Run kernel this many times
  gpu_time = 0

  for i in xrange(0,nTests):
    # Allocate device memory and copy host to device
    d_data = gpu.to_gpu(data)
    d_dmin = gpu.to_gpu(dmin)

    # Initialize the GPU event trackers for timing
    start_gpu_time = cu.Event()
    end_gpu_time = cu.Event()
    
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    min_kernel(d_data, d_dmin, block=blocksize, grid=gridsize, shared=smem_size)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_time += start_gpu_time.time_till(end_gpu_time) * 1e-3

    # Copy from device to host
    h_dmin = d_dmin.get()
    # Compute min using numpy
    np_min = np.min(data)
    # Compare data
    if np.abs(h_dmin[0] - np_min) < 1e-5:
      correctResults += 1 # Bookkeeping
    else:
      print 'Test %d failed: GPU_min: %f vs. Numpy_min: %f' % (i,h_dmin[0],np_min)
    # Generate new random data for next run
    data = np.float32(np.random.random(N))

  # Calculate average GPU runtime
  ave_gpu_time = gpu_time/nTests
  # Report GPU config and time
  print '###\n%d Threads, %d Blocks' % (blocksize[0], gridsize[0])
  print "Average GPU Time: %1.10f" % ave_gpu_time  
  # Report results
  if correctResults == nTests:
    print 'CUDA operation successful: %d of %d tests passed!\n###' % (correctResults,nTests)
  else:
    print 'CUDA operation failed! Only %d correct our of %d! Fix your code!\n###' % (correctResults,nTests)
