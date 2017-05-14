# Alperen Degirmenci
# CS 205 HW 4 P2

import numpy as np
import time
import matplotlib.pyplot as plt

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA saxpy kernel as a string.
D2x_kernel_source = \
"""
__global__ void D2x_kernel(double* z, const double* a, const int N, const double dx)
{
   //Local thread id
   int tid = threadIdx.x;
   //Global thread id
   int gid = blockDim.x * blockIdx.x + tid;
   //Number of threads per block
   int nThread = blockDim.x;

   //Allocate a shared array and read in values
   extern __shared__ double s_a[];
   s_a[tid] = a[gid]; // Read data into shared memory
   __syncthreads(); // Sync to make sure all data is read

   double temp = 0.0;
   if ((gid > 0) && (gid < N-1)){
      if((tid > 0) && (tid < (nThread-1))){
         temp = (s_a[tid-1] - 2*s_a[tid] + s_a[tid+1])/(dx*dx);}
      else if(tid==0){
         temp = (a[gid-1] - 2*s_a[tid] + s_a[tid+1])/(dx*dx);}
      else if(tid == (nThread-1)){
         temp = (s_a[tid-1] - 2*s_a[tid] + a[gid+1])/(dx*dx);}
   }
   __syncthreads();
   if((gid > 0) && (gid < N-1)){
      z[gid] = temp;
   }
}
"""
def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

if __name__ == '__main__':
  # Compile the CUDA kernel
  D2x_kernel = cuda_compile(D2x_kernel_source,"D2x_kernel")

  # On the host, define the variables
  N  = np.int32(1048576)
  x  = np.linspace(0,1,2**20)
  dx = np.float64(1.0/(N-1.0))
  sinx = np.float64(np.sin(x))
  z = sinx

  # On the host, define the kernel parameters
  blocksize = (256,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (4096,1)   # The number of thread blocks     (x,y)
  floatSize = np.float64(0).nbytes
  smem_size = floatSize*blocksize[0]
  gpu_time = 0

  for i in xrange(0,1000):
    if (i%50) == 0:
      print 'Run %d of 999' % i
    # Allocate device memory and copy host to device
    d_a = gpu.to_gpu(sinx)
    d_z = gpu.to_gpu(z)

    # Initialize the GPU event trackers for timing
    start_gpu_time = cu.Event()
    end_gpu_time = cu.Event()
    
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    D2x_kernel(d_z, d_a, N, dx, block=blocksize, grid=gridsize, shared=smem_size)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_time += start_gpu_time.time_till(end_gpu_time) * 1e-3

  ave_gpu_time = gpu_time/1000

  print "Average GPU Time: %1.10f" % ave_gpu_time

  # Copy from device to host
  z_gpu = d_z.get()
  
  negsin = -1.0*np.sin(x)
  # Compute the error between the two
  rel_error = np.linalg.norm(z_gpu[1:-1] - negsin[1:-1]) / np.linalg.norm(negsin[1:-1])

  print '%d Threads, %d Blocks' % (blocksize[0], gridsize[0])

  # Print error message
  if rel_error < 5.0e-3:
    print "Hello CUDA test passed with error %f" % rel_error
  else:
    print "Hello CUDA test failed with error %f" % rel_error

  plt.figure(1)
  plt.plot(x[1:-1],z_gpu[1:-1],color='r',label='Computed Value')
  plt.plot(x[1:-1],negsin[1:-1],color='b',label='True Value')
  plt.legend()
  plt.show()
