import numpy as np
import time

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA saxpy kernel as a string.
saxpy_kernel_source = \
"""
__global__ void saxpy_kernel(float* z, float alpha, float* x, float* y, int N)
{

}
"""
def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)


if __name__ == '__main__':
  # Compile the CUDA kernel
  saxpy_kernel = cuda_compile(saxpy_kernel_source,"saxpy_kernel")

  # generate data for bandwidth and latency determination
  n = 12
  nbytes = np.zeros(n, dtype=np.float64)
  timeTo = np.zeros(n, dtype=np.float64)
  timeFrom = np.zeros(n, dtype=np.float64)

  # On the host, define the kernel parameters
  blocksize = (256,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (1024,1)   # The number of thread blocks     (x,y)

  for p in xrange(n):
    nbytes[p] = int(8**p)
    data = np.float32(np.random.random(nbytes[p]/8.0))

    t_s1 = time.time() 
    # Allocate device memory and copy host to device
    data_d = gpu.to_gpu(data)
    t_s2 = time.time()
    # Copy from device to host
    data_gpu = data_d.get()
    t_e = time.time()
    
    timeTo[p] = t_s2 - t_s1
    timeFrom[p] = t_e - t_s2

  # Compute latency
  latTo = (sum(timeTo[:4])/4.0) * 1e6
  print '\nLatency To = %f usec' % latTo
  latFrom = (sum(timeFrom[:4])/4.0) * 1e6
  print 'Latency From = %f usec' % latFrom

  # Compute bandwidth
  MBpsTo = (((nbytes[n-1] - nbytes[n-2])/(timeTo[n-1] - timeTo[n-2]))/1e6)
  print 'Bandwidth To = %f MBPS' % MBpsTo
  MBpsFrom = (((nbytes[n-1] - nbytes[n-2])/(timeFrom[n-1] - timeFrom[n-2]))/1e6)
  print 'Bandwidth From = %f MBPS' % MBpsFrom

  print '%d Threads, %d Blocks\n' % (blocksize[0], gridsize[0])
