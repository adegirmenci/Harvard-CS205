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
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < N) { z[tid] += alpha*x[tid] + y[tid]; }
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

  # On the host, define the vectors, be careful about the types
  N      = np.int32(2**18)
  z      = np.float32(np.zeros(N))
  alpha  = np.float32(100.0)
  x      = np.float32(np.random.random(N))
  y      = np.float32(np.random.random(N))

  # On the host, define the kernel parameters
  blocksize = (256,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (1024,1)   # The number of thread blocks     (x,y)

  gpu_time = 0

  for i in xrange(0,1000):
    # Allocate device memory and copy host to device
    x_d = gpu.to_gpu(x)
    y_d = gpu.to_gpu(y)
    z_d = gpu.to_gpu(z)
    
    # Initialize the GPU event trackers for timing
    start_gpu_time = cu.Event()
    end_gpu_time = cu.Event()
    
    # Run the CUDA kernel with the appropriate inputs
    start_gpu_time.record()
    saxpy_kernel(z_d, alpha, x_d, y_d, N, block=blocksize, grid=gridsize)
    end_gpu_time.record()
    end_gpu_time.synchronize()
    gpu_time += start_gpu_time.time_till(end_gpu_time) * 1e-3

  ave_gpu_time = gpu_time/1000

  print "Average GPU Time: %1.10f" % ave_gpu_time

  # Copy from device to host
  z_gpu = z_d.get()

  # Compute the result in serial on the host
  start_serial_time = time.time()
  z_serial = alpha * x + y
  end_serial_time = time.time()
  print "Serial Time: %f" % (end_serial_time - start_serial_time)

  # Compute the error between the two
  rel_error = np.linalg.norm(z_gpu - z_serial) / np.linalg.norm(z_serial)

  print '%d Threads, %d Blocks' % (blocksize[0], gridsize[0])

  # Print error message
  if rel_error < 1.0e-5:
    print "Hello CUDA test passed with error %f" % rel_error
  else:
    print "Hello CUDA test failed with error %f" % rel_error
