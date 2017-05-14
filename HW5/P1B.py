# Alperen Degirmenci
# CS 205 HW5 P1B

import numpy as np
import Queue
import cv2
from cv2 import cv
from mpi4py import MPI

# Image files
in_file_name = "hp6_clip01.avi"
out_file_name = "hp6_carved.avi"
carveSize = 322 #pixels

DEFAULT_CODEC = cv.CV_FOURCC('P','I','M','1') # MPEG-1 codec

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

STOP = 99999
MASTER = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
device_id = pycuda.autoinit.device.pci_bus_id()
node_id = MPI.Get_processor_name()

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

def create_video_stream(source, output_filename):
  '''
  Given an input video, creates an output stream that is as similar
  to it as possible.  When no codec is detected in the input,
  default to MPG-1.
  '''
  return cv2.VideoWriter(
           filename=output_filename,
           fourcc=int(source.get(cv.CV_CAP_PROP_FOURCC)) or DEFAULT_CODEC,
           fps=source.get(cv.CV_CAP_PROP_FPS),
           frameSize=(int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH)-carveSize), # reduce width by carveSize
                      int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT))))

def process_frame_gpu(frame, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel):
  '''
  Given a source frame, processes it on the GPU using the provided kernels
  '''
  # We take our (3-channel) source frame
  height, width, channels = np.int32(frame.shape)

  # Initialize data on GPU
  d_energy = gpu.zeros(shape=(int(height),int(width)), dtype=np.float32)
  d_cmlEnergy = gpu.zeros_like(d_energy)
  d_trace = gpu.zeros(shape=int(height), dtype=np.int32)
  d_cropped = gpu.zeros(shape=(int(height),int(width)), dtype=gpu.vec.uchar3)

  # Transfer image
  d_image = gpu.to_gpu(frame.astype(np.uint8).view(gpu.vec.uchar3))
  size = width*height

  # On the host, define the kernel parameters
  blocksize = (128,8,1)     #128,8 The number of threads per block (x,y,z)
  gridx = int(np.ceil(width/(1.0*blocksize[0])))
  gridy = int(np.ceil(height/(1.0*blocksize[1])))
  gridsize  = (gridx,gridy)   # The number of thread blocks (x,y)
  # Block and grid size for the cumulative energy calculation
  blocksize_cml = (128,1,1)
  gridsize_cml = (int(np.ceil(width/(1.0*blocksize_cml[0]))),1)

  i = 0
  # Keep carving until done 
  while i < carveSize:
    # Run the CUDA kernel with the appropriate inputs
    energy_kernel(d_image, d_energy, width, height, np.int32(i), block=blocksize, grid=gridsize)
    for row in xrange(0,height):
      cmlEnergy_kernel(d_energy, d_cmlEnergy, width, height, np.int32(row), np.int32(i), block=blocksize_cml, grid=gridsize_cml)
    for row in reversed(xrange(0,height)):
      min_kernel(d_cmlEnergy, d_trace, width, height, np.int32(row), np.int32(i), block=(1,1,1), grid=(1,1))
    crop_kernel(d_image, d_cropped, d_trace, width, height, np.int32(i), block=blocksize, grid=gridsize)
    d_image, d_cropped = d_cropped, d_image # cropped image becomes the input for the next iteration
    d_cropped.fill(np.uint8(0)) # reset to zero
    # Increment counter
    i += 1
  # convert from uchar3 to 3-channel uint8 and get rid of the carved region
  return d_image.get().view(np.uint8).reshape([height,width,channels])[:,0:width-carveSize,:] 

def master(source, destination, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel, start_frame=0):
  # Seek our starting frame, if any specified
  if start_frame > 0:
    source.set(cv.CV_CAP_PROP_POS_FRAMES, start_frame)
    # Some video containers don't support precise frame seeking;
    # if this is the case, we bail.
    assert source.get(cv.CV_CAP_PROP_POS_FRAMES) == start_frame

  nFrames = source.get(cv.CV_CAP_PROP_FRAME_COUNT)
  #STOP = nFrames + 10
  height = source.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
  width = source.get(cv.CV_CAP_PROP_FRAME_WIDTH)
  print 'MASTER: Processing %d frames' % nFrames
  jobsLeft = nFrames - start_frame # this many frames will be carved
  # Queue
  que = Queue.Queue() # queue of available processes
  for i in xrange(1,size):
    que.put(i) # add processes to queue

  # Buffers
  status = MPI.Status() # Status object
  fIdx = np.zeros(1, dtype=np.int64) # send buffer for image frames
  img = np.zeros(height*(width-carveSize)*3, dtype=np.uint8) # recv buffer
  video = np.zeros([height,width-carveSize,jobsLeft*3], dtype=np.uint8) # video container

  j = start_frame # current frame index
  while jobsLeft > 0:
    while not que.empty(): # have available processes
      process = que.get() # get process id
      if source.grab(): # get frame
        fIdx[0] = np.int64(j)
        comm.Send(fIdx, dest=process, tag=j) # pass frame idx to slave
        if j%10 == 0:
          print '%d frames assigned' % j
        j += 1 # increment frame idx
      else:
        print 'All frames have been assigned to slaves'

    # Receive results from Slave
    comm.Recv(img, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    que.put(status.Get_source()) # process is now available
    frameIdx = status.Get_tag() # figure out which video frame this is
    video[:,:,(frameIdx-start_frame)*3:(frameIdx-start_frame+1)*3] = img.reshape([height,width-carveSize,3])
    jobsLeft -= 1 # decrement number of frames left to process
    
  # Tell each slave to terminate
  for k in xrange(1,size):
    fIdx[0] = STOP
    comm.Send(fIdx, dest=k, tag=STOP)

  # Write video to file
  for k in xrange(0,np.size(video,2)/3):
    destination.write(video[:,:,k*3:(k+1)*3])

  return

def slave(source, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel):
  status = MPI.Status()
  moreComp = True # bool for keeping track of termination
  buf = np.zeros(1, dtype=np.uint64) # recv buffer
  while moreComp:
    # Recv data from master
    comm.Recv(buf, source=MASTER, tag=MPI.ANY_TAG, status=status)
    frameIdx = status.Get_tag() # which frame is this
    if frameIdx == STOP: # is this the STOP signal
      moreComp = False # yes, then terminate
    else:
      source.set(cv.CV_CAP_PROP_POS_FRAMES, frameIdx) # go to indicated frame
      # Some video containers don't support precise frame seeking;
      # if this is the case, we bail.
      assert source.get(cv.CV_CAP_PROP_POS_FRAMES) == frameIdx
      source.grab() # grab frame
      _, frame = source.retrieve() # retreive frame data
      img = process_frame_gpu(frame, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel)
      comm.Send(img.reshape(-1), dest=MASTER, tag=frameIdx) # send to MASTER

  return

if __name__ == '__main__':
  if(rank == MASTER):
    t_s = MPI.Wtime()
  # Video source and output destination
  destination, source = None, None

  try:
    # Open our source video and create an output stream
    source = cv2.VideoCapture(in_file_name)
    if(rank == MASTER):
      destination = create_video_stream(source, out_file_name)
      print "MASTER: Processing %d x %d image" % (source.get(cv.CV_CAP_PROP_FRAME_WIDTH), source.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

    # Compile the CUDA kernel
    energy_kernel = cuda_compile(energy_kernel_source,"energy_kernel")
    cmlEnergy_kernel = cuda_compile(cmlEnergy_kernel_source,"cmlEnergy_kernel")
    min_kernel = cuda_compile(min_kernel_source,"min_kernel")
    crop_kernel = cuda_compile(crop_kernel_source,"crop_kernel")
  
    if(rank == MASTER):
      master(source, destination, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel, 0)
    else:
      slave(source, energy_kernel, cmlEnergy_kernel, min_kernel, crop_kernel)
  finally:
    # Clean up after ourselves.
    if destination: del destination
    if source: source.release()
  
  if(rank == MASTER):
    t_e = MPI.Wtime()
    print "Total Time: %10.2f s" % (t_e - t_s)

