import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import Queue

def mandelbrot(x, y):
  '''Compute a Mandelbrot pixel'''
  z = c = complex(x,y)
  it, maxit = 0, 511
  while abs(z) < 2 and it < maxit:
    z = z*z + c
    it += 1
  return it

# Global variables, can be used by any process
minX,  maxX   = -2.1, 0.7
minY,  maxY   = -1.25, 1.25
width, height = 2**10, 2**10
STOP = height + 10 # Stop tag
MASTER = 0 # Master process number

def slave(comm):
  rank = comm.Get_rank()

  C = np.zeros(width, dtype=np.uint16) # buffer for MPI.Send
  y = np.zeros(1, dtype=np.float) # buffer for MPI.Read
  status = MPI.Status() # Status object
  moreComp = True # Bool keeping track of function termination condition
  while moreComp: # If process needs to perform more computations
    # Receive data from Master with any tag
    comm.Recv(y, source=MASTER, tag=MPI.ANY_TAG, status=status)
    # The row number is passed using the tag. Tag is also used to pass
    # a signal 'STOP' indicating that the process should terminate
    i = status.Get_tag()
    if i == STOP: # If tag indicates that the process should stop computation
      moreComp = False # Set termination condition
    else: # compute mandelbrot along row
      for j,x in enumerate(np.linspace(minX, maxX, width)):
        C[j] = mandelbrot(x,y[0])
      comm.Send(C, dest=MASTER, tag=i) # Send row to Master with tag as row number
  # Terminate process
  return

def master(comm):
  size = comm.Get_size()
  # Queue
  que = Queue.Queue() # keeps track of next available process
  for i in xrange(1,size): # All processes are available initially
    que.put(i)
  # Buffers
  image = np.zeros([height,width], dtype=np.uint16) # Final image
  buf = np.zeros(width, dtype=np.uint16) # Receives one row of image
  status = MPI.Status() # Status object
  jobs = enumerate(np.linspace(minY, maxY, height)) # Iterable enumerator
  jobsLeft = height # Number of jobs (rows) left to process
  while jobsLeft > 0: # while we still have jobs to compute
    while not que.empty(): # and if there are processes available
      process = que.get() # which process is available
      # Get next row number and y value. If enumerator has reached the end
      # then set i and y to the STOP tag
      i,y = next(jobs, [STOP,STOP])
      if i != STOP: # if enumerator has not reached the end
        comm.Send(y, dest=process, tag=i) # Send y and i to slave
        if i%50 == 0: # Print every so often
          print "Line %d with y = %f sent to Proc %d" %(i,y, process)
      else: # We have assigned all jobs, but we are waiting for them to complete
        print 'No more jobs, waiting for computation to finish...'

    #Receive result from Slave
    comm.Recv(buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    que.put(status.Get_source()) # add slave to available processes queue
    row = status.Get_tag() # figure out which row was Received
    image[row, :] = buf # insert into final image
    jobsLeft -= 1; # decrement the number of jobs left to process

  print 'Completed computation. Freeing slaves...'

  # Tell each slave to return
  for i in xrange(1,size):
    data = np.zeros(1, dtype=np.float) # buffer
    comm.Send(data, dest=i, tag=STOP) # send STOP tag
  
  return image


if __name__ == '__main__':
  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  if rank == 0:
    start_time = MPI.Wtime()
    C = master(comm)
    end_time = MPI.Wtime()
    print "Time: %f secs" % (end_time - start_time)
    plt.imsave('Mandelbrot.png', C, cmap='spectral')
    print 'Image saved'
    #plt.imshow(C, aspect='equal', cmap='spectral')
    #plt.show()
  else:
    slave(comm)
