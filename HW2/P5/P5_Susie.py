import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

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

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Determine n and remainder issues
  n = int(np.floor(height/size))
  rem = np.mod(height,size)
  if rank < rem: # need to account for the remainder
    n += 1

  if rank == 0:
    img = np.zeros([height,width], dtype=np.uint16) # image buffer 
  C = np.zeros([height,width], dtype=np.uint16) # local image

  comm.barrier()
  t_s = MPI.Wtime()

  for i_ in xrange(0, n):
    i = rank + i_*size # actual i in the image
    y = np.interp(i, [0,height], [minY, maxY]) # calculate y
    #if i%50 == 0: # print every so often
    #  print "Proc %d: Line %d with y = %f" % (rank, i, y)
    for j,x in enumerate(np.linspace(minX, maxX, width)):
      C[i,j] += np.uint16(mandelbrot(x,y))

  # report
  print 'Proc %d: Work complete. Ready to send data.' % rank
  # gather local images
  C = comm.gather(C, root=0)

  comm.barrier()
  t_e = MPI.Wtime()

  if rank == 0:
    for i in xrange(0, size):
      img += C[i][:,:] # add gathered images to final image
    print "Time: %f secs" % (t_e - t_s)
    plt.imsave('Mandelbrot.png', img, cmap='spectral')
    print 'Image saved'
