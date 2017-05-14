from mpi4py import MPI
import numpy as np
import time

def get_size():
  '''The array size to use for P2'''
  return 12582912     # A big number

def get_big_arrays():
  '''Generate a big array rather than reading from file'''
  np.random.seed(0)  # Set the random seed
  return np.random.random(get_size()), np.random.random(get_size())

def serial_dot(a, b):
  '''The dot-product of the arrays'''
  result = 0
  for k in xrange(0, len(a)):
    result += a[k]*b[k]
  return result

def parallel_dot(a, b, comm, p_root=0):
  '''The parallel dot-product of the arrays.
  Assumes the arrays exist on process p_root
  and returns the result to process p_root.
  By default, p_root = process 0.'''
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Broadcast the arrays
  a = comm.bcast(a, root=p_root)
  b = comm.bcast(b, root=p_root)

  # Size of each process's local dot product
  n = np.floor(len(a) / size) # Largest chunk divisible by the number of procs
  # The work amount might not be divisible by the number of processes
  # Compute the remainder
  rem = len(a) - n*size
  # Start and end indices of the local dot product
  # Balancing is handled by taking the min of current rank and the size of the remainder
  start = n * rank + min(rank, rem)
  end   = n * rank + n + min(rank+1, rem)

  # Compute the partial dot product
  local_dot = serial_dot(a[start:end], b[start:end])

  # Reduce the partial results to the root process
  result = comm.reduce(local_dot, root=p_root)
  return result


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Get big arrays on process 0
  if rank == 0:
    a, b = get_big_arrays()
  else:
    a, b = None, None

  # Compute the dot product in parallel
  comm.barrier()
  p_start = MPI.Wtime()
  result = parallel_dot(a, b, comm)
  comm.barrier()
  p_stop = MPI.Wtime()

  # Check and output results on process 0
  if rank == 0:
    s_start = time.time()
    s_dot = serial_dot(a, b)
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)
    rel_error = abs(result - s_dot) / abs(s_dot)
    print "Parallel Result = %f" % result
    print "Serial Result   = %f" % s_dot
    print "Relative Error  = %e" % rel_error
    if rel_error > 1e-10:
      print "***LARGE ERROR - POSSIBLE FAILURE!***"
