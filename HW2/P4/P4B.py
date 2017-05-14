from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import math
plt.ion()

class data_transformer:
    '''A class to transform a line into a back-projected image'''
    def __init__(self, sample_size, image_size):
        '''Perform the required precomputation for the back-projection step'''
        [self.X,self.Y] = np.meshgrid(np.linspace(-1,1,image_size),
                                      np.linspace(-1,1,image_size))
        self.proj_domain = np.linspace(-1,1,sample_size)
        self.f_scale = abs(np.fft.fftshift(np.linspace(-1,1,sample_size+1)[0:-1]))
        
    def transform(self, data, phi):
        '''Transform a data line taken at an angle phi to its back-projected image'''
        # Compute the Fourier filtered data
        filtered_data = np.fft.ifft(np.fft.fft(data) * self.f_scale).real
        # Interpolate the data to the rotated image domain
        result = np.interp(self.X*np.cos(phi) + self.Y*np.sin(phi),
                           self.proj_domain, filtered_data)
        return result

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print 'Proc %d: Started program with %d processes' %(rank, size)

    # Metadata
    n_phi = 2048
    sample_size = 6144
    # Size of the final image
    ImageSize = 2048
    
    # Read data
    if rank == 0:
        # Read the projective data from file
        print 'Proc %d: Reading data...' % rank
        data = np.fromfile(file='TomoData.bin', dtype=np.float64)
        data = data.reshape(n_phi, sample_size)
        print 'Proc %d: Data read.' % rank
        # Plot raw data
        #plt.figure(1);
        #plt.imshow(data, cmap='bone');
        #plt.draw();
    else:
        data = None

    # Allocate buffers
    data_ = np.zeros(sample_size, dtype=np.float64)    
    result = np.zeros((ImageSize, ImageSize), dtype=np.float64)
    result_ = np.zeros((ImageSize, ImageSize), dtype=np.float64)
    
    # Figure out the size of each chunk of data
    n = n_phi/size # assuming n_phi and size are powers of 2

    # Initialize Transformer
    Transformer = data_transformer(sample_size, ImageSize)

    # Communicate data and compute back-projection
    comm.barrier()
    p_start = MPI.Wtime()

    for k in xrange(0, n):
        # Scatter data
        if rank == 0:
            comm.Scatter(data[k*size:(k+1)*size,:], data_, root=0)
        else:
            comm.Scatter(data, data_, root=0)
        phi = -(k*size + rank)*math.pi/n_phi
        # Compute and update local result
        result += Transformer.transform(data_, phi)
        if k%64 == 0:
            print 'Proc %d:' % rank, k, phi

    # Sum local results
    comm.Reduce(result, result_, root=0)
    
    comm.barrier()
    p_stop = MPI.Wtime()
    # stop timing

    if rank == 0:
        result_
        # plot/save the final result
        plt.figure(2)
        plt.imshow(result_, cmap=plt.cm.bone)
        plt.draw()
        imName = 'TomoRecB' + str(ImageSize) + '.png'
        plt.imsave(imName, result_, cmap='bone')
        print 'Results saved to:', imName
        # Report time
        print 'Time: %f secs\n' % (p_stop - p_start)
