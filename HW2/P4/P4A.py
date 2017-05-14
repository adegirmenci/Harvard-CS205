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
    ImageSize = 512
    
    # Read data
    if rank == 0:
        # Read the projective data from file
        print 'Proc %d: Reading data...' % rank
        data = np.fromfile(file='TomoData.bin', dtype=np.float64)
        data = data.reshape(n_phi, sample_size)
        # Plot raw data
        #plt.figure(1);
        #plt.imshow(data, cmap='bone');
        #plt.draw();
    else:
        data = None
    
    result = np.zeros((ImageSize, ImageSize), dtype=np.float64)
    
    # Figure out the size of each chunk of data
    n = n_phi/size
    begin = n*rank
    end = n*(rank+1)

    Transformer = data_transformer(sample_size, ImageSize)
    
    # Communicate data and compute back-projection
    comm.barrier()
    p_start = MPI.Wtime()
    
    if rank == 0:
        for k in xrange(1,size): # to other processes
            # send every process its respective chunk of data
            comm.Send(data[n*k:n*(k+1)], dest=k)
    else:
        # allocate buffer
        data = np.zeros((n, sample_size), dtype=np.float64)
        # Receive data from root
        comm.Recv(data, source=0)

    for k in xrange(0, n):
        phi = -(k+n*rank)*math.pi/n_phi
        result += Transformer.transform(data[k,:], phi)
        if k%64 == 0:
            print 'Proc %d:' % rank, k, phi

    # Receive local results from processes
    if rank == 0:
        # Allocate buffer
        res = np.zeros((ImageSize, ImageSize), dtype=np.float64)
        for k in xrange(1, size): # For every process other than itself
            comm.Recv(res, source=k)
            result += res # Update result
    else:
        comm.Send(result, dest=0) # Send to root
    
    comm.barrier()
    p_stop = MPI.Wtime()
    # stop timing

    if rank == 0:
        # plot/save the final result
        plt.figure(2)
        plt.imshow(result, cmap=plt.cm.bone)
        plt.draw()
        imName = 'TomoRecA' + str(ImageSize) + '.png'
        plt.imsave(imName, result, cmap='bone')
        print 'Results saved to', imName
        # Report time
        print 'Time: %f secs\n' % (p_stop - p_start)
