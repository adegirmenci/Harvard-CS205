# Alperen Degirmenci
# DATA I/O
# CS 205 Project
# 11/18/12

import numpy as np

class imgDataIO:
    """ Class for reading in MRI and US data """
    def __init__(self, data_file, dType, shape, removeData=0):
        """
        data_file: path to file
        dType: data type
        shape: shape of data [x,y]
        removeData: set whether or not to remove a chunk of data from original input to simulate corrupt/nongrided data
        """
        self.dType = dType
        self.data = np.fromfile(data_file, dtype=self.dType)
        # For some reason, the data needs to be reshaped using Fortran (column-major) order
        self.data = self.data.reshape(shape,order='F').reshape([-1,1])
        self.data = np.uint8(np.round(self.data/4095.*255))
        self.dimx = shape[0] # size of the x-dimension
        self.dimy = shape[1] # size of the y-dimension
        #self.dimz = shape[2] # size of the z-dimension
        self.size = self.dimx*self.dimy # number of elements
        self.meshX = []
        self.meshY = []
        self.boundingBox = []
        self.reso = 0.25
        self.getMesh()
        self.points = np.concatenate((self.imgX.reshape([-1,1]),self.imgY.reshape([-1,1])),axis=1)
        self.points = np.concatenate((self.points, self.data),axis=1)
        self.queryPoints = np.concatenate((self.meshX.reshape([-1,1]),self.meshY.reshape([-1,1])),axis=1)
        self.nPoints = self.size
        self.nQueries = np.size(self.queryPoints,0)
        self.queryPoints = np.concatenate((self.queryPoints, np.zeros([self.nQueries,1])),axis=1)
        self.points = np.concatenate((self.points,self.queryPoints),axis=0)
        self.points = np.float32(self.points)
        print 'File read: %d by %d matrix' % (self.dimx,self.dimy)

        self.removePoints = removeData
        # remove a strip of data
        if(self.removePoints == 1):
            self.points = np.concatenate((self.points[0:28*64,:],self.points[29*64:-1,:]),axis=0)
            self.nPoints = self.nPoints - 65

    def getMesh(self):
        [self.meshX, self.meshY] = np.mgrid[0.25:self.dimx+2:self.reso,0.25:self.dimy+2:self.reso]
        self.meshX = np.float32(self.meshX)
        self.meshY = np.float32(self.meshY)
        [self.imgX, self.imgY] = np.mgrid[1:self.dimx+1,1:self.dimy+1]
        self.imgX = np.float32(self.imgX)
        self.imgY = np.float32(self.imgY)
        self.boundingBox = np.asarray([[0,self.dimx+2],[0,self.dimy+2]])
        self.boundingBox = self.boundingBox.reshape([2,2])

class textDataIO:
    """ Class for reading in text data / only in 2D right now """
    def __init__(self, data_file, dType):
        """
        data_file: path to file
        dType: data type
        """
        self.dType = dType
        self.data = np.loadtxt(data_file, dtype=self.dType) #np.float32
        self.dimx = np.size(self.data,0) # size of the x-dimension
        self.dimy = 3 # size of the y-dimension
        #self.dimz = 1 # size of the z-dimension
        self.shape = [self.dimx,self.dimy]
        self.data = self.data.reshape(-1).reshape(self.shape)
        self.size = self.dimx*self.dimy # number of elements
        self.meshX = []
        self.meshY = []
        self.boundingBox = []
        self.reso = 0.25 # resolution of the mesh / 0.25 is good
        self.getMesh()
        self.queryPoints = np.concatenate((self.meshX.reshape([-1,1]),self.meshY.reshape([-1,1])),axis=1)
        self.nPoints = self.dimx
        self.nQueries = np.size(self.queryPoints,0)
        self.queryPoints = np.concatenate((self.queryPoints, np.zeros([self.nQueries,1])),axis=1)
        self.points = np.concatenate((self.data,self.queryPoints),axis=0)
        self.points = self.dType(self.points)
        print 'File read: %d by %d matrix' % (self.dimx,self.dimy)

    def getMesh(self):
        [self.meshX, self.meshY] = np.mgrid[0.25:25.:self.reso,0.25:25.:self.reso]
        self.meshX = self.dType(self.meshX)
        self.meshY = self.dType(self.meshY)
        self.boundingBox = np.asarray([[np.floor(np.min(self.meshX)),np.ceil(np.max(self.meshX))],[np.floor(np.min(self.meshY)),np.ceil(np.max(self.meshY))]])
        self.boundingBox = self.boundingBox.reshape([2,2])
if __name__  == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    header_file = '../data/HeadMRVolume.mhd'
    data_file = '../data/HeadMRVolume.raw'

    data_x = 48
    data_y = 62
    data_z = 42

    data = imgDataIO(data_file, dType=np.uint8, shape=[data_x,data_y,data_z])

    plt.figure(1)
    plt.imshow(data.data[:,:,0], cmap='bone')
    plt.draw()
    raw_input('wait')
