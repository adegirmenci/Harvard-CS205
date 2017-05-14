# Alperen Degirmenci
# CS 205 Project
# Jump Flooding Algorithm Implementation

import numpy as np
import cv2
from cv2 import cv

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit
# Import dataIO tools
import dataIO

# Define the CUDA sharpening kernel as a string
discretizePoints_kernel_source = \
'''
/**
* Discretizes a given set of points and sets these points as the seeds in the jump flooding map.
* points: Given points. .x and .y are the coordinates, .z holds the data (0-255). This also contains the query grid positions, which are concatenated to the end of the given points. 
* discPoints: Discretized points. .x and .y are coordinates, .z holds the data.
* floodMap: Jump flooding map for the given data points.
* queryMap: Jump flooding map for the query points.
* nPoints: Number of given data points.
* nQueries: Number of query points.
* mapRes: Resolution of the discretization grid (1 unit in data corresponds to 1/mapRes units in the Voronoi diagram)
* mapLength: Width (and height) of the jump flooding map. We are assuming a square map.
* The position of x_0, y_0. In general, this should be 0, but just in case.
* The reason we have a queryMap is because we will compute the Voronoi diagram of the query points.
* This is necessary because we do not assume the user always wants to use a grid for query (in other words,
* the user might not want to go from non-gridded data to gridded data). This allows us to go from non-gridded to non-gridded data.
* For speed-up purposes, this can be eliminated. This will reduce the computation time by a factor of ~2.
*/
__global__ void discretizePoints_kernel(const float3 *points, int3 *discPoints, int *floodMap, int *queryMap, const int nPoints, const int nQueries, const float mapRes, const int mapLength, const int gridMin)
{
   int idx = blockIdx.x*blockDim.x + threadIdx.x; //thread index

   if(idx < (nPoints+nQueries)) // thread index less than number of points in the points vector
   {
      float3 point = points[idx]; // get the point at location idx
      int3 newLoc; // discretized location
      newLoc.x = (int)roundf(point.x/mapRes); // scale by map resolution and round to nearest integer
      newLoc.y = (int)roundf(point.y/mapRes);
      newLoc.z = (int)roundf(point.z); // data value rounded to neares integer

      int newI_x = newLoc.x - gridMin; // account for offset, if there is any
      int newI_y = newLoc.y - gridMin;
      int newIdx = newI_y*mapLength + newI_x; // new index of the point on the flood map

      discPoints[idx] = newLoc; // add descretized point to discrete points vector
      if(idx < nPoints) // if our point belongs to the original data set
         floodMap[newIdx] = idx; // assign the thread index (which is the index of the point in the points array) as its ID in the floodMap
      else // if our points belongs to the query points
         queryMap[newIdx] = idx; // add the query point to the query map with ID = idx
   }
   __syncthreads();
}
'''
floodFill_kernel_source = \
'''
/**
* Calculates the Euclidean distance between two points (square root is not taken in order to preserve precision)
* xDist: distance between two points in the x-direction
* yDist: distance between two points in the y-direction
*/
__device__ int distanceCalc(int xDist, int yDist)
{  
   return (xDist*xDist + yDist*yDist); // Euclidean Distance
   //return (abs(xDist) + abs(yDist)); // Manhattan Distance
}

/**
* Calculates the Euclidean distance between two points
* Helps us determine which Voronoi site a point belongs to
* If the current owner (Voronoi site) of the point is not the one that is closest to it, this function will set the owner to the one that is closer
* x,y : coordinates of a point
* x_dir, y_dir : indices of a second point that is step size away from the first point
* mapLength: width of the flood map
* floodMap: the flood map
* discPoints: discretized points
* owner: the owner of the Voronoi site that the first point belongs to
* distance: the distance between the first point and the center of the Voronoi site this first point belongs to
*/
__device__ void distanceCalc2(const int x, const int y, const int x_dir, const int y_dir, const int mapLength, const int **floodMap, const int3 **discPoints, int *owner, int *distance)
{
   int idx_dir = y_dir*mapLength + x_dir;  // index of the second point in the vector map array       
   int stepOwner = (*floodMap)[idx_dir]; // the ID of the Voronoi site that the second point belongs to

   if(stepOwner != -1) // if the second point belongs to a Voronoi site
   {
      int3 point = (*discPoints)[stepOwner]; // get the location of the center of this Voronoi site
      int xDist = x - point.x; // x-distance from the first point to the center of this Voronoi site
      int yDist = point.y - y; // y-distance from the first point to the center of this Voronoi site
      int dist_owner = distanceCalc(xDist,yDist); // calculate the Euclidean distance to the center
      if((dist_owner < *distance) || (*distance == -1)) // If the new Voronoi site is closer to the first point, or if the first point didn't have a Voronoi site assigned to it
         { *owner = stepOwner; *distance = dist_owner; } // update owner and distance
   } // otherwise, we don't need to do anything
}

/**
* Performs jump flooding on a given flood map which contains seed(s)
* floodMap: flood map containing seeds (non-seeded locations are set to -1) (seeds have an ID corresponding to their location in the discPoints array)
* newMap: temporary flood map holding the updated floodMap
* discPints: discretized points
* mapLength: width of the flood map
* gridMin: grid offset
* stepSize: step size for the current jump flooding iteration
*/
__global__ void floodFill_kernel(const int *floodMap, int *newMap, const int3 *discPoints, const int mapLength, const int gridMin, const int stepSize)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x; // thread index in the x-direction
   int i_y = blockIdx.y*blockDim.y + threadIdx.y; // thread index in the y-direction

   if((i_x < mapLength) && (i_y < mapLength)) // make sure the thread is inside the map bounds
   {
      int idx = i_y*mapLength + i_x; // global thread index

      //check in 8 directions (north,ne,east,se,south,sw,west,nw)
      int x_north = i_x;
      int y_north = i_y - stepSize;
      int x_ne = i_x + stepSize;
      int y_ne = i_y - stepSize;
      int x_east = i_x + stepSize;
      int y_east = i_y;
      int x_se = i_x + stepSize;
      int y_se = i_y + stepSize;
      int x_south = i_x;
      int y_south = i_y + stepSize;
      int x_sw = i_x - stepSize;
      int y_sw = i_y + stepSize;
      int x_west = i_x - stepSize;
      int y_west = i_y;
      int x_nw = i_x - stepSize;
      int y_nw = i_y - stepSize;

      int owner = floodMap[idx]; // to which Voronoi site the current pixel belongs to
      int distance; // distance to the center of the Voronoi site
      if(owner == -1) // don't belong to any Voronoi site
         distance = -1; // infinite distance
      else // do belong to a site
      {
         int3 point = discPoints[owner]; // get the center of the Voronoi site we belong to
         int xDist = i_x - point.x; 
         int yDist = i_y - point.y;
         distance = distanceCalc(xDist,yDist); // calculate distance to the center of the Voronoi site
      }

      // check if we are closer to any other neighboring Voronoi sites
      if((0 < x_north) && (x_north < mapLength) && (0 < y_north) && (y_north < mapLength)) // make sure this location is inside the map boundaries
         distanceCalc2(i_x, i_y, x_north, y_north, mapLength, &floodMap, &discPoints, &owner, &distance); // check if we are closer 
      if((0 < x_ne) && (x_ne < mapLength) && (0 < y_ne) && (y_ne < mapLength))
         distanceCalc2(i_x, i_y, x_ne, y_ne, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_east) && (x_east < mapLength) && (0 < y_east) && (y_east < mapLength))
         distanceCalc2(i_x, i_y, x_east, y_east, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_se) && (x_se < mapLength) && (0 < y_se) && (y_se < mapLength))
         distanceCalc2(i_x, i_y, x_se, y_se, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_south) && (x_south < mapLength) && (0 < y_south) && (y_south < mapLength))
         distanceCalc2(i_x, i_y, x_south, y_south, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_sw) && (x_sw < mapLength) && (0 < y_sw) && (y_sw < mapLength))
         distanceCalc2(i_x, i_y, x_sw, y_sw, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_west) && (x_west < mapLength) && (0 < y_west) && (y_west < mapLength))
         distanceCalc2(i_x, i_y, x_west, y_west, mapLength, &floodMap, &discPoints, &owner, &distance);
      if((0 < x_nw) && (x_nw < mapLength) && (0 < y_nw) && (y_nw < mapLength))
         distanceCalc2(i_x, i_y, x_nw, y_nw, mapLength, &floodMap, &discPoints, &owner, &distance);

      newMap[idx] = owner; // write the owner to the flood map 
   }
   __syncthreads();
}
'''
island_removal_kernel_source = \
'''
/**
* Calculates the Euclidean distance between two points (square root is not taken in order to preserve precision)
* xDist: distance between two points in the x-direction
* yDist: distance between two points in the y-direction
*/
__device__ int distanceCalc(int xDist, int yDist)
{  
   return (xDist*xDist + yDist*yDist); // Euclidean Distance
}
/**
* Detects any islands (cells in a Voronoi cell that are labeled as belonging to another cell,
* but actually belong to the current cell) and fix them
* floodMap: flood map containing the Voronoi sites
* newMap: temporary flood map holding the updated floodMap
* discPoints: discretized points
* mapLength: width of flood map
* gridMin: grid offset
*/
__global__ void island_removal_kernel(const int *floodMap, int *newMap, const int3 *discPoints, const int mapLength, const int gridMin)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   if((i_x < mapLength) && (i_y < mapLength)) // within bounds
   {
      int idx = i_y*mapLength + i_x; // global thread index

      int owner = floodMap[idx]; // Voronoi site that owns the current pixel
      int3 ownerLoc = discPoints[owner]; // center of the Voronoi site
      int owner_x = ownerLoc.x; // x-position of the center
      int owner_y = ownerLoc.y; // y-position of the center
      int dir_x = owner_x - i_x; // vector in the x-direction from current pixel to the Voronoi center
      int dir_y = i_y - owner_y; // vector in the y-direction from current pixel to the Voronoi center
      int quad = 99; // which quadrant the center lies in with respect to the current pixel
      int3 neighbors; //3 neighbors, around the pixel
      if(dir_x > 0) // center is to our right
      {
         if(dir_y > 0) { quad = 2; neighbors.x = idx-mapLength; neighbors.y = neighbors.x+1; neighbors.z = idx+1; }
         else if(dir_y == 0) { quad = 1; neighbors.x = idx-mapLength+1; neighbors.y = idx+1; neighbors.z = idx+mapLength+1; }
         else { quad = 8; neighbors.x = idx+1; neighbors.y = idx+mapLength+1; neighbors.z = neighbors.y-1; }
      }
      else if(dir_x == 0)
      {
         if(dir_y > 0){ quad = 3; neighbors.x = idx-mapLength-1; neighbors.y = neighbors.x+1; neighbors.z = neighbors.y+1; }
         else if(dir_y == 0){ quad = 0; neighbors.x = idx; neighbors.y = idx; neighbors.z = idx; }
         else{ quad = 7; neighbors.x = idx+mapLength+1; neighbors.y = neighbors.x-1; neighbors.z = neighbors.y-1; }
      }
      else // center is to our left
      {
         if(dir_y > 0){ quad = 4; neighbors.x = idx-1; neighbors.y = idx-mapLength-1; neighbors.z = neighbors.y+1; }
         else if(dir_y == 0){ quad = 5; neighbors.x = idx+mapLength-1; neighbors.y = idx-1; neighbors.z = idx-mapLength-1; }
         else{ quad = 6; neighbors.x = idx+mapLength; neighbors.y = neighbors.x-1; neighbors.z = idx-1; }
      }

      //check neighbors
      int idxLimit = mapLength*mapLength;
      int owner1 = owner; int owner2 = owner; int owner3 = owner;
      if(neighbors.x < idxLimit){owner1 = floodMap[neighbors.x];}
      if(neighbors.y < idxLimit){owner2 = floodMap[neighbors.y];}
      if(neighbors.z < idxLimit){owner3 = floodMap[neighbors.z];}
      
      if(owner != owner1 || owner != owner2 || owner != owner3)
      {
         int3 o1loc = discPoints[owner1];
         int3 o2loc = discPoints[owner2];
         int3 o3loc = discPoints[owner3];

         int dist0 = distanceCalc(dir_x,dir_y);
         int dist1 = distanceCalc(i_x-o1loc.x, i_y+o1loc.y);
         int dist2 = distanceCalc(i_x-o2loc.x, i_y+o2loc.y);
         int dist3 = distanceCalc(i_x-o3loc.x, i_y+o3loc.y);

         int minOwner = owner;
         int minDist = dist0;
         if(dist1 <= minDist){ minOwner = owner1; minDist = dist1;}
         if(dist2 <= minDist){ minOwner = owner2; minDist = dist2;}
         if(dist3 <= minDist){ minOwner = owner3; minDist = dist3;}

         owner = minOwner;
      }
      newMap[idx] = owner;
   }
  __syncthreads();
}
'''
query_kernel_source= \
'''
/**
* floodMap: Voronoi diagram of the original points
* queryMap: Voronoi diagram of the query points
* discPoints: discretized points
* queryValues: the numerator (sum of stolen area*stolen value) and the denominator (total stolen area)
* mapLength: width of the flood map
* nPoints: number of original data points
*/
__global__ void query_kernel(const int *floodMap, const int *queryMap, const int3 *discPoints, float2 *queryValues, const int mapLength, const int nPoints)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   if((i_x < mapLength) && (i_y < mapLength))
   {
      int idx = i_y*mapLength + i_x; // global thread index
      int floodOwner = floodMap[idx]; // owner of the current pixel in the Voronoi diagram of the original data
      int queryOwner = queryMap[idx]; // owner of the current pixel in the Voronoi diagram of the query points
      
      if(floodOwner != queryOwner) // if the Voronoi sites that owns the current pixel are different
      {
         int queryIdx = queryOwner - nPoints; // figure out which query point this is
         if(queryOwner >= nPoints) // sanity check, this should always be true
         {
            int fOwnerVal = discPoints[floodOwner].z; // stolen data
            atomicAdd(&(queryValues[queryIdx].x), (float)fOwnerVal); // add the value of the owner to the numerator, this is equivalent to adding the stolen area times stolen value
            atomicAdd(&(queryValues[queryIdx].y), 1.0); // add to total area
         }
      }
   }
   __syncthreads();
}
'''
determineColors_kernel_source= \
'''
/**
* Create a 3-channel color map for all the points depending on their value
* colors: 3-channel color map
* discPoints: discretized points (including the query points)
* queryValues: numerator and denominator
* nPoints: number of original data points
* nQueries: number of query points
*/
__global__ void determineColors_kernel(uchar3 *colors, const int3 *discPoints, const float2 *queryValues, const int nPoints, const int nQueries)
{
   int idx = blockIdx.x*blockDim.x + threadIdx.x;

   if(idx < (nPoints + nQueries)) // within bounds
   {
      uchar3 color; // new color
      if(idx < nPoints) // this is one of the original points
      {
         unsigned char val = (unsigned char)discPoints[idx].z;
         color.x = val; // all three channels get the original value
         color.y = val;
         color.z = val;
      }
      else
      {
         float2 q = queryValues[idx-nPoints]; // get num and denom
         int val = roundf(q.x/q.y); // calculate the stolen value
         if(val > 255) // cutoff at 255
            val = 255;
         color.x = (unsigned char)val; // all three channels get this value
         color.y = (unsigned char)val;
         color.z = (unsigned char)val;
      }
      colors[idx] = color; // add color to array
   }
   __syncthreads();
}
'''
voronoi_kernel_source = \
'''
/**
* Render Voronoi diagram
* floodMap: Voronoi map (can be floodMap or queryMap)
* voronoi: Image
* colors: colors for each site
* mapLength: width of the map
* gridMin: grid offset
*/
__global__ void voronoi_kernel(const int *floodMap, uchar3 *voronoi, const uchar3 *colors, const int mapLength, const int gridMin)
{
   int i_x = blockIdx.x*blockDim.x + threadIdx.x;
   int i_y = blockIdx.y*blockDim.y + threadIdx.y;

   if((i_x < mapLength) && (i_y < mapLength))
   {
      int idx = i_y*mapLength + i_x;

      int owner = floodMap[idx]; // owner of the current pixel
      uchar3 color; // new color
      if(owner != -1) // belongs to a Voronoi site
         color = colors[owner]; // gets the corresponding color
      else // doesn't belong to a site (should NEVER happen, but just in case)
      {
         color.x = 150;
         color.y = 151;
         color.z = 152;
      }
      voronoi[idx] = color; // add to image
   }
  __syncthreads();
}
'''

class JFA:
    """ Class for creating Voronoi Diagrams using Jump Flooding """
    def __init__(self, data, npoints, nqueries, mapres, mapbounds):
        """
        points: data points and query points
        nPoints: number of data points
        nQueries: number of query points
        mapRes: resolution of the flood map
        mapBounds: bounds of the flood map (assumes square map)
        """
        self.points = np.float32(data)
        self.nPoints = npoints
        self.nQueries = nqueries
        self.mapRes = mapres # size of each grid = 1/mapRes
        self.mapBounds = mapbounds # the min and max of the grid
        self.mapLength = int(np.ceil((self.mapBounds[1] - self.mapBounds[0])/self.mapRes)) # width of map, assuming square grid
        self.mapSize = int(self.mapLength*self.mapLength) # number of elements in the map
        self.totalPoints = self.nPoints+self.nQueries # total number of points given

        self.compile() # compile CUDA code
        self.initializeTimer() # initialize CUDA timers
        self.gpuParams() # set block and grid size
        self.dataToGPU() # send data to GPU
        self.discretizePoints() # discretize data points and initialize seeds
        self.floodFill() # floodFill of initial data
        self.swapFloodMaps() # swap floodMap and queryMap
        self.floodFill() # floodFill of queryMap
        self.swapFloodMaps() # swap floodMap and queryMap
        self.query() # interpolate the data
        self.determineColors() # determine color map
        self.voronoiImage(choice=1) #0: compute base image, 1: compute queried image
        self.dataToHost() # transfer data to host
        self.reportBenchmark() # report computation time
        #self.saveImage() user will call this

    def cuda_compile(self, source_string, function_name):
      # Compile the CUDA Kernel at runtime
      source_module = nvcc.SourceModule(source_string)
      # Return a handle to the compiled CUDA kernel and the texture
      return source_module.get_function(function_name)#, source_module.get_texref('tex')

    def compile(self):
      # Compile the CUDA kernel
      self.discretizePoints_kernel = self.cuda_compile(discretizePoints_kernel_source,"discretizePoints_kernel") #, texref
      self.floodFill_kernel = self.cuda_compile(floodFill_kernel_source,"floodFill_kernel")
      self.island_removal_kernel = self.cuda_compile(island_removal_kernel_source,"island_removal_kernel")
      self.query_kernel = self.cuda_compile(query_kernel_source,"query_kernel")
      self.determineColors_kernel = self.cuda_compile(determineColors_kernel_source,"determineColors_kernel")
      self.voronoi_kernel = self.cuda_compile(voronoi_kernel_source,"voronoi_kernel")

    def initializeTimer(self):
      # Initialize the GPU event trackers for timing
      self.start_gpu_time = cu.Event()
      self.end_gpu_time = cu.Event()
      self.gpu_transfer_time = 0.0
      self.gpu_compute_time = 0.0
      
    def gpuParams(self):
      # On the host, define the kernel parameters
      self.blocksize = (64,8,1) #64,8 The number of threads per block (x,y,z)
      gridx = int(np.ceil(self.mapLength/(1.0*self.blocksize[0])))
      gridy = int(np.ceil(self.mapLength/(1.0*self.blocksize[1])))
      self.gridsize  = (gridx,gridy)   # The number of thread blocks (x,y)
      # Kernel parameters for discretizePoints_kernel
      if(self.totalPoints < 1024): # data fits in one block
          self.blocksize_dP = (int(self.totalPoints),1,1)
          self.gridsize_dP = (1,1)
      else: # doesn't fit in one block
          self.blocksize_dP = (512,1,1)
          self.gridsize_dP = (int(np.ceil(self.totalPoints/512.)),1)

    def dataToGPU(self):
      # Allocate device memory and copy host to device
      self.start_gpu_time.record()
      self.d_points = gpu.to_gpu(self.points.reshape(-1).view(gpu.vec.float3))
      self.d_discPoints = gpu.zeros(shape=int(self.nPoints+self.nQueries), dtype=gpu.vec.int3) # dicretized location of points
      self.d_floodMap = gpu.empty(shape=int(self.mapLength*self.mapLength), dtype=np.int32) # 1+JFA map
      self.d_floodMap.fill(np.int32(-1)) # initialize to -1
      self.d_tempMap = gpu.zeros_like(self.d_floodMap) # swap memory for floodMap
      self.d_tempMap.fill(np.int32(-1)) # initialize to -1
      self.d_queryMap = gpu.zeros_like(self.d_floodMap) # query (interpolation) map
      self.d_queryMap.fill(np.int32(-1)) # initialize to -1
      self.d_queryValues = gpu.zeros(shape=int(self.nQueries), dtype=gpu.vec.float2) # for calculating stolen area
      self.d_colors = gpu.zeros(shape=int(self.nPoints+self.nQueries), dtype=gpu.vec.uchar3) # color map
      self.d_voronoi = gpu.zeros(shape=int(self.mapLength*self.mapLength), dtype=gpu.vec.uchar3) # rendered Voronoi image 
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_transfer_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def discretizePoints(self):
      # Run the CUDA kernel with the appropriate inputs
      print 'Calling discretizePoints kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize_dP[0],self.blocksize_dP[1],self.gridsize_dP[0],self.gridsize_dP[1])
      self.start_gpu_time.record()
      self.discretizePoints_kernel(self.d_points, self.d_discPoints, self.d_floodMap, self.d_queryMap, np.int32(self.nPoints), np.int32(self.nQueries), np.float32(self.mapRes), np.int32(self.mapLength), np.int32(self.mapBounds[0]), block=self.blocksize_dP, grid=self.gridsize_dP)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def floodFill(self):
      # Perform 1+JFA
      self.stepSize = 1
      self.floodFillStep()
      self.stepSize = self.mapLength
      while self.stepSize > 1:
        self.stepSize = np.ceil(self.stepSize/2)
        #print 'Running flood fill with step size %d' % self.stepSize
        self.floodFillStep()
      self.islandRemoval() # remove islands

    def floodFillStep(self):
      print 'Calling floodFill kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize[0],self.blocksize[1],self.gridsize[0],self.gridsize[1])
      self.start_gpu_time.record()
      self.floodFill_kernel(self.d_floodMap, self.d_tempMap, self.d_discPoints, np.int32(self.mapLength), np.int32(self.mapBounds[0]), np.int32(self.stepSize), block=self.blocksize, grid=self.gridsize)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3
      self.swapMaps() # swap maps

    def swapMaps(self):
      # swap maps (flood map and temp map)
      self.start_gpu_time.record()
      self.d_floodMap, self.d_tempMap = self.d_tempMap, self.d_floodMap # swap tempMap with floodMap for next iteration
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_transfer_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def swapFloodMaps(self):
      # swap maps (flood map and query map)
      self.start_gpu_time.record()
      self.d_floodMap, self.d_queryMap = self.d_queryMap, self.d_floodMap # swap queryMap with floodMap for next iteration
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_transfer_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def query(self):
      print 'Calling Query kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize[0],self.blocksize[1],self.gridsize[0],self.gridsize[1])
      self.start_gpu_time.record()
      self.query_kernel(self.d_floodMap, self.d_queryMap, self.d_discPoints, self.d_queryValues, np.int32(self.mapLength), np.int32(self.nPoints), block=self.blocksize, grid=self.gridsize)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def islandRemoval(self):
      print 'Calling Island Removal kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize[0],self.blocksize[1],self.gridsize[0],self.gridsize[1])
      self.start_gpu_time.record()
      self.island_removal_kernel(self.d_floodMap, self.d_tempMap, self.d_discPoints, np.int32(self.mapLength), np.int32(self.mapBounds[0]), block=self.blocksize, grid=self.gridsize)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3
      self.swapMaps()

    def determineColors(self):
      print 'Calling determineColors kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize_dP[0],self.blocksize_dP[1],self.gridsize_dP[0],self.gridsize_dP[1])
      self.start_gpu_time.record()
      self.determineColors_kernel(self.d_colors, self.d_discPoints, self.d_queryValues, np.int32(self.nPoints), np.int32(self.nQueries), block=self.blocksize_dP, grid=self.gridsize_dP)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def voronoiImage(self, choice=1):
      print 'Calling Voronoi kernel: Block Size (%d,%d), Grid Size (%d,%d)' % (self.blocksize[0],self.blocksize[1],self.gridsize[0],self.gridsize[1])
      self.start_gpu_time.record()
      if(choice == 0): # save the Voronoi Diagram of the original points
          self.voronoi_kernel(self.d_floodMap, self.d_voronoi, self.d_colors, np.int32(self.mapLength), np.int32(self.mapBounds[0]), block=self.blocksize, grid=self.gridsize)
      else: # save the Voronoi diagram of the query points
          self.voronoi_kernel(self.d_queryMap, self.d_voronoi, self.d_colors, np.int32(self.mapLength), np.int32(self.mapBounds[0]), block=self.blocksize, grid=self.gridsize)
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_compute_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def dataToHost(self):
      self.start_gpu_time.record()
      #self.h_floodMap = self.d_floodMap.get().reshape([self.mapLength,self.mapLength])
      self.h_voronoi = self.d_voronoi.get().view(np.uint8).reshape([self.mapLength,self.mapLength,3])
      self.h_discPoints = self.d_discPoints.get().view(np.int32).reshape([self.nPoints+self.nQueries,3])
      self.end_gpu_time.record()
      self.end_gpu_time.synchronize()
      self.gpu_transfer_time += self.start_gpu_time.time_till(self.end_gpu_time)*1e-3

    def reportBenchmark(self):
      print "Total GPU Time: %1.10f" % (self.gpu_transfer_time + self.gpu_compute_time)
      print "Data Transfer Time: %1.10f" % (self.gpu_transfer_time)
      print "Computation Time: %1.10f" % (self.gpu_compute_time)

    def saveImage(self, out_file_name="voronoi.png"):
      # Put blue dots on the query point locations
      showCenter = 1
      if(showCenter == 1):
          pixWid = int(np.floor(0.04/self.mapRes))
          for i in xrange(0,self.nQueries):
              coords = self.h_discPoints[i+self.nPoints,:];
              for j in xrange(-pixWid,pixWid+1):
                  for k in xrange(-pixWid,pixWid+1):
                      self.h_voronoi[coords[1]+j,coords[0]+k,:] = [255,127,127]
      # save image
      cv2.imwrite(out_file_name, self.h_voronoi)


if __name__ == '__main__':
  # Image files
  in_file_name = "../data/2D/testdata2.txt"
  out_file_name = "voronoi2.png"

  # Import data using dataIO
  data = dataIO.textDataIO(in_file_name, np.float32)
  points = data.points
  nPoints = data.nPoints
  nQueries = data.nQueries
  mapRes = 0.05 #0.05 - 0.005 # size of each grid / should be calculated automatically depending on the data
  mapBounds = [0,25] # the min and max of the grid

  jfa = JFA(points, nPoints, nQueries, mapRes, mapBounds)  
  jfa.saveImage(out_file_name)
  
  # IF A CUSTOM QUERY GRID/POINTS IS GOING TO BE USED
  # THEN THE CODE BELOW SHOULD BE USED TO LOAD DATA
  # Load data points
  #points = np.loadtxt(in_file_name, dtype=np.float32)
  #nPoints = np.size(points,0)
  #print 'File read: %d by %d matrix' % (nPoints,np.size(points,1))
  # Load query points
  #queryPoints = np.loadtxt('../data/2D/gridFull.txt', dtype=np.float32)
  #nQueries = np.size(queryPoints,0)
  # Combine points and queryPoints
  #points = np.concatenate((points,queryPoints),axis=0)

  # CUSTOM QUERY POINTS CAN BE CREATED USING THE CODE BELOW IF YOU DON'T WANT TO USE A FILE
  #queryPoints = np.float32(np.asarray([[5.,6.,0],[5,7,0]]))
  #nQueries = 2
