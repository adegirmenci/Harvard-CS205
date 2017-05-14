# Alperen Degirmenci
# CS 205 Project
# Driver Program

import numpy as np
import dataIO
import JumpFlooding

if __name__ == '__main__':
  # Inputs
  in_file_name = "../data/2D/testdata2.txt"
  #in_file_name = "../data/2D/testMRimage.raw"
  out_file_name = "voronoi2.png"
  mapRes = 0.05 #0.1 - 0.01 / 0.05 is good / should be a fraction of mesh resolution (which is 0.25)

  # Import data using dataIO
  data = dataIO.textDataIO(in_file_name, np.float32)
  #data = dataIO.imgDataIO(in_file_name, np.uint16, [64,64], 1)
  points = data.points
  nPoints = data.nPoints 
  nQueries = data.nQueries
  mapBounds = data.boundingBox[0,:]# the min and max of the grid
  # Perform NNI
  jfa = JumpFlooding.JFA(points, nPoints, nQueries, mapRes, mapBounds)  
  # Save image
  jfa.saveImage(out_file_name)
