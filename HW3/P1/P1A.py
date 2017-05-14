# Written by Alperen Degirmenci
# CS 205 HW3 P1A
# 10/24/12

import numpy as np
from Plotter3DCS205 import MeshPlotter3D as MP3D, MeshPlotter3DParallel as MP3DP
from mpi4py import MPI

# Global constants
xMin, xMax = 0.0, 1.0
yMin, yMax = 0.0, 1.0     # Domain boundaries
Nx = 64                  # Numerical grid size
dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
Ny, dy = Nx, dx           # Ny = Nx, dy = dx
dt = 0.4 * dx             # Time step (Magic factor of 0.4)
T = 7                     # Time end
DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar
NxLocal, NyLocal = 0, 0   # initialize local grid size
Ix, Iy, Gx, Gy = 0, 0, 0, 0 # initialize grid size variables
Px = 1 # processes in x-dir

def initial_conditions(x0, y0, cartComm):
    '''Construct the grid cells and set the initial conditions'''
    coords = cartComm.Get_coords(cartComm.Get_rank())
    um = np.zeros([Gx,Gy])     # u^{n-1}  "u minus"
    u  = np.zeros([Gx,Gy])     # u^{n}    "u"
    up = np.zeros([Gx,Gy])     # u^{n+1}  "u plus"
    # Set the initial condition on interior cells: Gaussian at (x0,y0)
    #[I,J] = np.mgrid[Ix*coords[0]+1:Ix*(coords[0]+1), Iy*coords[1]+1:Iy*(coords[1]+1)]
    [I,J] = np.mgrid[NxLocal*coords[0]+1:NxLocal*(coords[0]+1)+1, NyLocal*coords[1]+1:NyLocal*(coords[1]+1)+1]
    u[1:Ix,1:Iy] = np.exp(-50 * (((I-1)*dx-x0)**2 + ((J-1)*dy-y0)**2))
    # Set the ghost cells to the boundary conditions
    set_ghost_cells(u, cartComm)
    # Set the initial time derivative to zero by running backwards
    apply_stencil(um, u, up)
    um *= 0.5
    # Done initializing up, u, and um
    return up, u, um

def apply_stencil(up, u, um):
    '''Apply the computational stencil to compute up.
    Assumes the ghost cells exist and are set to the correct values.'''
    # Update interior grid cells with vectorized stencil
    up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]
                     + DTDX*(u[0:Ix-1,1:Iy] + u[2:Ix+1,1:Iy] +
                             u[1:Ix,0:Iy-1] + u[1:Ix,2:Iy+1]))

def set_ghost_cells(u, cartComm):
    rank = cartComm.Get_rank()
    cRank = cartComm.Get_coords(rank)
    
    # get neighboring procs
    up,down = cartComm.Shift(0,1) # neighbors in the y-direction
    left,right = cartComm.Shift(1,1) # neighbors in the x-direction
    
    # determine ghost cell assignment depending on neighbors
    if up == -2:
        u[0,:] = u[2,:];       # u_{0,j}    = u_{2,j}
    else:
        cartComm.Sendrecv(sendbuf=u[1,:], dest=up, recvbuf=u[0,:], source=up)
    if down == -2:
        u[NxLocal+1,:] = u[NxLocal-1,:];    # u_{Nx+1,j} = u_{Nx-1,j}
    else:
        cartComm.Sendrecv(sendbuf=u[NxLocal,:], dest=down, recvbuf=u[NxLocal+1,:], source=down)
    if left == -2:
        u[:,0] = u[:,2];       # u_{i,0}    = u_{i,2}
    else:
        buf = np.array(u[:,0])
        cartComm.Sendrecv(sendbuf=np.array(u[:,1]), dest=left, recvbuf=buf, source=left)
        u[:,0] = buf.T
    if right == -2:
        u[:,NyLocal+1] = u[:,NyLocal-1];    # u_{i,Ny+1} = u_{i,Ny-1}
    else:
        buf = np.array(u[:,NyLocal+1])
        cartComm.Sendrecv(sendbuf=np.array(u[:,NyLocal]), dest=right, recvbuf=buf, source=right)
        u[:,NyLocal+1] = buf.T

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # number of processes
    rank = comm.Get_rank() # rank of current process
    #Px = 1
    Py = size/Px # assume size is a power of 2
    assert Px*Py == size # sanity check
    
    cartComm = comm.Create_cart((Px,Py)) # create a cartesian comm
    # size of local compute grid
    NxLocal = Nx/Px
    NyLocal = Ny/Py # assume Nx is a power of 2
    
    # Domain
    Gx, Gy = NxLocal+2, NyLocal+2 # Ghost cells
    Ix, Iy = NxLocal+1, NyLocal+1

    # Initialize grid
    up, u, um = initial_conditions(0.5, 0, cartComm)

    print 'Proc %d: Initial conditions set. Processing...' % rank

    # Coordinates of current process in the Cart comm
    coords = cartComm.Get_coords(rank)
    # Mesh indices
    [I,J] = np.mgrid[NxLocal*coords[0]+1:NxLocal*(coords[0]+1)+1, NyLocal*coords[1]+1:NyLocal*(coords[1]+1)+1]
    # Parallel plotter
    plotter = MP3DP(I,J,u[1:Ix,1:Iy], cartComm)
    comm.Barrier() # sync procs    
    p_start = MPI.Wtime() # start timing
    # Domain decomposition
    for k,t in enumerate(np.arange(0,T,dt)):
        apply_stencil(up, u, um)
        um, u, up = u, up, um
        set_ghost_cells(u, cartComm)
        # update plot every 10 frames
        if k % 10 == 0:
            plotter.draw_now(I,J,u[1:Ix,1:Iy])
    
    comm.Barrier()
    p_stop = MPI.Wtime() # stop timing
    t = p_stop - p_start # total iteration time
    t_ave = t/k # average time per iteration
    if rank == 0:
        print '\nP1A results:::'
        print 'Grid size: %d x %d' % (Nx, Ny)
        print 'Overall iteration time: % secs' % t
        print 'Average time per iteration: %f secs' % t_ave
        print 'Px = %d, Py = %d' % (Px,Py)
    # Save final image
    plotter = MP3DP(I,J,u[1:Ix,1:Iy], cartComm)
    plotter.save_now("FinalWave.png")
