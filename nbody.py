# most of the code was taken from example:
# https://github.com/numba/numba-benchmark/blob/master/benchmarks/bench_cuda.py
# and modified

import math
import numpy as np

from numba import cuda, float32

# N-body simulation. Here we calculate accelerations and move the thing
# CUDA version adapted from http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html

eps_2 = np.float32(1e-14)
zero = np.float32(0.0)
one = np.float32(1.0)
@cuda.jit(device=True, inline=True)
def body_body_interaction(xi, yi, zi, xj, yj, zj, wj, fxi, fyi, fzi):
    """
    Compute the influence of body j on the acceleration of body i.
    """
    # f - field
    # w - kind of wrongly named weight (Gm for gravity, kZe for ions)
    # f = w*r/r**3
    # dumb it down for just-in-time compiler
    rx = xi - xj
    ry = yi - yj
    rz = zi - zj
    sqr_dist = rx*rx + ry*ry + rz*rz + eps_2
    sixth_dist = sqr_dist * sqr_dist * sqr_dist
    inv_dist_cube = one / math.sqrt(sixth_dist)
    s = wj * inv_dist_cube
    fxi += rx * s
    fyi += ry * s
    fzi += rz * s
    return fxi, fyi, fzi


@cuda.jit(device=True, inline=True)
def tile_calculation(xi, yi, zi, fxi, fyi, fzi, positions, weights):
    """
    Compute the contribution of this block's tile to the acceleration
    of body i.
    """
    for j in range(cuda.blockDim.x):
        xj = positions[j,0]
        yj = positions[j,1]
        zj = positions[j,2]
        w = weights[j]
        fxi, fyi, fzi = body_body_interaction(xi, yi, zi, xj, yj, zj, w, fxi, fyi, fzi)
    return fxi, fyi, fzi

tile_size = 64

# I have not checked the statement below
# Don't JIT this function at the top-level as it breaks until Numba 0.16.
# I have not checked the statement above
def calcMove(positions, velocities, weights, f2a, viscosities,
             beamR, xSize, ySize, zSize, xConf, yConf, zConf, dt):
    """
    Calculate accelerations produced on all bodies by mutual forces.
    """
    sh_positions = cuda.shared.array((tile_size, 3), float32)
    sh_weights = cuda.shared.array(tile_size, float32)
    i = cuda.grid(1)
    xi = positions[i,0]
    yi = positions[i,1]
    zi = positions[i,2]
    vxi = velocities[i,0]
    vyi = velocities[i,1]
    vzi = velocities[i,2]
    v2ax = viscosities[i]
    v2ay = viscosities[i]
    v2az = viscosities[i]
    fxi = zero
    fyi = zero
    fzi = zero
    if (xi < xSize and yi < ySize and zi < zSize and
        -xi < xSize and -yi < ySize and -zi < zSize):
        fxi -= xConf*xi
        fyi -= yConf*yi
        fzi -= zConf*zi
    for j in range(0, len(weights), tile_size):
        index = (j // tile_size) * cuda.blockDim.x + cuda.threadIdx.x
        sh_index = cuda.threadIdx.x
        sh_positions[sh_index,0] = positions[index,0]
        sh_positions[sh_index,1] = positions[index,1]
        sh_positions[sh_index,2] = positions[index,2]
        sh_weights[sh_index] = weights[index]
        # this here is the reason we use CUDA
        # this function is called from multiple threads
        # make sure all threads are ready to calculate field
        cuda.syncthreads()
        # they are all ready, let's calculate
        fxi, fyi, fzi = tile_calculation(xi, yi, zi, fxi, fyi, fzi,
                                         sh_positions, sh_weights)
        # wait for others to finish too
        cuda.syncthreads()

    axi = fxi * f2a[i]
    ayi = fyi * f2a[i]
    azi = fzi * f2a[i]
    if (xi < beamR and yi < beamR and zi < beamR and
        -xi < beamR and -yi < beamR and -zi < beamR):
        # spherical area of viscosity
        axi -= v2ax*vxi
        ayi -= v2ay*vyi
        azi -= v2az*vzi
    # now we know all accelerations
    # apply dt shift
    # let's not use fancy runge-kutta while on the device
    # keep it simple
    # copy it back using NBody.results, then think about what we've done using CPU
    positions[i,0] = xi + vxi*dt
    positions[i,1] = yi + vyi*dt
    positions[i,2] = zi + vzi*dt
    velocities[i,0] = vxi + axi*dt
    velocities[i,1] = vyi + ayi*dt
    velocities[i,2] = vzi + azi*dt


class NBody:
    def __init__(self, positions, velocities, weights, viscosities, f2a):
        self.calcMove = cuda.jit(
            argtypes=(float32[:,:], float32[:,:], float32[:], float32[:], float32[:],
                      float32, float32, float32, float32, float32, float32, float32, float32)
            )(calcMove)
        
        self.N = len(weights)
        self.positions = positions.astype(np.float32, copy=False)
        self.velocities = velocities.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.viscosities = viscosities.astype(np.float32, copy=False)
        self.f2a = f2a.astype(np.float32, copy=False)
        self.stream = cuda.stream()
        self.d_pos = None
        self.d_vel = None
        self.d_wei = None
        self.d_f2a = None
        self.d_vis = None
        self.t = zero # for observant
        self.T = zero # high-precision t

    def prepare(self):
        # this stuff is relatively heavy
        # let's do several dt steps at once
        
        self.d_pos = cuda.to_device(self.positions, self.stream)
        self.d_vel = cuda.to_device(self.velocities, self.stream)
        self.d_wei = cuda.to_device(self.weights, self.stream)
        self.d_f2a = cuda.to_device(self.f2a, self.stream)
        self.d_vis = cuda.to_device(self.viscosities, self.stream)
        self.stream.synchronize()

    def run(self, dt, steps):
        for step in range(steps):
            self.t += dt
            self.T += dt
            xConf, yConf, zConf = self.confinement()
            xSize, ySize, zSize = self.size()
            beamR = self.viscosityR()
            blockdim = tile_size
            griddim = int(math.ceil(self.N / blockdim))
            self.calcMove[griddim, blockdim, self.stream](
                self.d_pos, self.d_vel, self.d_wei, self.d_f2a, self.d_vis,
                beamR, xSize, ySize, zSize, xConf, yConf, zConf, dt)
            self.stream.synchronize()
    
    def confinement(self):
        """
        x'' = -kx, returns k
        """
        return (zero,zero,zero)
    
    def size(self):
        return (42,42,42)
    
    def viscosityR(self):
        return zero

    def results(self):
        # another heavy stuff
        # get back results and check who has flown away using CPU
        self.d_pos.copy_to_host(self.positions, self.stream)
        self.d_vel.copy_to_host(self.velocities, self.stream)
        
        self.stream.synchronize()
        # cuda.api.close()
        # this line seems to reduce the rate of device's memory glitches
        cuda.current_context().deallocations.clear()
        # del self.stream
        return

    def __call__(self, dt, steps):
        self.prepare()
        self.run(dt, steps)
        return self.results()
