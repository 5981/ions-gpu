import time
import numpy as np
from scipy.constants import (physical_constants as phys, c, m_u, hbar, pi, e, m_e)

from nbody import NBody

class Ion:
    name = 0
    def __init__(self, r=np.zeros(3), v=np.zeros(3), m=m_u, k=0, Z=1):
        self.r = r
        self.v = v
        self.m = m
        self.Z = Z
        self.k = k
        self.name = Ion.name
        Ion.name += 1
        return

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "{:06.2f} {:06.2f}".format(self.r[0], self.v[0])


class Trap(NBody):
    """
    linear quadrupole trap
    fill it with ions:
    positions
    velocities
    weights (weights of ions in the sum used to calculate field
             i.e. E=k*Z*e*r/r**3, so weight = k*Z*e)
    f2a (field to acceleration for each ion, i.e. Z*e/m)
    viscosities (a = -kv, k is viscosity)
    """
    micro = True # calculate micromotion
    V = 350 # V, AC
    w = 2*pi*18.0e6 # 2*pi*Hz
    U = 0 # V, DC
    Uz = 100 # V, DC
    xySize = 1.14e-3 # m, radius
    zSize = 7e-3 #m, half of length
    
    beamR = 1e-3 # m
    # use self.config(Z=1, m=171) to change any of the above
    # or during self.init(1e4, sigmaV=1e-2, Z=1, m=171)

    # load ions, or use 'ions' random ones if 'ions' is a number
    def __init__(self, ions, sigmaV=1, viscosity=1e3, **kwargs):
        self.config(**kwargs)
        # assume Thorium trap by default
        E = e/(4*pi*phys['electric constant'][0])
        Z = 1
        try:
            N = int(ions)
            rnd = True
        except TypeError:
            rnd = False
        if rnd:
            np.random.seed(42)
            positions = (2*np.random.rand(N,3)-1)*min(self.xySize, self.zSize)/100
            velocities = np.random.randn(N,3)*sigmaV
            weights = np.array([Z*E]*N)
            
            # TODO
            # this should really be of the shape (N,3)
            viscosities = np.array([viscosity]*N)
            
            f2a = np.array([Z*e/(25*m_u)]*N)
        else:
            positions = np.array(ion.r for ion in ions)
            velocities = np.array(ion.v for ion in ions)
            weights = np.array(E*ion.Z for ion in ions)
            viscosities = np.array(ion.k for ion in ions)
            f2a = np.array(e*ion.Z/ion.m for ion in ions)
        self.N = len(positions)
        return super().__init__(positions, velocities, weights,
                                viscosities, f2a)
    
    def config(self, **kwargs):
        # careful not to change something you are not supposed to
        # I won't fool-proof this
        for key,value in kwargs.items():
            setattr(self, key, value)
        self.u = self.U/(self.xySize*self.xySize)
        self.v = self.V/(self.xySize*self.xySize)
        self.uz = self.Uz/(self.zSize*self.zSize)
        return
    
    def confinement(self):
        if self.micro:
            xConf = self.u - self.v*np.cos(self.w*self.T)
            yConf = -xConf
            if self.T*self.w > 42*2*pi:
                self.T -= 42*2*pi/self.w
        else:
            xConf = self.v/1.414
            yConf = xConf
        zConf = self.uz/1.414
        return (xConf, yConf, zConf)
    
    def size(self):
        return self.xySize, self.xySize, self.zSize
    
    def viscosityR(self):
        return self.beamR

    def clean(self):
        # leave everyone who is out of range of our trap
        newPos = np.empty([self.N,3], dtype=np.float32)
        newVel = np.empty([self.N,3], dtype=np.float32)
        newWei = np.empty(self.N, dtype=np.float32)
        newVis = np.empty(self.N, dtype=np.float32)
        newf2a = np.empty(self.N, dtype=np.float32)
        j = 0
        for i in range(self.N):
            if (abs(self.positions[i][0]) < self.xySize and
                abs(self.positions[i][1]) < self.xySize and
                abs(self.positions[i][2]) < self.zSize):
                newPos[j] = self.positions[i]
                newVel[j] = self.velocities[i]
                newWei[j] = self.weights[i]
                newVis[j] = self.viscosities[i]
                newf2a[j] = self.f2a[i]
                j += 1
        if j < 1:
            print("WE'VE LOST THEM ALL", self.velocities[0])
            return
        self.positions = newPos[:j,:]
        self.velocities = newVel[:j,:]
        self.weights = newWei[:j]
        self.viscosities = newVis[:j]
        self.f2a = newf2a[:j]
        self.N = len(self.weights)

    def __call__(self, dt=1e-8, steps=1, debug=False):
        before = time.perf_counter()

        super().__call__(dt, steps)
        self.clean()

        return ('{0:10.2e}'.format(self.t*1000),
                '{0:05d}'.format(self.N),
                '{:10.2e}'.format(time.perf_counter() - before))
