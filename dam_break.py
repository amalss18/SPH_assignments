import numpy as np
from mayavi import mlab
from numba import njit

@njit
def cubic_spline_2d(xij,yij,h):
	rij = np.sqrt(xij**2+yij**2)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		return a*(1.0-(1.5*(q**2.0)+(0.75*(q**3.0))))
	elif q<=2.0:
		return 0.25*a*(2.0-q)**3.0
	else:
		return 0.0

@njit
def cubic_spline_derivative_2d(xij,yij,h):
	rij = np.sqrt(xij**2+yij**2)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		if not xij ==0.0 :
		    return float((a*((-3.0*q)+ 2.25*(q**2))/h)*np.sign(xij))
		else :
		    return 0.0
	elif q<=2.0:
		    return float(((-0.75*a*(2-q)**2)/h)*np.sign(xij))
	elif q>2.0:
		return 0.0

class DamBreak2D():
    def __init__(self):
        self.rho = 1000
        self.g = -9.81
        self.gamma = 7
        self.container_width = 4
        self.container_height = 4
        self.layers = 3
        self.fluid_height = 2
        self.fluid_width = 1
        self.aplha = 0.1
        self.beta = 0
        self.dx = 0.05
        self.sx, self.sy, self.fx, self.fy = self.create_particles()
        self.solid = {"m": np.ones_like(self.sx),"p" : np.zeros_like(self.sx),"rho" : np.ones_like(self.sx)*(self.dx**2),
                 "x": self.sx, "y" : self.sy, "v": np.zeros_like(self.sx)}
        self.fluid = {"m": np.ones_like(self.fx),"p" : np.zeros_like(self.fx),"rho" : np.ones_like(self.fx)*(self.dx**2),
                 "x": self.fx, "y" : self.fy, "v": np.zeros_like(self.fx)}


    def create_particles(self):

        # Creating the fluid particles
        fluid_x, fluid_y = np.mgrid[0:self.fluid_width+self.dx:self.dx,0:self.fluid_height+self.dx:self.dx]
        fluid_x = fluid_x.ravel()
        fluid_y = fluid_y.ravel()
        fluid_x, fluid_y = fluid_x + self.dx, fluid_y + self.dx

        # Creating the solid particles

        layers = self.layers
        solid_x, solid_y = np.mgrid[-3*layers:self.container_width+4*self.dx:self.dx, -3*layers:self.container_height:self.dx]
        idx = []
        solid_x = solid_x.ravel()
        solid_y = solid_y.ravel()

        for i in range(len(solid_x)):
            if solid_x[i] <= self.container_width and solid_x[i] >=0:
                if solid_y[i] >= 0:
                    idx.append(i)

        solid_x = np.delete(solid_x,idx)
        solid_y = np.delete(solid_y,idx)

        return solid_x, solid_y, fluid_x, fluid_y

