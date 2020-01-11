import numpy as np
from mayavi import mlab
from numba import njit, jit
from numpy.linalg  import norm
import time

def cubic_spline_2d(xij, h):
	rij = norm(xij)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		return a*(1.0 - 1.5*(q**2.0) + 0.75*(q**3.0))
	elif q<=2.0:
		return 0.25* a * (2.0-q)**3.0
	else:
		return np.array([0.0, 0.0])

def cubic_spline_derivative_2d(xij,h):
	rij = norm(xij)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		if not rij == 0.0 :
			return a*((-3.0*q)+ 2.25*(q**2))*(xij/rij)*(1/h)
		else :
			return np.array([0.0, 0.0])
	elif q<=2.0:
		return (-0.75*a*(2-q)**2)*(xij/rij)*(1/h)
	elif q>2.0:
		return np.array([0.0, 0.0])


class DamBreak2D():
    def __init__(self):
        self.rho0 = 1000
        self.g = 9.81
        self.gamma = 7
        self.container_width = 4
        self.container_height = 4
        self.layers = 3
        self.fluid_height = 2
        self.fluid_width = 1
        self.aplha = 0.1
        self.beta = 0
        self.dx = 0.05
        self.eta = 1e-4
        self.epsilon = 0.3
        self.hdx = 1.3
        self.sx, self.sy, self.fx, self.fy = self.create_particles()
        self.c0 = 10 * np.sqrt(2 * self.g * self.fluid_height)
        self.B = (self.c0**2)*self.rho0/self.gamma

        self.sx1 = np.vstack((self.sx,self.sy)).T
        self.fx1 = np.vstack((self.fx,self.fy)).T

        self.h = self.hdx * self.dx

        self.solid = {"m": np.ones_like(self.sx)*(self.dx**2)*self.rho0,"p" : np.zeros_like(self.sx),
                      "rho" : np.ones_like(self.sx)*self.rho0, "x": self.sx1, "v": np.zeros((len(self.sx),2))}

        self.fluid = {"m": np.ones_like(self.fx)*(self.dx**2)*self.rho0,"p" : np.zeros_like(self.fx),
                      "rho" : np.ones_like(self.fx)*self.rho0, "x": self.fx1, "v": np.zeros((len(self.fx),2))}


    def create_particles(self):

        # Creating the fluid particles
        fluid_x, fluid_y = np.mgrid[0:self.fluid_width+self.dx:self.dx,0:self.fluid_height+self.dx:self.dx]
        fluid_x = fluid_x.ravel()
        fluid_y = fluid_y.ravel()
        fluid_x, fluid_y = fluid_x + self.dx/2, fluid_y + self.dx/2

        # Creating the solid particles

        layers = self.layers
        dx = self.dx
        solid_x, solid_y = np.mgrid[-layers*dx:self.container_width+(layers+1)*dx:dx, -layers*dx:self.container_height:dx]
        idx = []
        solid_x = solid_x.ravel()
        solid_y = solid_y.ravel()

        for i in range(len(solid_x)):
            if solid_x[i] <= self.container_width and solid_x[i] >= 0:
                if solid_y[i] >= 0:
                    idx.append(i)

        solid_x = np.delete(solid_x,idx)
        solid_y = np.delete(solid_y,idx)

        return solid_x, solid_y, fluid_x, fluid_y

    def update_values(self,dt):


        print(dt)
        solid = self.solid
        fluid = self.fluid

        arho_solid = np.zeros_like(self.solid["m"])
        B = (self.c0**2)*self.rho0/self.gamma
        arho_fluid = np.zeros_like(self.fluid["m"])
        acc_fluid = np.zeros_like(self.fluid["x"])
        dxdt_fluid = np.zeros_like(self.fluid["x"])

        h = self.h

        # for solid particles
        for i in range(len(solid["m"])):
            sum_rho = 0

            # fluid source
            for j in range(len(fluid["m"])):
                xij = solid["x"][i] - fluid["x"][j]
                rij = norm(xij)

                if rij < 3 * h:
                    vij = solid["v"][i] - fluid["v"][j]
                    sum_rho += fluid["m"][j] * np.dot(vij,cubic_spline_derivative_2d(xij,h)) / fluid["rho"][j]

            arho_solid[i] = solid["rho"][i] * sum_rho


        # for fluid particles

        for i in range(len(fluid["x"])):
            sum_rho = 0
            sum_vel = 0
            sum_x = 0

            # solid source
            for j in range(len(solid["x"])):
                xij = fluid["x"][i] - solid["x"][j]
                rij = norm(xij)

                vij = fluid["v"][i] - solid["v"][j]
                rhoij = 0.5 * (fluid["rho"][i] + solid["rho"][j])

                muij = h * np.dot(vij,xij) / (rij**2 + self.eta**2)
                pi_ij = 0

                if muij > 0:
                    pi_ij = 0
                else :
                    cij = self.c0
                    # ci = self.c0 * pow(fluid["rho"][i]/self.rho0, 0.5 * (self.gamma - 1))
                    # cj = self.c0 * pow(solid["rho"][j]/self.rho0, 0.5 * (self.gamma - 1))
                    # cij = 0.5*(ci+cj)
                    pi_ij = -self.aplha * cij * muij / rhoij

                pj_rhoj = solid["p"][j] / pow(solid["rho"][j],2)
                pi_rhoi = fluid["p"][i] / pow(fluid["rho"][i],2)

                if rij < 3 * h:

                    sum_rho += solid["m"][j] * np.dot(vij,cubic_spline_derivative_2d(xij,h)) / solid["rho"][j]
                    sum_x += solid["m"][j] * vij * cubic_spline_2d(xij,h) / rhoij
                    sum_vel += (solid["m"][j] * (pi_rhoi+pj_rhoj+pi_ij) * cubic_spline_derivative_2d(xij,h))


            # fluid source
            for j in range(len(fluid["x"])):
                xij = fluid["x"][i] - fluid["x"][j]
                rij = norm(xij)

                vij = fluid["v"][i] - fluid["v"][j]
                rhoij = 0.5*(fluid["rho"][i]+fluid["rho"][j])

                muij = h * np.dot(vij,xij) / (rij**2+self.eta**2)
                pi_ij = 0

                if muij > 0:
                    pi_ij = 0
                else :
                    # ci = self.c0 * pow(fluid["rho"][i]/self.rho0, 0.5 * (self.gamma - 1))
                    # cj = self.c0 * pow(fluid["rho"][j]/self.rho0, 0.5 * (self.gamma - 1))
                    # cij = 0.5*(ci+cj)
                    cij = self.c0
                    pi_ij = -self.aplha * cij * muij / rhoij

                pj_rhoj = fluid["p"][j] / pow(fluid["rho"][j],2)
                pi_rhoi = fluid["p"][i] / pow(fluid["rho"][i],2)

                if rij < 3 * h:

                    sum_rho += fluid["m"][j] * np.dot(vij,cubic_spline_derivative_2d(xij,h)) / fluid["rho"][j]
                    sum_x += fluid["m"][j] * vij * cubic_spline_2d(xij,h) / rhoij
                    sum_vel += (fluid["m"][j] * (pi_rhoi+pj_rhoj+pi_ij) * cubic_spline_derivative_2d(xij,h))


            arho_fluid[i] = fluid["rho"][i] * sum_rho
            acc_fluid[i] = -1 * sum_vel - np.array([0,self.g])
            dxdt_fluid[i] = fluid["v"][i] - (self.epsilon * sum_x)


        self.solid["rho"] += dt * arho_solid
        self.solid["p"] = B * (pow(self.solid["rho"]/self.rho0, self.gamma) - 1)

        for i in range(len(self.solid["p"])):
            if self.solid["p"][i] < 0:
                self.solid["p"][i] = 0

        self.fluid["rho"] += arho_fluid * dt
        self.fluid["v"] += acc_fluid * dt
        self.fluid["x"] += dxdt_fluid * dt
        self.fluid["p"] = B * (pow(self.fluid["rho"]/self.rho0, self.gamma) - 1)



if __name__ == "__main__":
    dam = DamBreak2D()
    dt = 0.125 * dam.h / dam.c0
    tf = 2
    t = np.arange(0,tf,dt)
    hist1, hist2 = [] , []
    for i in range(len(t)):
        a = time.time()
        dam.update_values(dt)
        hist1.append(dam.fluid["x"])
        hist2.append(dam.solid["x"])
        print(dam.fluid["x"])
        np.savez("data1.npz", np.array(hist1))
        np.savez("data2.npz", np.array(hist2))
        b = time.time()
        print("time",i,len(t), b-a, "s")

    mlab.points3d(dam.fluid["x"].T[0],dam.fluid["x"].T[1],np.zeros_like(dam.fluid["x"].T[0]))
    mlab.points3d(dam.solid["x"].T[0],dam.solid["x"].T[1],np.zeros_like(dam.solid["x"].T[0]))
