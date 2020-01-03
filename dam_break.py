import numpy as np
from mayavi import mlab
from numba import njit
from numpy.linalg  import norm

# @njit
def cubic_spline_2d(xij, h):
	rij = norm(xij)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		return a*(1.0-(1.5*(q**2.0)+(0.75*(q**3.0))))
	elif q<=2.0:
		return 0.25*a*(2.0-q)**3.0
	else:
		return 0.0

# @njit
def cubic_spline_derivative_2d(xij,h):
	rij = norm(xij)
	q = rij/h
	a = float(10.0/(7.0*np.pi*(h**2)))
	if q<=1.0 and q>=0.0:
		if not xij ==0.0 :
		    return float(a*((-3.0*q)+ 2.25*(q**2))*(xij/rij)/h)
		else :
		    return 0.0
	elif q<=2.0:
		    return float((-0.75*a*(2-q)**2)*(xij/rij)/h)
	elif q>2.0:
		return 0.0


class DamBreak2D():
    def __init__(self):
        self.rho0 = 1000
        self.c0 = 330
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
        self.eta = 1e-4
        self.epsilon = 0.5

        self.sx, self.sy, self.fx, self.fy = self.create_particles()

        self.sx1 = np.vstack((self.sx,self.sy)).T
        self.fx1 = np.vstack((self.fx,self.fy)).T

        self.solid = {"m": np.ones_like(self.sx),"p" : np.zeros_like(self.sx),"rho" : np.ones_like(self.sx)*self.rho0,
                 "x": self.sx1, "v": np.zeros((len(self.sx),2)), "h": np.ones_like(self.sx)*1.2*self.dx}

        self.fluid = {"m": np.ones_like(self.fx),"p" : np.zeros_like(self.fx),"rho" : np.ones_like(self.fx)*self.rho0,
                 "x": self.fx1, "v": np.zeros((len(self.fx),2)), "h": np.ones_like(self.sx)*1.2*self.dx}

        self.x = np.concatenate((self.sx1,self.fx1))

        self.sf = {"m": np.ones_like(self.x),"p" : np.zeros_like(self.x),"rho" : np.ones_like(self.x)*(self.dx**2),
                 "x": self.x, "v": np.zeros_like(self.x), "h": np.ones_like(self.x)*1.2*self.dx}


    def create_particles(self):

        # Creating the fluid particles
        fluid_x, fluid_y = np.mgrid[0:self.fluid_width+self.dx:self.dx,0:self.fluid_height+self.dx:self.dx]
        fluid_x = fluid_x.ravel()
        fluid_y = fluid_y.ravel()
        fluid_x, fluid_y = fluid_x + self.dx, fluid_y + self.dx

        # Creating the solid particles

        layers = self.layers
        dx = self.dx
        solid_x, solid_y = np.mgrid[-layers*dx:self.container_width+4*dx:dx, -layers*dx:self.container_height:dx]
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

    # @njit(parallel=True)
    def solid_calc(self,solid,sf):
        arho_solid = np.zeros_like(solid["m"])
        for i in range(len(solid["m"])):
            sum_rho = 0

            for j in range(sf["m"]):
                xij = solid["x"][i] - sf["x"][j]
                vij = solid["v"][i] - sf["v"][j]

                sum_rho += (sf["m"][j]*vij*cubic_spline_2d(xij,solid["h"][i]))/sf["rho"][j]

            arho_solid[i] = solid["rho"][i] * sum_rho

        return arho_solid

    # @njit(parallel=True)
    def fluid_calc(self,fluid, sf):
        arho_fluid = np.zeros_like(fluid["m"])
        acc_fluid = np.zeros_like(fluid["x"])
        dxdt_fluid = np.zeros_like(fluid["x"])

        for i in range(len(fluid["x"])):
            sum_rho = 0
            sum_vel = 0
            sum_x = 0

            for j in range(len(sf["x"])):
                xij = fluid["x"][i] - sf["x"][j]
                rij = norm(xij)
                vij = fluid["v"][i] - sf["v"][j]
                rhoij = 0.5*(fluid["rho"][i]+sf["rho"][j])

                muij = fluid["h"][i] * np.dot(vij,xij) / (rij+self.eta**2)
                pi_ij = -self.aplha * self.c0 * muij / rhoij

                pj_rhoj = sf["p"][j] / sf["rho"][j]**2
                pi_rhoi = sf["p"][i] / sf["rho"][i]**2

                sum_rho += (sf["m"][j]*vij*cubic_spline_2d(xij,fluid["h"][i]))/sf["rho"][j]
                sum_vel += sf["m"][j]*(pi_rhoi+pj_rhoj+pi_ij)*cubic_spline_derivative_2d(xij,fluid["h"][i]) - np.array([0,self.g])
                sum_x += sf["m"][j]*vij*cubic_spline_2d(xij,fluid["h"][i])

            arho_fluid[i] = fluid["rho"][i] * sum_rho
            acc_fluid[i] = -1 * sum_vel
            dxdt_fluid[i] = fluid["v"][i] + self.epsilon * sum_x

        return arho_fluid, acc_fluid, dxdt_fluid


    def update_values(self,tf,dt):

        solid = self.solid
        fluid = self.fluid
        sf = self.sf

        arho_solid = np.zeros_like(self.solid["m"])
        B = (self.c0**2)*self.rho0/self.gamma
        arho_fluid = np.zeros_like(fluid["m"])
        acc_fluid = np.zeros_like(fluid["x"])
        dxdt_fluid = np.zeros_like(fluid["x"])

        # for solid particles
        for i in range(len(solid["m"])):
            sum_rho = 0

            for j in range(len(sf["m"])):
                xij = solid["x"][i] - sf["x"][j]
                vij = solid["v"][i] - sf["v"][j]

                sum_rho += (sf["m"][j]*np.dot(vij,cubic_spline_2d(xij,solid["h"][i])))/sf["rho"][j]

            print(sum_rho)
            arho_solid[i] = solid["rho"][i] * sum_rho

        # arho_solid = self.solid_calc(solid, sf)

        self.solid["rho"] += dt*arho_solid
        self.solid["p"] = B*((solid["rho"]/self.rho0)**(self.gamma) - 1)

        # for fluid particles

        for i in range(len(fluid["x"])):
            sum_rho = 0
            sum_vel = 0
            sum_x = 0

            for j in range(len(sf["x"])):
                xij = fluid["x"][i] - sf["x"][j]
                rij = norm(xij)
                vij = fluid["v"][i] - sf["v"][j]
                rhoij = 0.5*(fluid["rho"][i]+sf["rho"][j])

                muij = fluid["h"][i] * np.dot(vij,xij) / (rij+self.eta**2)
                pi_ij = -self.aplha * self.c0 * muij / rhoij

                pj_rhoj = sf["p"][j] / sf["rho"][j]**2
                pi_rhoi = sf["p"][i] / sf["rho"][i]**2

                sum_rho += sf["m"][j]*np.dot(vij,cubic_spline_2d(xij,fluid["h"][i]))/sf["rho"][j]
                sum_vel += sf["m"][j]*(pi_rhoi+pj_rhoj+pi_ij)*cubic_spline_derivative_2d(xij,fluid["h"][i]) - np.array([0,self.g])
                sum_x += sf["m"][j]*vij*cubic_spline_2d(xij,fluid["h"][i])

            arho_fluid[i] = fluid["rho"][i] * sum_rho
            acc_fluid[i] = -1 * sum_vel
            dxdt_fluid[i] = fluid["v"][i] + self.epsilon * sum_x

        # arho_fluid, acc_fluid, dxdt_fluid = self.fluid_calc(fluid, sf)

        self.fluid["rho"] += arho_fluid * dt
        self.fluid["v"] += acc_fluid * dt
        self.fluid["x"] += dxdt_fluid * dt
        self.fluid["p"] = B*((fluid["rho"]/self.rho0)**(self.gamma) - 1)

        self.sf["p"] = np.append(self.solid["p"],self.fluid["p"])
        self.sf["rho"] = np.append(self.solid["rho"],self.fluid["rho"])
        self.sf["x"] = np.append(self.solid["x"],self.fluid["x"])
        self.sf["v"] = np.append(self.solid["v"],self.fluid["v"])


if __name__ == "__main__":
    dam = DamBreak2D()
    dt = 0.005
    tf = 2
    t = np.arange(0,tf,0.005)
    for i in range(len(t)):
        dam.update_values(dt,tf)
        print(i,len(t))

    mlab.points3d(dam.sf["x"].T[0],dam.sf["x"].T[1],np.zeros_like(dam.sf["x"].T[0]))
