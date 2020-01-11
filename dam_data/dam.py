import numpy as np
from numba import njit, jit, prange
from numpy.linalg import norm
import time

def create_particles():
    # Creating the fluid particles
    fluid_x, fluid_y = np.mgrid[0:fluid_width+dx:dx, 0:fluid_height+dx:dx]
    fluid_x = fluid_x.ravel()
    fluid_y = fluid_y.ravel()
    fluid_x, fluid_y = fluid_x + dx/2 , fluid_y + dx/2

    # Creating the solid particles

    solid_x, solid_y = np.mgrid[-layers*dx:container_width+(layers+1)*dx:dx, -layers*dx:container_height:dx]
    idx = []
    solid_x = solid_x.ravel()
    solid_y = solid_y.ravel()

    for i in range(len(solid_x)):
        if solid_x[i] <= container_width and solid_x[i] >= 0:
            if solid_y[i] >= 0:
                idx.append(i)

    solid_x = np.delete(solid_x, idx)
    solid_y = np.delete(solid_y, idx)

    return solid_x, solid_y, fluid_x, fluid_y


@njit(parallel=True)
def update_values(dt, solid_m, solid_p,  solid_rho, solid_x, solid_v, fluid_m, fluid_p, fluid_rho, fluid_x, fluid_v):

    arho_solid = np.zeros_like(solid_m)
    arho_fluid = np.zeros_like(fluid_m)
    acc_fluid = np.zeros_like(fluid_x)
    dxdt_fluid = np.zeros_like(fluid_x)
    # for solid particles
    for i in prange(len(solid_m)):
        sum_rho = 0

        # fluid source
        for j in prange(len(fluid_m)):
            xij = solid_x[i] - fluid_x[j]
            rij = norm(xij)

            if rij < 2 * h:
                vij = solid_v[i] - fluid_v[j]
                csd = np.array([0.0, 0.0])

                q = rij/h
                a = float(10.0/(7.0*np.pi*(h**2)))

                if q <= 1.0 and q >= 0.0:
                    if not rij == 0.0:
                        csd = a*((-3.0*q) + 2.25*(q**2))*(xij/rij)*(1/h)
                    else:
                        csd = np.array([0.0, 0.0])

                elif q <= 2.0:
                    csd = (-0.75*a*(2-q)**2)*(xij/rij)*(1/h)

                dot = vij[0]*csd[0] + vij[1]*csd[1]

                sum_rho = sum_rho + fluid_m[j] * dot / fluid_rho[j]

        arho_solid[i] = solid_rho[i] * sum_rho

    # for fluid particles
    for i in prange(len(fluid_x)):
        sum_rho = 0
        sum_vel = np.array([0.0, 0.0])
        sum_x = np.array([0.0, 0.0])

        # solid source
        for j in prange(len(solid_x)):
            xij = fluid_x[i] - solid_x[j]
            rij = norm(xij)

            vij = fluid_v[i] - solid_v[j]
            rhoij = 0.5 * (fluid_rho[i] + solid_rho[j])

            muij = h * np.dot(vij, xij) / (rij**2 + eta**2)
            pi_ij = 0

            if muij > 0:
                pi_ij = 0
            else:
                cij = c0
                pi_ij = -aplha * cij * muij / rhoij

            pj_rhoj = solid_p[j] / (solid_rho[j]**2)
            pi_rhoi = fluid_p[i] / (fluid_rho[i]**2)

            if rij < 2 * h:

                csd = np.array([0.0, 0.0])
                cs = 0.0

                q = rij/h
                a = float(10.0/(7.0*np.pi*(h**2)))

                if q <= 1.0 and q >= 0.0:
                    if not rij == 0.0:
                        csd = a*((-3.0*q) + 2.25*(q**2))*(xij/rij)*(1/h)
                    else:
                        csd = np.array([0.0, 0.0])
                elif q <= 2.0:
                    csd = (-0.75*a*(2-q)**2)*(xij/rij)*(1/h)

                if q <= 1.0 and q >= 0.0:
                    cs = a*(1.0 - 1.5*(q**2.0) + 0.75*(q**3.0))
                elif q <= 2.0:
                    cs = 0.25 * a * (2.0-q)**3.0

                dot = vij[0]*csd[0] + vij[1]*csd[1]

                sum_rho = sum_rho + solid_m[j] * dot / solid_rho[j]
                sum_x = sum_x + solid_m[j] * vij * cs / rhoij
                sum_vel = sum_vel + (solid_m[j] * (pi_rhoi+pj_rhoj+pi_ij) * csd)


        # fluid source
        for j in prange(len(fluid_x)):
            xij = fluid_x[i] - fluid_x[j]
            rij = norm(xij)

            vij = fluid_v[i] - fluid_v[j]
            rhoij = 0.5*(fluid_rho[i]+fluid_rho[j])

            muij = h * np.dot(vij, xij) / (rij**2+eta**2)
            pi_ij = 0

            if muij > 0:
                pi_ij = 0
            else:
                cij = c0
                pi_ij = -aplha * cij * muij / rhoij

            pi_rhoi = fluid_p[i] / (fluid_rho[i]**2)
            pj_rhoj = fluid_p[j] / (fluid_rho[j]**2)

            if rij < 2 * h:

                csd = np.array([0.0, 0.0])
                cs = 0.0

                q = rij/h
                a = float(10.0/(7.0*np.pi*(h**2)))

                if q <= 1.0 and q >= 0.0:
                    if not rij == 0.0:
                        csd = a*((-3.0*q) + 2.25*(q**2))*(xij/rij)*(1/h)
                    else:
                        csd = np.array([0.0, 0.0])
                elif q <= 2.0:
                    csd = (-0.75*a*(2-q)**2)*(xij/rij)*(1/h)

                if q <= 1.0 and q >= 0.0:
                    cs = a*(1.0 - 1.5*(q**2.0) + 0.75*(q**3.0))
                elif q <= 2.0:
                    cs = 0.25 * a * (2.0-q)**3.0

                dot = vij[0]*csd[0] + vij[1]*csd[1]
                sum_rho = sum_rho + fluid_m[j] * dot / fluid_rho[j]
                sum_x = sum_x + fluid_m[j] * vij * cs / rhoij
                sum_vel = sum_vel + (fluid_m[j] * (pi_rhoi+pj_rhoj+pi_ij) * csd)

        arho_fluid[i] = fluid_rho[i] * sum_rho
        acc_fluid[i] = (-1 * sum_vel) - np.array([0, g])
        dxdt_fluid[i] = fluid_v[i] - (epsilon * sum_x)

    solid_rho = solid_rho + (dt * arho_solid)
    solid_p = B * (((solid_rho/rho0)**gamma) - 1)

    for i in prange(len(solid_p)):
        if solid_p[i] < 0:
            solid_p[i] = 0

    fluid_rho = fluid_rho + arho_fluid * dt
    fluid_v = fluid_v + acc_fluid * dt
    fluid_x = fluid_x + dxdt_fluid * dt
    fluid_p = B * (((fluid_rho/rho0)**gamma) - 1)

    return solid_m, solid_p,  solid_rho, solid_x, solid_v, fluid_m, fluid_p, fluid_rho, fluid_x, fluid_v


if __name__ == "__main__":
    rho0 = 1000
    g = 9.81
    gamma = 7
    container_width = 4
    container_height = 4
    layers = 3
    fluid_height = 2
    fluid_width = 1
    aplha = 0.1
    beta = 0
    dx = 0.1
    eta = 1e-4
    epsilon = 0.3
    hdx = 1.3
    c0 = 10 * np.sqrt(2 * g * fluid_height)
    B = (c0**2)*rho0/gamma
    h = hdx * dx

    sx, sy, fx, fy = create_particles()
    sx1 = np.vstack((sx, sy)).T
    fx1 = np.vstack((fx, fy)).T

    solid_m = np.ones_like(sx)*(dx**2)*rho0
    solid_p = np.zeros_like(sx)
    solid_rho = np.ones_like(sx)*rho0
    solid_x = sx1
    solid_v = np.zeros((len(sx), 2))

    fluid_m = np.ones_like(fx)*(dx**2)*rho0
    fluid_p = np.zeros_like(fx)
    fluid_rho = np.ones_like(fx)*rho0
    fluid_x = fx1
    fluid_v = np.zeros((len(fx), 2))

    dt = 0.5 * 0.125 * h / c0
    tf = 2
    t = np.arange(0, tf, dt)

    for i in prange(len(t)):
        a = time.time()
        solid_m, solid_p,  solid_rho, solid_x, solid_v, fluid_m, fluid_p, fluid_rho, fluid_x, fluid_v = update_values(dt, solid_m, solid_p,  solid_rho, solid_x, solid_v, fluid_m, fluid_p, fluid_rho, fluid_x, fluid_v)

        if i%50 == 0:
            np.savez("data" + "f" + str(i) + ".npz", fluid_x)
            np.savez("data" + "s" + str(i) + ".npz", solid_x)
            np.savez("data" + "fv" + str(i) + ".npz", fluid_v)
            np.savez("data" + "fp" + str(i) + ".npz", fluid_p)
            np.savez("data" + "frho" + str(i) + ".npz", fluid_rho)
            np.savez("data" + "srho" + str(i) + ".npz", solid_rho)
            print("Saved")
        b = time.time()
        print("time", i, len(t), b-a, "s")
