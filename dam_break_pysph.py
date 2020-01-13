import numpy as np

from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.sph.integrator import Integrator
from pysph.sph.integrator import IntegratorStep
from pysph.solver.application import Application
from pysph.sph.equation import Group

class EulerIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)


class EulerStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_x, d_y,
                  d_z, d_rho, d_arho, d_ux, d_vx, dt=0.0):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]

        d_rho[d_idx] += dt*d_arho[d_idx]


class TaitEOS(Equation):
    def __init__(self, dest, sources=None,
                 rho0=1000.0, c0=1.0, gamma=7.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.B = rho0*c0*c0/gamma
        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.B * (tmp - 1.0)


class ContinuityEquation(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij

class MomentumEquation(Equation):
    def __init__(self, dest, sources,
                 alpha=0.1, beta=0, gy=-9.81, c=1):
        self.aplha = alpha
        self.beta = beta
        self.c = c
        self.g = gy
        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0
        d_av[d_idx] = 0


    def loop(self, d_idx, s_idx, d_au, d_av, DWIJ, VIJ, XIJ, d_p, s_p, d_h, RIJ, d_rho, s_rho, s_m, RHOIJ):

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1]
        muij = d_h[d_idx] * vijdotxij / (RIJ**2 + (1e-4)**2)
        piij = 0
        if muij < 0:
            piij = -self.aplha * self.c * muij / RHOIJ

        pi = d_p[d_idx] / d_rho[d_idx]**2
        pj = s_p[s_idx] / s_rho[s_idx]**2

        d_au[d_idx] += -s_m[s_idx]*(pi+pj+piij)*DWIJ[0]
        d_av[d_idx] += -s_m[s_idx]*(pi+pj+piij)*DWIJ[1]

    def post_loop(self, d_idx, d_av):
        d_av[d_idx] += self.g


class XSPHCorrection(Equation):
    def initialize(self, d_idx, d_ax, d_ay):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, WIJ, RHOIJ1, VIJ):
        tmp = -0.3* s_m[s_idx]*WIJ*RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]

    def post_loop(self, d_idx, d_ax, d_ay, d_u, d_v):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]


class DamBreak(Application):
    def initialize(self):
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
        self.dx = 0.1
        self.eta = 1e-4
        self.epsilon = 0.3
        self.hdx = 1.3
        self.c0 = 10 * np.sqrt(2 * self.g * self.fluid_height)
        self.B = (self.c0**2)*self.rho0/self.gamma
        self.h = self.hdx*self.dx

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

        fluid = get_particle_array(name="fluid", x=fluid_x, y=fluid_y, rho = self.rho0, m= self.rho0 *self.dx**2, h = self.h)
        solid = get_particle_array(name="solid", x=solid_x, y=solid_y, rho=self.rho0, m= self.rho0 * self.dx**2, h=self.h)


        for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'ux',
                     'vx', 'w0', 'x0', 'y0', 'z0'):
            solid.add_property(name)
            fluid.add_property(name)

        fluid.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm',
                              'h', 'p', 'pid', 'tag', 'gid'])

        solid.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm',
                              'h', 'p', 'pid', 'tag', 'gid'])

        return [fluid, solid]

    def create_equations(self):
        equations = [
            Group(equations =[
                TaitEOS(dest = 'fluid', sources=None, rho0 = self.rho0, c0 = self.c0, gamma = 7.0),
                TaitEOS(dest = 'solid', sources=None, rho0 = self.rho0, c0 = self.c0, gamma = 7.0)
            ], real = False),
            Group(equations=[
                ContinuityEquation(dest='solid', sources=['fluid']),
                ContinuityEquation(dest='fluid', sources=['solid', 'fluid']),

                MomentumEquation(dest='fluid', sources=['fluid', 'solid'], alpha= self.aplha,
                                 beta = self.beta, gy=-1*self.g, c=self.c0),
                XSPHCorrection(dest='fluid', sources=['fluid'])
            ]),
        ]

        return equations

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EulerIntegrator(fluid=EulerStep(),solid=EulerStep())

        dt = 0.125 * 0.125 * self.h / self.c0
        tf = 2
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=True,
                        cfl=0.05, n_damp=50)

        return solver

if __name__ == "__main__":
    app = DamBreak()
    app.run()

