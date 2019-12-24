import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit

@njit
def cubic_spline(xij,h):
    rij = abs(xij)
    q = rij/h
    a = 2/(3*h)
    if q<=1:
        return a*(1-(1.5*(q**2)*(1-0.5*q)))
    elif q<=2:
        return 0.25*a*(2-q)**3
    elif q>2:
        return 0

@njit
def cubic_spline_derivative(xij,h):
    rij = abs(xij)
    q = rij/h
    a = 2/(3*h)
    if q<=1:
        if not xij ==0 :
            return -1.5*a*((-(0.5*q**2)/h)+(2*q/h)*(1-0.5*q))*(abs(xij)/xij)
        else :
            return -1.5*a*((-(0.5*q**2)/h)+(2*q/h)*(1-0.5*q))
    elif q<=2:
        if not xij==0:
            return (-0.75*a/h*(2-q)**2)*(abs(xij)/xij)
        else:
            return (-0.75*a/h*(2-q)**2)
    elif q>2:
        return 0

@njit
def EOS(e,rho):
    gamma = 1.4
    p = (gamma-1)*rho*e
    return p

def density_calc(m,x,h):
    rho = []
    for i in range(len(x)):
        sum = 0
        for j in range(len(x)):
            xij = x[i]-x[j]
            sum += m[j]*(cubic_spline(xij,h[j])+cubic_spline(xij,h[i]))*0.5
        rho.append(sum)
    return rho

@njit(parallel=True)
def integrate(intial,dt,r_change):
    final = intial+dt*r_change
    return final

@njit(parallel=True)
def continuity_eqaution(rho,m,v,h,x,dt):
    arho = np.zeros_like(rho)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = x[a]-x[b]
            sum+=(m[b]/rho[b])*(v[a]-v[b])* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5
        arho[a]= (rho[a]*sum)

    rho_updated = integrate(rho,dt,arho)

    return rho_updated

@njit(parallel=True)
def artificial_viscosity(xab,vab,rhoab,cab,hab,eta):
    if vab*xab<0:
        muab = hab*vab*xab/(abs(xab)**2+eta**2)

        piab = (-cab*muab + muab**2)/rhoab
        return piab

    else:
        return 0

@njit(parallel=True)
def momentum_equation(rho,m,x,p,v,c,h,dt,eta):
    acc = np.zeros_like(m)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = x[a]-x[b]
            vab = v[a]-v[b]
            rhoab = (rho[a]+rho[b])*0.5
            cab = c
            hab = (h[a]+h[b])*0.5

            sum+=(m[b])*((p[a]/rho[a]**2)+(p[b]/rho[b]**2)+artificial_viscosity(xab,vab,rhoab,cab,hab,eta))* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5

        acc[a] = -1*sum

    vel_updated = integrate(v,dt,acc)
    dvdt = (vel_updated-v)/dt

    x_updated = integrate(x,dt,dvdt)

    return vel_updated, x_updated

@njit(parallel=True)
def energy_equation(rho,m,x,p,v,c,h,dt,e,eta):
    ae = np.zeros_like(m)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = x[a]-x[b]
            vab = v[a]-v[b]
            rhoab = (rho[a]+rho[b])*0.5
            cab = c
            hab = (h[a]+h[b])*0.5

            sum+=(m[b])*((p[a]/rho[a]**2)+(p[b]/rho[b]**2)+artificial_viscosity(xab,vab,rhoab,cab,hab,eta))* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5

        ae[a] = 0.5*sum

    e_updated = integrate(e,dt,ae)

    return e_updated

def create_particles(xmin,xmax,dx1,dx2):
    x1 = np.arange(xmin,0,dx1)
    x2 = np.arange(0,xmax,dx2)
    return x1,x2

@njit(parallel=True)
def intialize(x1,x2,dx1,dx2):
    rho1 = np.ones_like(x1)
    rho2 = np.ones_like(x2)*0.125
    rho = np.concatenate((rho1,rho2))

    p1 = np.ones_like(x1)
    p2 = np.ones_like(x2)*0.1
    p = np.concatenate((p1,p2))

    v1 = np.zeros_like(x1)
    v2 = np.zeros_like(x2)*0
    v = np.concatenate((v1,v2))

    h1 = np.ones_like(x1)*dx1
    h2 = np.ones_like(x2)*dx2
    h = np.concatenate((h1,h2))

    e = p/(rho*0.4)
    x = np.concatenate((x1,x2))
    return rho,p,v,h,e,x


if __name__ == "__main__":
    xmin, xmax= -0.5,0.5
    dx1, dx2 = 0.0015625,0.0125
    x1, x2 = create_particles(xmin,xmax,dx1,dx2)

    tf = 0.2
    dt = 0.8e-4


    l = len(x1)+len(x2)
    n = int(tf/dt)+1

    p_hist = np.zeros((n,l))
    rho_hist = np.zeros((n,l))
    v_hist = np.zeros((n,l))
    x_hist = np.zeros((n,l))
    e_hist = np.zeros((n,l))
    m = np.zeros(l)

    rho_hist[0], p_hist[0], v_hist[0], h, e_hist[0], x_hist[0] = intialize(x1,x2,dx1,dx2)

    for i in range(l):
        if i<=len(x1):
            m[i] = rho_hist[0][i]*dx1
        else:
            m[i] = rho_hist[0][i]*dx2

    eta = 0.01*min(h)
    c = 330

    time = np.arange(0,tf,dt)
    time = time+dt

    for i in range(n-1):
        print(time[i])
        p_hist[i+1] = EOS(e_hist[i],rho_hist[i])
        rho_hist[i+1] = continuity_eqaution(rho_hist[i],m,v_hist[i],h,x_hist[i],dt)
        v_hist[i+1], x_hist[i+1] = momentum_equation(rho_hist[i],m,x_hist[i],p_hist[i],v_hist[i],c,h,dt,eta)
        e_hist[i+1] = energy_equation(rho_hist[i],m,x_hist[i],p_hist[i],v_hist[i],c,h,dt,e_hist[i],eta)

plt.plot(rho_hist[-1],x_hist[-1])
plt.show()




