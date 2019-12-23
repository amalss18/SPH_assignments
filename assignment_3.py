import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(xij,h):
    rij = np.linalg.norm(xij)
    q = rij/h
    a = 10/(7*np.pi*h**2)
    if q<=1:
        return a*(1-(1.5*(q**2)*(1-0.5*q)))
    elif q<=2:
        return 0.25*a*(2-q)**3
    elif q>2:
        return 0

def cubic_spline_derivative(xij,h):
    rij = abs(xij)
    q = rij/h
    a = 10/(7*np.pi*h**2)
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

def EOS(e,rho):
    gamma = 1.4
    p = (gamma-1)*rho*e
    return p

def density_calc(m,x,y,h):
    rho = []
    for i in range(len(x)):
        sum = 0
        for j in range(len(x)):
            xij = np.zeros(2)
            xij[0] = x[i]-x[j]
            xij[1] = y[i]-y[j]
            sum += m[j]*(cubic_spline(xij,h[j])+cubic_spline(xij,h[i]))*0.5
        rho.append(sum)
    return rho


def continuity_eqaution(rho,m,v,h,x,y,dt):
    arho = np.zeros(rho)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = np.zeros(2)
            xab[0] = x[a]-x[b]
            xab[1] = y[a]-y[b]
            sum+=(m[b]/rho[b])*(v[a]-v[b])* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5
        arho[a]= (rho[a]*sum)

    rho_updated = integrate(rho,dt,arho)

    return rho_updated


def momentum_equation(rho,m,x,y,p,v,c,h,dt):
    acc = np.zeros(m)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = np.zeros(2)
            xab[0] = x[a]-x[b]
            xab[1] = y[a]-y[b]
            vab = np.zeros(2)
            vab[0] = v[a]-v[b]
            rhoab = (rho[a]+rho[b])*0.5
            cab = (c[a]+c[b])*0.5
            hab = (h[a]+h[b])*0.5

            sum+=(m[b])*((p[a]/rho[a]**2)+(p[b]/rho[b]**2)+artificial_viscosity(xab,vab,rhoab,cab,hab))* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5

        acc[a] = -1*sum

    vel_updated = integrate(v,dt,acc)

    return vel_updated

def energy_equation(rho,m,x,y,p,v,c,h,dt,e):
    ae = np.zeros(m)
    for a in range(len(m)):
        sum = 0
        for b in range(len(m)):
            xab = np.zeros(2)
            xab[0] = x[a]-x[b]
            xab[1] = y[a]-y[b]
            vab = np.zeros(2)
            vab[0] = v[a]-v[b]
            rhoab = (rho[a]+rho[b])*0.5
            cab = (c[a]+c[b])*0.5
            hab = (h[a]+h[b])*0.5

            sum+=(m[b])*((p[a]/rho[a]**2)+(p[b]/rho[b]**2)+artificial_viscosity(xab,vab,rhoab,cab,hab))* \
                (cubic_spline_derivative(xab,h[a])+cubic_spline_derivative(xab,h[b]))*0.5

        ae[a] = 0.5*sum

    e_updated = integrate(e,dt,ae)

    return e_updated



