import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def cubic_spline(xij,h):
	rij = abs(xij)
	q = rij/h
	a = float(2.0/(3.0*h))
	if q<=1.0 and q>=0.0:
		return a*(1.0-(1.5*(q**2.0)+(0.75*(q**3.0))))
	elif q<=2.0:
		return 0.25*a*(2.0-q)**3.0
	else:
		return 0.0

@njit
def cubic_spline_derivative(xij,h):
	rij = abs(xij)
	q = rij/h
	a = (2.0/(3.0*h))
	if q<=1.0 and q>=0.0:
		if not xij ==0.0 :
		    return float((a*((-3.0*q)+ 2.25*(q**2))/h)*np.sign(xij))
		else :
		    return 0.0
	elif q<=2.0:
		    return float(((-0.75*a*(2-q)**2)/h)*np.sign(xij))
	elif q>2.0:
		return 0.0


def create_particles(xmin,xmax,dx1,dx2):
	x1 = np.linspace(-0.5,0,320)
	x2 = np.linspace(0,0.5,40)[1:]

	x1_boundary = np.linspace(-0.5 - (35 * dx1), -0.5 - dx1, 35)
	x2_boundary = np.linspace(0.5 + dx2, 0.5 + (35 * dx2), 35)

	x1 = np.append(x1_boundary, x1)
	x2 = np.append(x2, x2_boundary)

	return x1, x2

@njit(parallel=True)
def intialize(x1,x2,dx1,dx2):
	rho1 = np.ones_like(x1)
	rho2 = np.ones_like(x2)*0.125
	rho = np.concatenate((rho1,rho2))

	p1 = np.ones_like(x1)
	p2 = np.ones_like(x2)*0.1
	p = np.concatenate((p1,p2))

	v1 = np.zeros_like(x1)
	v2 = np.zeros_like(x2)
	v = np.concatenate((v1,v2))

	h1 = 2*np.ones_like(x1)*dx2
	h2 = 2*np.ones_like(x2)*dx2
	h = np.concatenate((h1,h2))

	e = p/(rho*0.4)
	x = np.concatenate((x1,x2))
	return rho,p,v,h,e,x


@njit
def piab(rho,c,h,x,v):
	if x*v<0:
		mu = h*x*v/(x**2+(1e-4)**2)
		pi = (-1*c*mu+mu**2)/rho
		return pi
	else:
		return 0

@njit(parallel=True)
def update_values(p,rho,v,e,x,m,dt,h):
	arho = np.zeros_like(m)
	acc = np.zeros_like(m)
	ae = np.zeros_like(m)
	xsph = np.zeros_like(m)

	for a in range(35,len(m)-35):
		sum_rho = 0
		sum_v = 0
		sum_e = 0
		for b in range(len(m)):
			vab = v[a]-v[b]
			xab = x[a]-x[b]
			cab = 0.5*(np.sqrt(1.4*p[a]/rho[a])+np.sqrt(1.4*p[b]/rho[b]))
			rhoab = 0.5*(rho[a]+rho[b])
			hab = 0.5*(h[a]+h[b])
			p_rho_a = p[a]/(rho[a]**2)
			p_rho_b = p[b]/(rho[b]**2)


			sum_rho+= m[b]*vab*cubic_spline_derivative(xab,hab)/rho[b]
			sum_v += m[b]*(p_rho_a+p_rho_b+piab(rhoab,cab,hab,xab,vab))*cubic_spline_derivative(xab,hab)
			sum_e += m[b]*(p_rho_a+p_rho_b+piab(rhoab,cab,hab,xab,vab))*cubic_spline_derivative(xab,hab)*vab

		arho[a] = sum_rho*rho[a]
		acc[a] = sum_v*-1
		ae[a] = 0.5*sum_e

	rho_new = rho+(dt*arho)
	v_new = v+(dt*acc)
	e_new = e + (dt*ae)
	x_new = x+(dt*(v))

	p_new = np.sqrt(0.4*rho_new*e_new)

	return p_new, rho_new, v_new, e_new, x_new

if __name__=="__main__":

	xmin, xmax= -0.5,0.5
	dx1, dx2 = 0.0015625,0.0125
	x1, x2 = create_particles(xmin,xmax,dx1,dx2)

	tf = 0.2
	dt = 0.5e-4

	l = len(x1)+len(x2)
	n = int(tf/dt)+1

	p_hist = np.zeros((n,l),dtype=("float32","float32"))
	rho_hist = np.zeros((n,l),dtype=("float32","float32"))
	v_hist = np.zeros((n,l),dtype=("float32","float32"))
	x_hist = np.zeros((n,l),dtype=("float32","float32"))
	e_hist = np.zeros((n,l),dtype=("float32","float32"))
	m = np.ones(l)*0.0015625000000000

	rho_hist[0], p_hist[0], v_hist[0], h, e_hist[0], x_hist[0] = intialize(x1,x2,dx1,dx2)

	time = np.arange(0,tf,dt)
	time = time+dt

	for i in range(n-1):
		print(time[i],i)
		p_hist[i+1],rho_hist[i+1],v_hist[i+1],e_hist[i+1],x_hist[i+1] = \
		update_values(p_hist[i],rho_hist[i],v_hist[i],e_hist[i],x_hist[i],m,dt,h)

	# plt.figure(1)
	# plt.xlabel("x")
	# plt.ylabel("rho")
	# plt.scatter(x_hist[-1],rho_hist[-1])
	# plt.savefig("rho.png")

	# plt.figure(2)
	# plt.xlabel("x")
	# plt.ylabel("e")
	# plt.scatter(x_hist[-1],e_hist[-1])
	# plt.savefig("e.png")

	# plt.figure(3)
	# plt.xlabel("x")
	# plt.ylabel("p")
	# plt.scatter(x_hist[-1],p_hist[-1])
	# plt.savefig("p.png")

	plt.figure(4)
	plt.xlabel("x")
	plt.ylabel("v")
	plt.scatter(x_hist[-1],v_hist[-1])
	plt.savefig("v.png")