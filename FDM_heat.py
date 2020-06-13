import numpy
import matplotlib.pyplot as mp
import matplotlib.animation

# a solution to the heat PDE using FDM

def u0(x):
    
    """
    returns the initial condition in the interval x=[0,a].
    parameters:
        x: spatial range interval
    """
    
    return numpy.sin(numpy.pi*x)**3*numpy.exp(-x**2*50)

N_x = 200
x = numpy.linspace(0,1,N_x)
Dx = numpy.diff(x)[0]

# 1) Finite difference explicit method

Dt = 0.3*Dx**2 # Dt is chosen to ensure stability/convergence
t = numpy.arange(0,0.1,Dt)
N_t = len(t)

r = Dt/(Dx**2)
a = 0.9 # diffusion parameter

FDM_u = numpy.zeros((N_t,N_x))
FDM_u[0,:] = u0(x) # initial condition
for t_ in range(0,N_t-1):
    for x_ in range(1,N_x-1): # boundaries u(0,t) = 0 ; u(1,t) = 0
        FDM_u[t_+1,x_] = (a*(Dt/(Dx**2))*(FDM_u[t_,x_+1]-
                                   2*FDM_u[t_,x_]+FDM_u[t_,x_-1])
                                + FDM_u[t_,x_])
    print(t[t_])
    

# 2) integral approximation of the heat PDE solution

tc = numpy.arange(1,N_t,N_t/40) 

u = numpy.zeros((len(tc),N_x))
u[0,:] = u0(x)
n = numpy.arange(1,50,1) # cut off the orthonormal basis sequence

cn0 = numpy.zeros(len(n)) # sine transform with integral approximation
for n_ in range(0,len(n)):
    cn0[n_] = numpy.sum(u[0,:]*numpy.sin(numpy.pi*n[n_]*x))*Dx*numpy.sqrt(2)

fig2 = mp.figure()# initialize animation
ims = []

for t_ in range(0,len(tc)):
    for x_ in range(0,N_x): 
        u[t_,x_] = (numpy.matmul(cn0,numpy.sin(numpy.pi*n*x[x_])*
                         numpy.exp(-a*(n*numpy.pi)**2*t[int(tc[t_])]))
                        *numpy.sqrt(2))
    gee = mp.plot(x,u[t_,:],'r',x,FDM_u[int(tc[t_]),:],'k--')
    ims.append(gee)
    
# animation of diffusion

im_ani = matplotlib.animation.ArtistAnimation(
        fig2, ims, interval=100, repeat=True, blit=True)

mp.ylim((-0.005,0.03))
mp.xlim((x[0],x[-1]))
mp.show()
