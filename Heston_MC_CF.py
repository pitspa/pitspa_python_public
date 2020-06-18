import numpy
import scipy.stats
import matplotlib.pyplot as mp

# Use Feynman-Kac to estimate the Heston CF through Monte Carlo sampling
# of simulations at maturity; then use numerical Fourier transform of the
# smoothed CF to obtain an estimate of the PDF P(lnXT|X0,v0)

T = 1 # maturity
eta = 0.09 # vol of vol 
lam = 0.5 # mean reversion speed
rho = -0.8 # correlation
v_ = 0.04 # long term volatility

# initial conditions:

v0 = 0.01 
lnX0 = 0

I = 5000 # number of simulations
N_t = 1000
t = numpy.linspace(0,T,N_t) # discretized time range
dt = numpy.diff(t)[0]

# Simulate Heston log-price paths and variance paths

lnX = numpy.zeros((N_t,I))
vt = numpy.zeros((N_t,I))
lnX[0,:] = lnX0
vt[0,:] = v0
for i in range(0,N_t-1): # Euler scheme with variance correction
    W1, W2 = (scipy.stats.norm.rvs(size=I,scale=numpy.sqrt(dt)) for i in range(0,2))
    vt[i+1,:] = numpy.absolute(vt[i,:] + lam*(v_-vt[i,:])*dt + eta*numpy.sqrt(vt[i,:])*(rho*W1+numpy.sqrt(1-rho**2)*W2))
    lnX[i+1,:] = lnX[i,:] + numpy.sqrt(vt[i,:])*W1
    

N_u = 1000
u = numpy.linspace(-200,200,N_u) # discretized frequency range
du = numpy.diff(u)[0]

# Monte Carlo sampling of the characteristic function at maturity

exp = numpy.zeros((I,N_u),dtype=numpy.complex) # payoff function e^(-iuX(T))
Eexp = numpy.zeros(N_u,dtype=numpy.complex)
for i in range(0,N_u):
    exp[:,i] = numpy.exp(-1j*u[i]*lnX[-1,:]) # convention
    Eexp[i] = numpy.average(exp[:,i]) # CF estimator

tap = numpy.exp(-u**2/4000) # window function
Eexptap = Eexp*tap # the taper smoothes the sample CF 
                    # by filtering out high frequencies

N_x = 2000
lnx = numpy.linspace(-numpy.pi/du,numpy.pi/du,N_x) # discretized inverse-freq (log price) range
dx = numpy.diff(lnx)[0]

xmesh, umesh = numpy.meshgrid(lnx, u)
D = numpy.exp(1j*xmesh*umesh)
 
P = numpy.real(numpy.matmul((D).T,Eexptap)/(2*numpy.pi)*du) # discrete CFT
print(numpy.sum(P)*dx)

# Compare with the numerical Fourier inversion of the exact Heston pdf

def phi(u,T,eta,lam,rho,v_,v0):
    
    """
    returns the exact Heston characteristic function in a range
    of frequencies.
    parameters:
        u: the frequency range
        T: maturity
        eta: vol of vol
        lam: mean reversion speed
        rho: correlation
        v_: long term volatility
        v0: initial volatility condition 
    """
    
    a = -u**2/2 - 1j*u/2
    b = lam - 1j*rho*eta*u
    gam = eta**2/2
    d = numpy.sqrt(b**2 - 4*a*gam)
    rplus = (b + d)/(2*gam)
    rminus = (b - d)/(2*gam)
    g = rminus/rplus
    
    D = rminus*(1-numpy.exp(-d*T))/(1-g*numpy.exp(-d*T))
    C = lam*(rminus*T-2/(eta**2)*numpy.log((1-g*numpy.exp(-d*T))/(1-g)))
    
    return numpy.exp(C*v_+D*v0)

def pdf(x,u,T,eta,lam,rho,v_,v0):
    
    """
    approximation of the inverse continuous Fourier transform of the
    Heston characteristic function.
    parameters:
        x: the inverse-frequency range
        u: the frequency range
        T: maturity
        eta: vol of vol
        lam: mean reversion speed
        rho: correlation
        v_: long term volatility
        v0: initial volatility condition 
    """
    
    
    cf = phi(u,T,eta,lam,rho,v_,v0)
    du = numpy.diff(u)[0]
    
    M = len(x) 
    pdf = numpy.zeros(M,dtype=numpy.complex)
    for h in range(0,M): # CFT discretization
        pdf[h] = numpy.sum(cf*numpy.exp(-1j*x[h]*u))*du/(2*numpy.pi)
    
    return numpy.real(pdf)

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

ax1.plot(u,tap,'y--',label='window function')

ax1.plot(u,numpy.imag(Eexp),'r',label='Imag(MC_CF)')
ax1.plot(u,numpy.real(Eexp),'b',label='Real(MC_CF)')

ax1.plot(u,numpy.real(Eexptap),'k',label='tapered CF')
ax1.plot(u,numpy.imag(Eexptap),'k')
ax1.legend()
ax1.set_xlim((-150,150))

ax2.plot(lnx,pdf(lnx,u,T,eta,lam,rho,v_,v0),'k--',label='pdf')
ax2.plot(lnx,P,'b',label='MC pdf')
ax2.set_xlim((-1,1))
ax2.legend()
mp.show()



    
