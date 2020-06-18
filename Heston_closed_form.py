import numpy
import scipy.stats
import matplotlib.pyplot as mp

# Numerical Fourier inversion of the exact Heston characteristic function
#
# thanks to https://mfe.baruch.cuny.edu/wp-content/uploads/2015/06/VW2.pdf
# for the characteristic function form written in R

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

T = 1
eta = 0.09 
lam = 0.5 
rho = -0.8 
v_ = 0.04 
v0 = 0.01 

N = 2000
u = numpy.linspace(-200,200,N)
du = numpy.diff(u)[0]

f = phi(u,T,eta,lam,rho,v_,v0)  

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
    
M = 2001
x = numpy.linspace(-2,2,M)
dx = numpy.diff(x)[0]

P = pdf(x,u,T,eta,lam,rho,v_,v0)

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

ax1.plot(u,numpy.real(f),'b')
ax1.plot(u,numpy.imag(f),'r')
ax2.plot(x,P,'k')
ax1.set_xlim((-50,50))
ax2.set_xlim((-1,1))
mp.show()

print(numpy.sum(P)*dx)




    

    
