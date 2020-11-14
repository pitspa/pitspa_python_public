import numpy
import scipy.stats
import matplotlib.pyplot as mp

# probability mass function of the multiplicative binomial tree asset model

def tree_outcomes(x0,u,n): # grid of admissible events at nth step given x0
    d = 1/u
    k_r = numpy.arange(0,n+1)
    return x0*u**(k_r)*d**(n-k_r)

def tree_binomial(p,n): # probability mass at nth step
    k_r = numpy.arange(0,n+1)
    return scipy.stats.binom.pmf(k_r,n,p)

def m_moment(m,x0,n,p,u): # moments of the asset at nth step
    return x0**m*(p*u**m+(1/u)**m*(1-p))**n

def Ptp(x,m,s,t,X0): # proper lognormal distribution
    return ((1/(x*numpy.sqrt(2*numpy.pi*s**2*t)))*
                  numpy.exp(-0.5*((numpy.log(x/X0)-(m-0.5*s**2)*t)/
                                                  (s*numpy.sqrt(t)))**2))
T = 3 # maturity
n_ = 10000 # steps
Dt = T/n_ # step size

mu = 0.05 # GBM drift
s = 0.15 # GBM diffusion coeff.

x0 = 10 # initial condition
u = numpy.exp(s*numpy.sqrt(Dt)) # up factor
p = (numpy.exp(mu*Dt)-(1/u))/(u-(1/u)) # P(U)

xs = tree_outcomes(x0,u,n_) 
pmf = tree_binomial(p,n_)
print(numpy.sum(pmf))

EV = m_moment(1,x0,n_,p,u)
V = m_moment(2,x0,n_,p,u)-m_moment(1,x0,n_,p,u)**2

# plot

mp.plot(xs,pmf,'.',markersize=1.5)
mp.xlim((EV-numpy.sqrt(V)*3,EV+numpy.sqrt(V)*5))
mp.show()

# measure comparison

asym_cdf = scipy.stats.lognorm.cdf(xs,s=s*numpy.sqrt(T),
                                   scale=numpy.exp(numpy.log(x0)+(mu-0.5*s**2)*T))
cdf = numpy.cumsum(pmf)

# plot

x = numpy.linspace(1e-13,300,50000)
dx = numpy.diff(x)[0]

mp.plot(xs,cdf)
mp.plot(xs,asym_cdf,'--')
mp.xlim((EV-numpy.sqrt(V)*3,EV+numpy.sqrt(V)*5))
mp.show()

# moments comparison

print(numpy.sum(tree_outcomes(x0,u,n_)*tree_binomial(p,n_)))
print(m_moment(1,x0,n_,p,u))
print(numpy.sum(x*Ptp(x,mu,s,T,x0))*dx)

print(numpy.sum(tree_outcomes(x0,u,n_)**2*tree_binomial(p,n_)))
print(m_moment(2,x0,n_,p,u))
print(numpy.sum(x**2*Ptp(x,mu,s,T,x0))*dx)

print(numpy.sum(tree_outcomes(x0,u,n_)**3*tree_binomial(p,n_)))
print(m_moment(3,x0,n_,p,u))
print(numpy.sum(x**3*Ptp(x,mu,s,T,x0))*dx)
