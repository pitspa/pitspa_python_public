import numpy
import scipy.stats
import matplotlib.pyplot as mp

# black scholes PDE to heat equation

def a(s,r): # alpha parameter
    return (s**2-2*r)/(2*s**2)

def b(s,r): # beta parameter
    return -((s**2+2*r)/(2*s**2))**2

def Ker(x,x0,t): # heat kernel
    return numpy.exp(-(x0-x)**2/(4*t))/(numpy.sqrt(4*numpy.pi*t))

def BS_call(S,K,r,s,T,t):
    
    d1 = (numpy.log(S/K)+(r+s**2/2)*(T-t))/(s*numpy.sqrt(T-t))
    d2 = d1 - s*numpy.sqrt(T-t)
    
    return (S*scipy.stats.norm.cdf(d1)-
            K*scipy.stats.norm.cdf(d2)*numpy.exp(-r*(T-t)))
    
def BS_put(S,K,r,s,T,t):
    
    d1 = (numpy.log(S/K)+(r+s**2/2)*(T-t))/(s*numpy.sqrt(T-t))
    d2 = d1 - s*numpy.sqrt(T-t)
    
    return (-S*scipy.stats.norm.cdf(-d1)+
            K*scipy.stats.norm.cdf(-d2)*numpy.exp(-r*(T-t)))

N_x = 5000
x = numpy.linspace(-15,15,N_x) # the 'log-return'-like variable
dx = numpy.diff(x)[0]
S = numpy.exp(x) # S as a function of x

K = 100
s = 0.1
r = 0.01

T = 3
term_tau = T*s**2/2 
dtau = term_tau/6
tau = numpy.arange(dtau,term_tau,dtau) # the 'reversed time' variable
t = T-2*tau/s**2 # t as a function of tau

# check that the simple numerical integration of the HK convolution with
# the initial condition approximates the closed form solution

# Call and put options:

V_ts, trus, V_tsp, trusp = (numpy.zeros((N_x,len(tau))) for i in range(4))
for u in range(0,len(tau)):
    for i in range(0,N_x):
        ux_ = numpy.sum(numpy.maximum(S-K,0)*numpy.exp(-x*a(s,r))*Ker(x,x[i],tau[u]))*dx
        ux_p = numpy.sum(numpy.maximum(K-S,0)*numpy.exp(-x*a(s,r))*Ker(x,x[i],tau[u]))*dx
        V_ts[i,u] = ux_*numpy.exp(a(s,r)*x[i]+b(s,r)*tau[u])
        trus[i,u] = BS_call(S[i],K,r,s,T,T-2*tau[u]/s**2)
        V_tsp[i,u] = ux_p*numpy.exp(a(s,r)*x[i]+b(s,r)*tau[u])
        trusp[i,u] = BS_put(S[i],K,r,s,T,T-2*tau[u]/s**2)

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

ax2.plot(S,numpy.maximum(S-K,0)*numpy.exp(-r*T),'b') 
ax1.plot(S,numpy.maximum(-S+K,0)*numpy.exp(-r*T),'b')        
for u in range(0,len(tau)):
    ax2.plot(S,V_ts[:,u],'k') # numerical error since improper integral 
                             # is discretized and bounded
    ax2.plot(S,trus[:,u],'r--')
    ax1.plot(S,V_tsp[:,u],'k')
    ax1.plot(S,trusp[:,u],'r--')
ax2.set_xlim((80,120))
ax2.set_ylim((-1,20))
ax1.set_xlim((80,120))
ax1.set_ylim((-1,20))

mp.show()


    

    


