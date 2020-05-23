import numpy
import scipy.stats
import matplotlib.pyplot as mp

# analysis of finite difference scheme for a call option under BS

def BS_call(S,K,r,s,T,t): # black-scholes call price
    
    d1 = (numpy.log(S/K)+(r+s**2/2)*(T-t))/(s*numpy.sqrt(T-t))
    d2 = d1 - s*numpy.sqrt(T-t)
    
    return (S*scipy.stats.norm.cdf(d1)-
            K*scipy.stats.norm.cdf(d2)*numpy.exp(-r*(T-t)))
    
def FDM_call(S,K,r,s,T,t): # FDM explicit method, backwards in time
                           # as payoff is a terminal condition
    
    N_x = len(S)
    N_t = len(t)
    Dt = numpy.diff(t)[0]
    Dx = numpy.diff(x)[0]
    
    def uT(x,K):
        return numpy.maximum(x-K,0)
    
    def g(vfw,v,vbw,x,Dx,r,s): # backward-recursive drift term 
        return r*x*(vfw-v)/Dx + 0.5*s**2*x**2*(vfw-2*v+vbw)/(Dx**2) - r*v
    
    FDM_bs = numpy.zeros((N_t,N_x))
    
    FDM_bs[0,:] = uT(x,K)
    FDM_bs[:,-1] = uT(x[-1],K) # boundary forcing:
                               # boundary error can be partially corrected
                               # by forcing the S boundary = payoff
    
    for k in range(1,N_t): # FDM backward difference equation
        for h in range(1,N_x-1): 
            FDM_bs[k,h] = FDM_bs[k-1,h] + Dt*g(FDM_bs[k-1,h+1],
                                                  FDM_bs[k-1,h],
                                                  FDM_bs[k-1,h-1],
                                                  x[h],Dx,r,s)
    return FDM_bs
    
r = 0.02
s = 0.15
T = 2
N_t = 16000 # N_t and N_x must be calibrated to ensure stability/convergence
N_x = 600
t = numpy.linspace(0,T,N_t)
x = numpy.linspace(0.01,300,N_x)

K = 100

FDM = FDM_call(x,K,r,s,T,t)

BS = numpy.zeros((N_t,N_x)) # true prices under BS assumptions
BS[0,:]  = numpy.maximum(x-K,0)
for k in range(1,N_t):
    BS[k,:] = BS_call(x,K,r,s,T,t[-k-1])
    
error = (BS-FDM)**2
    
# plot

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten() 

for k in range(1,int(N_t/1000)):        
    ax1.plot(x,FDM[int(k*1000),:],'r')
    ax1.plot(x,BS[int(k*1000),:],'k--')
ax1.plot(x,numpy.maximum(x-K,0),'b')
ax1.set_ylim((-2,30))
ax1.set_xlim((50,130))
ax1.set_xlabel('S')
ax1.set_ylabel('V(x)')
ax2.imshow(error,cmap=mp.cm.inferno,aspect='auto',
           extent=(x[0],x[-1],t[0],t[-1]),
           interpolation='none')
ax2.set_title('Squared error') 
ax2.set_xlabel('S')
ax2.set_ylabel('t')
mp.show()
    


