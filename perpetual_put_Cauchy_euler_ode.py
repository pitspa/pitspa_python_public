import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

# The perpetual put in BS is priced by a Cauchy-Euler second order ODE
# with constant vol and rf rate in the form:
#
# 0.5s**2x**2dV/dx + rxdV/dx - rV = 0
#
# Roots study:

def D(r,s):
    return  ((r-0.5*s**2)**2+2*s**2*r)

def m1(r,s):
    return (- (r-0.5*s**2) + D(r,s)**(0.5))/(s**2)

def m2(r,s):
    return (- (r-0.5*s**2) - D(r,s)**(0.5))/(s**2)

N = 41
r_range = numpy.linspace(-0.2,0.2,N)
s_range = numpy.linspace(0.05,0.25,N)

rmesh, smesh = numpy.meshgrid(r_range, s_range)

Ds = D(rmesh, smesh)
m1s = m1(rmesh,smesh)
m2s = m2(rmesh,smesh)

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

ax1.imshow(m1s, cmap=mp.cm.viridis,
           aspect='auto',interpolation='none',
           extent=[r_range[0],r_range[-1],s_range[-1],s_range[0]])
ax1.set_ylabel('sigma')
ax1.set_xlabel('rf rate')
ax2.imshow(m2s, cmap=mp.cm.viridis,
           aspect='auto',interpolation='none',
           extent=[r_range[0],r_range[-1],s_range[-1],s_range[0]])
ax2.set_xlabel('rf rate')
mp.show()

# case r > 0, the solution is:

def V(r,s,x,K): 
    
    S_ = K*m2(r,s)/(m2(r,s)-1)
    B = (K-S_)/(S_**m2(r,s))
    
    if isinstance(x, numpy.float64) == True:
        
        S_ = K*m2(r,s)/(m2(r,s)-1)
        B = (K-S_)/(S_**m2(r,s))

        if x == S_: # force boundaries
            V_ = K - x
        #elif x < S_: # we show prices also for x < x_
        #    V_ = numpy.nan
        else:
            V_ = B*x**m2(r,s)
            
        if V_ > K:
            V_ = numpy.nan

    else:
        
        V_ = numpy.zeros(len(x))
        for i in range(0,len(x)): 
            if x[i] == S_:
                V_[i] = K - x[i]
            #elif x[i] < S_:
            #    V_[i] = numpy.nan
            else:
                V_[i] = B*x[i]**m2(r,s)
            if V_[i] > K:
                V_[i] = numpy.nan
              
    return V_, S_

M = 1000
K = 120
dx = 0.05
x = numpy.arange(dx,400,dx)
r = 0.01
s = numpy.array([0.1,0.2,0.3,0.4])


# check the delta of the option with finite difference:

def dVdx(r,s,x,K): # closed form delta
    
    S_ = K*m2(r,s)/(m2(r,s)-1)
    B = (K-S_)/(S_**m2(r,s))
    
    dV_ = numpy.zeros(len(x))
    for i in range(0,len(x)): # force boundaries
        if x[i] == S_:
            dV_[i] = -1
        #elif x[i] < S_:
        #    dV_[i] = numpy.nan
        else:
            dV_[i] = B*m2(r,s)*x[i]**(m2(r,s)-1)
          
    return dV_, S_

f_, axarr = mp.subplots(1, 2, sharex=False)
ax1, ax2 = axarr.flatten()

for sig in s:
    Vp, S_ = V(r,sig,x,K)
    ax1.plot(x,Vp)
ax1.plot(x,numpy.maximum(K-x,0),'k')
ax1.legend(s,title='sigma =')
ax1.set_ylim((-5,K))
ax1.set_xlim((x[0],K+150))
ax1.set_xlabel('x')
ax1.set_ylabel('V(x)')

s = numpy.array([0.2,0.3,0.4,0.5])
for sig in s:
    dVp, S_ = dVdx(r,sig,x,K)
    ax2.plot(x,dVp)
ax2.legend(s,title='sigma =')
for sig in s:
    Vp, S_ = V(r,sig,x,K)
    ax2.plot(x,numpy.append(numpy.diff(Vp)[0]/dx,numpy.diff(Vp)/dx),'k--')
ax2.set_xlabel('x')
ax2.set_ylabel('dV(x)/dx')
ax2.set_ylim((-1,0))
ax2.set_xlim(x[0],K+150)
mp.show()


# simulate paths
# in the case of a discrete simulation, P(x=x_)=0 as x_ has a continuous
# conditional pdf. Therefore the exercise is approximated by the condition
# of crossing the optimal exercise given x(0) > x_

N = 200
I = 200
T = 15
t = numpy.linspace(0,T,N)
dt = numpy.diff(t)[0]
s = 0.2

Xs = numpy.zeros((N,I))
Vs = numpy.zeros((N,I))
Xs[0,:] = 100
Vs[0,:], S_ = V(r,s,Xs[0],K)
for u in range(0,I):
    for h in range(1,N):
        Xs[h,u] = Xs[h-1,u]*numpy.exp((r-0.5*s**2)*dt+
                              s*numpy.sqrt(dt)*scipy.stats.norm.rvs())
        Vs[h,u], S_ = V(r,s,Xs[h,u],K)
        
f_, axarr = mp.subplots(1, 3, sharex=False)
ax1, ax2, ax3 = axarr.flatten()
        
Vop = []
Xop = []
strats = []
for u in range(0,30):
    for h in range(1,N):
        #if  numpy.isnan(Vs[h,u]) == True:
        if Vs[h,u] >= K - S_:
            Vs[h+1:,u] = numpy.nan
            Xs[h+1:,u] = numpy.nan
            Vop.append(Vs[:,u])
            Xop.append(Xs[:,u])
            strats.append(K-Xop[-1]-Vop[-1])
            ax1.plot(t,Vs[:,u])
            ax2.plot(t,Xs[:,u])
            ax3.plot(t,strats[-1])
            break 
        else:
            continue
   
ax1.plot(t,numpy.full(shape=N,fill_value= K - S_),'k--')
ax2.plot(t,numpy.full(shape=N,fill_value= S_),'k--')
ax3.plot(t,numpy.full(shape=N,fill_value= 0),'k--')
ax1.set_xlim((t[0],t[-1]))
ax2.set_xlim((t[0],t[-1]))
ax1.set_ylabel('V(x,t)')
ax2.set_ylabel('x(t)')
ax3.set_ylabel('(K-x(t))-V(t)')
ax1.set_xlabel('t')
ax2.set_xlabel('t')
ax3.set_xlabel('t')
mp.show()















