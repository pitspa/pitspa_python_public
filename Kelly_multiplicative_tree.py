import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.animation

# Kelly problem with the multiplicative tree: maximize the log return on 
# wealth W(Dt) given that W(0)=1 and a fraction A is invested in the risky asset
# and a fraction (1-A) in the riskless rate. The optimal A* maximizes the
# expected log-return

def logW(a,s,mu,r,Dt):
    u = numpy.exp(s*numpy.sqrt(Dt))
    d = 1/u
    p = (numpy.exp(mu*Dt)-d)/(u-d)
    return p*numpy.log(a*u+(1-a)*numpy.exp(r*Dt))+(1-p)*numpy.log(a*d+(1-a)*numpy.exp(r*Dt))

def op(a): # objective function for numerical optimization
    return -logW(a,s,mu,r,Dt)

def a_opt(s,mu,r,Dt): # analytical optimization solution
    u = numpy.exp(s*numpy.sqrt(Dt))
    d = 1/u
    #p = (numpy.exp(mu*Dt)-d)/(u-d)
    return numpy.exp(r*Dt)*(numpy.exp(mu*Dt)-numpy.exp(r*Dt))/((-numpy.exp(r*Dt)+u)*(numpy.exp(r*Dt)-d))
        #numpy.exp(r*Dt)*(numpy.exp(r*Dt)-d+p*(d-u))/((numpy.exp(r*Dt)-u)*(numpy.exp(r*Dt)-d))

s = 0.18
Dt = 0.001
mu = 0.12
r = 0.08


u = numpy.exp(s*numpy.sqrt(Dt))
d = 1/u

N_a = 1000
a = numpy.linspace(-1,2,N_a)

# numerical solution

max_ = scipy.optimize.minimize(op,x0=0.5)
a_numerical = max_['x']

# exact solution:
    
a_opt_sol =  a_opt(s,mu,r,Dt)
max_logW = logW(a_opt_sol,s,mu,r,Dt)

print(a_opt_sol, a_numerical)
print((mu-r)/s**2) # limit for Dt -> 0
print(mu-r)
print(s**2)

mp.plot(a,numpy.full(shape=N_a,fill_value=0),'k--')
mp.axvline(x=0,ymin=-100,ymax=100,color='k')
mp.axvline(x=1,ymin=-100,ymax=100,color='k')
mp.plot(a,logW(a,s,mu,r,Dt),label='log-return')
mp.plot(a,numpy.full(shape=N_a,fill_value=max_logW),'r')
mp.axvline(x=a_opt_sol,ymin=-100,ymax=100,color='r',label='a*')

mp.xlim((a[0],a[-1]))
mp.xlabel('a')
mp.legend()
mp.show()



