import numpy
import scipy.stats
import matplotlib.pyplot as mp

# dx(t) = th(mu-x(t))dt + s*dW(t)

th = 0.5 # mean reversion speed
mu = 0.3 # long term mean
x0 =  0.5 # initial value
s = 0.03 # error standard deviation

# simulation of a Vasicek path

T = 10
N_t = 5000
t = numpy.linspace(0,T,N_t)
dt = numpy.diff(t)[0]

x = numpy.zeros(N_t)
x[0] = x0
for j in range(1,N_t):
    x[j] = x[j-1] + th*(mu-x[j-1])*dt + s*scipy.stats.norm.rvs(scale=numpy.sqrt(dt))
    
mp.plot(t,x,linewidth=0.3)
mp.plot(t,mu*(1-numpy.exp(-th*t))+x0*numpy.exp(-th*t),'r')
mp.plot(t,numpy.full(shape=N_t,fill_value=mu),'k--')
mp.xlim((0,T))
mp.show()

# regression

# x(t+dt) = th*m*dt + (1-th*dt)*x(t) + e(t)
# V(e) = s**2*dt

X = numpy.ones((N_t-1,2))
X[:,1] = numpy.roll(x,-1)[:-1]
b = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(X.T,X)),X.T),x[:-1])
th_est = (1-b[1])/dt
mu_est = b[0]/(dt*th_est)
e = (x[:-1]-numpy.matmul(X,b))
s_est = numpy.std(e)/numpy.sqrt(dt)

SW_t, SW_p = scipy.stats.shapiro(e/(numpy.std(e))) # normality test on residuals

mp.plot(e/(numpy.std(e)),'.',markersize=0.8)
mp.xlim(0,N_t)
mp.show()
print(SW_p)
    
# maximum likelihood

# x(t) dist. N(mu*(1-numpy.exp(-th*t))+x0*numpy.exp(-th*t),(1-numpy.exp(-2*th))*s**2/(2*th))

def op(theta):
    
    mu = theta[0]
    x0 = x[0]
    s = theta[1]
    th = theta[2]
    
    gee = scipy.stats.norm.pdf(x[1:],loc=mu*(1-numpy.exp(-th*t[1:]))+x0*numpy.exp(-th*t[1:]),
                           scale=numpy.sqrt((1-numpy.exp(-2*th*t[1:]))*s**2/(2*th)))
    
    return -numpy.sum(numpy.log(gee))

T0 = numpy.ones(3)*0.01
op_ = scipy.optimize.minimize(op,x0=T0,method='Nelder-Mead')
res_ML = op_['x']

# results

print('true',mu,s,th)
print('reg',mu_est,s_est,th_est)
print('ML',res_ML[0],numpy.absolute(res_ML[1]),res_ML[2])

    



    
    
    
    
    

