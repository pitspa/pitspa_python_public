import numpy
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D

m0 = 5
C0 = 2
Ve = 2


def y(N):
    
    theta0 = scipy.stats.norm.rvs(loc=m0,scale=numpy.sqrt(C0))
    y = scipy.stats.norm.rvs(size=N,loc=theta0,scale=numpy.sqrt(Ve))
    
    return theta0, y

N = 500

theta0, traj = y(N) # location is unknown, we only have noisy measurements
                    # the bayesian approach will compute the distribution
                    # of the location given the measurements for every
                    # additional measurement in the sample

print(theta0)

mp.plot(traj)
mp.show()

mn_ = numpy.zeros(N)
Cn_ = numpy.zeros(N)

# moments of the (normal) location distribution in time 
# are calculated thanks to Bayes rule:
# p(theta|y(1:t)) prop. p(y(1:t)|theta)p(theta)
# where
# p(y(1:t)|theta) is the likelihood
# p(theta) is the prior

mn_[0] = traj[0] 
Cn_[0] = Ve*C0/(Ve)

for t in range(1,N):
    mn_[t] = (numpy.average(traj[:t])*C0/(C0+Ve/t)+
               m0*(Ve/t)/(C0+Ve/t))
    Cn_[t] = Ve*C0/(Ve + t*C0)

theta_space = numpy.linspace(theta0-0.5,theta0+0.5,1000)
A = numpy.zeros((N,1000)) # Location distribution changing in time
for t in range(0,N):
    A[t] = scipy.stats.norm.pdf(theta_space,
         loc=mn_[t],scale=numpy.sqrt(Cn_[t]))

mp.imshow(A.T,cmap=mp.cm.inferno,aspect='auto',
           interpolation='none')
mp.show()

time = numpy.linspace(0,N,num=N)
space = theta_space
space, time = numpy.meshgrid(space, time)

fig = mp.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(space,time,A, 
                          cstride=2, rstride=2, cmap=mp.cm.inferno)
ax.set_xlabel('space')
ax.set_zlabel('p(theta|y(1:t))')
ax.set_ylabel('time')
mp.show()

