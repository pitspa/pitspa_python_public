import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.animation

# generalization of Karhunen-Loéve of a Brownian motion 
# mean is not influent (we consider the zero-mean transformation)
# (see notes for details)

T = 3
N_t = 10000
t = numpy.linspace(0,T,N_t)
dt = numpy.diff(t)[0]

N = 500
n = numpy.arange(1,N+1,1)

tgrid, ngrid = numpy.meshgrid(t,n)

H = numpy.sqrt(2/T)*numpy.sin(numpy.pi*(tgrid/T)*(ngrid-0.5)) 
            # matrix for numerical computation of the epsilon_n coefficients
            # of a realization of the Brownian motion                                     

s = 0.25
mu = 0.005
r0 = 100

I = 1000 # Monte-Carlo simulations
sims = numpy.zeros((N_t,I))
es = numpy.zeros((N,I))
vcv_e = numpy.zeros((N,N)) # simulated vcv matrix of epsilon_n coefficients
sims[0,:] = r0
for j in range(1,N_t):
    sims[j,:] = sims[j-1,:]+mu*dt+s*scipy.stats.norm.rvs(size=I,scale=numpy.sqrt(dt))
for j in range(0,N_t):
    sims[j,:] = sims[j,:] - r0 - mu*t[j] # zero-mean transformation
for k in range(0,I):
    es[:,k] = numpy.matmul(H,sims[:,k])*dt
    vcv_e += numpy.outer(es[:,k],es[:,k])/I
    
lams_sim = numpy.diag(vcv_e)
lams = s**2/(numpy.pi/T*(n-0.5))**2 # show that the simulated eigenvalues are
                                    # approximately close to the theoretical 
                                    # ones
                                    
# partial reconstruction of a realization

fig2 = mp.figure()
ims = []

rec = numpy.zeros((N,N_t))
for k in range(0,50):
    rec[k,:] = r0 + mu*t + numpy.matmul(H[:k,:].T,es[:k,0])
    
    gee = mp.plot(t,sims[:,0]+r0 + mu*t,'b')
    gee += mp.plot(t,rec[k,:],'r')
    ims.append(gee)

ims.append(mp.plot())
im_ani = matplotlib.animation.ArtistAnimation(
        fig2, ims, interval=100, repeat=True, blit=True)

mp.xlim((t[0],T))
mp.show()

#im_ani.save('im.mp4')


