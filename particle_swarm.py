import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.animation

# particle swarm to find the minimum of a function (one dimensional)

I = 1000
x = numpy.linspace(-3,3,I)

def f(x): 
    return (numpy.sin(2*numpy.pi*x)*(-numpy.exp(-10*(x-0.25)**2))+
            0.1*(x-0.25)**6*(1-numpy.exp(-10*(x+1.5)**2))*
            (2-numpy.exp(-100*(x-1.5)**2)))

mp.xlim(-2,2)
mp.ylim(-1.2,2)
mp.plot(x,f(x))
mp.show()
              
kmax = 1000 # stopping time

c1 = 0.005 # sensitivity to particle best
c2 = 0.0025 # sensitivity to group best
o = 0.8 # velocity decay/increase (decay will eventually stop the particles)

p = 100 # number of swarming particles
x = numpy.zeros((kmax,p)) # particles positions (i.e. x) for all times
P = numpy.zeros(p) # best positions 
v = numpy.zeros((kmax,p)) # 'velocities'
x[0,:] = scipy.stats.uniform.rvs(size=p,loc=-10,scale=20) # initial positions
v[0,:] = scipy.stats.uniform.rvs(size=p,loc=-0.02,scale=0.04) # initial vel.

eval_ = numpy.zeros(p) # function evaluation f(x)
fbest = numpy.zeros(p)  # best minimum value of f(x) found per particle

# algorithm output:

fbestg = 0 # best function minimum in all group
pg = 0 # group best position



fig2 = mp.figure() 

ims = []
for k in range(1,kmax):
    eval_ = f(x[k-1,:]) # evaluate the function at all particles positions
    for i in range(0,p):
        
        # for any particle:
        # if the evaluation is less than the personal best:
        #   evaluation becomes personal best
        #   the position becomes the best position
        # if the evaluation is less than the group best:
        #   evaluation becomes group best
        #   the position becomes the group's best
        
        if eval_[i] <= fbest[i]:           
            fbest[i] = eval_[i]
            P[i] = x[k-1,i]
        if eval_[i] <= fbestg:
            fbestg = eval_[i]
            pg = x[k-1,i]
            
    # now move the particles:
                  
    r1 = scipy.stats.uniform.rvs(size=p) # random movement lengths
    r2 = scipy.stats.uniform.rvs(size=p)
    
    # displacement depends on:
    # velocity decay (o)
    # distance from personal best
    # distance from group best
    #
    # if personal/group best is > (<) position -> particle will move towards it
    # by increasing (decreasing) position
    #
    # the random movements will force the particles to 'search' the 
    # function space
    
    
    v[k,:] = o*v[k-1,:] + c1*r1*(P-x[k-1,:])+c2*r2*(pg-x[k-1,:])
    x[k,:] = x[k-1,:] + v[k,:]
 
    mp.xlim(-2,2)
    mp.ylim(-1.2,2)
    mp.grid(True)
    
    gee = mp.plot(x[k-1,:],eval_,'k.')
    
    ims.append(gee)

   
im_ani = matplotlib.animation.ArtistAnimation(
        fig2, ims, interval=20, repeat=True, blit=True)

mp.show() 

print('Swarm minimum: ',fbestg,' at x=',pg)

    
    


    
            
        
    

                     