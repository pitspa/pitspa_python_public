import numpy
import scipy.stats
import matplotlib
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches

# theory ref. https://people.eecs.berkeley.edu/~stephentu/writeups
# dirichlet-conjugate-prior.pdf

# --------------------------------------------------------------- #
# Recursive Bayesian estimation of a prior Dirichlet-distributed
# probability mass function: likelihood function is categorical
# and posterior is Dirichlet distributed
# --------------------------------------------------------------- #

def rand():
    return scipy.stats.uniform.rvs()

alpha = numpy.array([1,1,1,1]) 
f = scipy.stats.dirichlet.rvs(alpha) # this is the unknown proportions
print("True p:", f)


# Streaming data generating function:

cumf = numpy.cumsum(f)

def fun(x):
    if x <= cumf[0]:
        a = 1
    elif cumf[0] < x <= cumf[1]:
        a = 2
    elif cumf[1] < x <= cumf[2]:
        a = 3
    elif cumf[2] < x <= cumf[3]:
        a = 4
    return a

N = 100

alpha = numpy.array([1,1,1,1]) # prior guess is 0.25,0.25,0.25,0.25

Sample = []

# Point estimate

Estis = numpy.zeros((N+1,4))
Estis[0,:] = scipy.stats.dirichlet.mean(alpha) 

# Variances (stdevs)

Estis_vars = numpy.zeros((N+1,4))
Estis_vars[0,:] = (numpy.sqrt(scipy.stats.dirichlet.var(alpha)))


for i in range(1,N+1):    
    y = int(fun(rand()))
    Sample.append(y)
    A = numpy.bincount(Sample,minlength=5)[1:]
    alpha_prime = alpha + A 
    posterior_estimate = scipy.stats.dirichlet.mean(alpha_prime) 
    Estis[i,:] = posterior_estimate
    Estis_vars[i,:] = (
        numpy.sqrt(scipy.stats.dirichlet.var(alpha_prime)))
    
print("Final Bayesian estimate:",posterior_estimate)  

mp.plot(numpy.arange(0,N+1,1),Estis)
mp.grid(True)
mp.show() 

mp.plot(numpy.arange(0,N+1,1),Estis_vars)
mp.grid(True)
#for i in range(0,len(Estis[0,:])):
    #mp.fill_between(x=numpy.arange(0,N+1,1),
                    #y1=Estis_vars[:,i],color='k',
                    #alpha=0.2)
mp.show() 




    
    

    
        


    
    
        
