import numpy
import scipy.stats
import matplotlib.pyplot as mp

N = 1000
x = numpy.linspace(-10,20,N)

# sampling from fx using Metropolis-Hastings with known fx for simplicity

def fx(x): 
    
    """
    Returns a Gaussian mixture probability density.
    parameters:
        x : function support range
    """
    return (scipy.stats.norm.pdf(x,loc=0,scale=1) +
            scipy.stats.norm.pdf(x,loc=5,scale=3))/2
            
def gx(y,s): 
    
    """
    Returns a Gaussian random variable centered at y with stdev s.
    parameters:
        y : mean
        s : standard deviation
    """
    return scipy.stats.norm.rvs(loc=y,scale=s)

# Sampling scheme
            
I = 5000 # number of samples
x0 = -1
sig = 4
ss = numpy.array([x0])
for i in range(1,I):
    cand = gx(ss[-1],sig) 
    accept_ = fx(cand)/fx(ss[-1])
    u = scipy.stats.uniform.rvs()
    if accept_ >= u:
        ss = numpy.append(ss,cand)
    else:
        ss = numpy.append(ss,ss[-1])
        
# Kernel density estimation using Gaussian kernel

h = 0.6 # scale / smoothing factor
ran = 5000 # use up to n=ran samples
# Kpdf is asymptotically unbiased as N->inf and h->0

def Kernel(x,xi,h):
    
    """
    Gaussian Kernel density function.
    parameters:
        x : function support range
        xi : density location
        h : density scale
    """
    
    return scipy.stats.norm.pdf((x-xi)/h,loc=0,scale=1)
    # K((x-xi)/h)*1/h equivalent to N(xi,h**2) see Kpdf

def Kpdf(x,ran,h):
    
    """
    Kernel density estimation with a Gaussian Kernel.
    parameters:
        x : function support range
        ran : number of observations
        h : Kernel scaling / smoothing factor     
    """
    
    K = Kernel(x,ss[0],h)
    for j in range(1,ran):
        K += Kernel(x,ss[j],h)
    return K/(ran*h)

mp.plot(x,fx(x),'r--')
mp.plot(x,Kpdf(x,ran,0.2),'m')
mp.plot(x,Kpdf(x,ran,h),'b')
mp.plot(x,Kpdf(x,ran,1.2),'k')
mp.xlim((x[0],x[-1]))
mp.show()



        

        
            
