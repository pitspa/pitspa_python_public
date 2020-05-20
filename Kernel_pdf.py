import numpy
import scipy.optimize
import scipy.stats
import scipy.signal
import statsmodels.tsa.stattools
import pandas
import cmath
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
pandas.set_option('Display.max_rows', None)
pandas.set_option('Display.max_columns', None)
pandas.set_option('display.width', 1000)

N = 1000
x = numpy.linspace(-10,20,N)

# sampling from fx using Metropolis-Hastings with known fx for simplicity

def fx(x): 
    return (scipy.stats.norm.pdf(x,loc=0,scale=1) +
            scipy.stats.norm.pdf(x,loc=5,scale=3))/2
            
def gx(y): 
    return scipy.stats.norm.rvs(loc=y,scale=4)
            
I = 1000
x0 = -1
ss = numpy.array([x0])
for i in range(1,I):
    cand = gx(ss[-1]) 
    accept_ = fx(cand)/fx(ss[-1])
    u = scipy.stats.uniform.rvs()
    if accept_ >= u:
        ss = numpy.append(ss,cand)
    else:
        ss = numpy.append(ss,ss[-1])
        
# Kernel density estimation using Gaussian kernel (-inf, inf support)

h = 0.6 # smoothing factor
ran = 1000 # use up to n=ran samples
# Kpdf is asymptotically unbiased as N->inf and h->0

def Kernel(x,xi,h):
    return scipy.stats.norm.pdf((x-xi)/h,loc=0,scale=1)
    # K((x-xi)/h)*1/h equivalent to N(xi,h**2) see Kpdf

def Kpdf(N,x,ran,h):
    
    K = Kernel(x,ss[0],h)
    for j in range(1,ran):
        K += Kernel(x,ss[j],h)
    return K/(ran*h)

mp.plot(x,fx(x),'r--')
mp.plot(x,Kpdf(N,x,ran,0.2),'m')
mp.plot(x,Kpdf(N,x,ran,h),'b')
mp.plot(x,Kpdf(N,x,ran,1.2),'k')
mp.grid(True)
mp.show()



        

        
            
