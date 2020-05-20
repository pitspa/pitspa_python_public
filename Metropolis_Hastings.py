import numpy
import scipy.stats
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches

N = 1000
truv = 1
x = numpy.linspace(-10,20,N)

def fx(x): # unknown pdf
    return (scipy.stats.norm.pdf(x,loc=0,scale=truv) +
            scipy.stats.norm.pdf(x,loc=5,scale=truv*3))/2
            
int_, err = scipy.integrate.quad(fx,-numpy.infty,numpy.infty)
print(int_)

def Px(x): # proportional candidate
    return fx(x)*1/5

mp.plot(x,fx(x),'b')
mp.plot(x,Px(x),'r')
mp.grid(True)
mp.show()

def gx(y): # MC conditional density
    return scipy.stats.norm.rvs(loc=y,scale=4)

def a(x_,x): # acceptance function
    return Px(x_)/Px(x)

I = 10000
x0 = -1
ss = numpy.array([x0])
for i in range(1,I):
    cand = gx(ss[-1]) # MC (pseudo-random walk): y(t) -> N(y(t-1),0.1**2)
    accept_ = a(cand,ss[-1])
    u = scipy.stats.uniform.rvs()
    if accept_ >= u:
        ss = numpy.append(ss,cand)
    else:
        ss = numpy.append(ss,ss[-1])

h, bins = numpy.histogram(ss[1000:],bins=100) # 1000 burn-in

mp.plot(bins[1:], h, 'g')
mp.grid(True)
mp.show()

print(0/2+5/2) # true mean
print(numpy.average(ss)) # emoirical mean




