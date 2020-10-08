import numpy
import matplotlib.pyplot as mp
import matplotlib.animation

def f(t):  # function to be reconstructed 

    return (2+numpy.sin(2*numpy.pi*t*1.5)+
            numpy.sin(2*numpy.pi*t*2)+0.5*numpy.cos(2*numpy.pi*t*5))

def sincf(f,t,dtk): # sinc interpolation

    """
    f: observed sequence of samples
    t: denser time grid
    dtk: time interval between observed samples
    
    """
    
    N_t = len(t)
    f_rec = numpy.zeros(N_t)
    n = numpy.arange(0,len(f))
    for t_ in range(0,N_t):
        f_rec[t_] = numpy.sum(f*numpy.sinc((t[t_]-n*dtk)/dtk))
        
    return f_rec
    
T = 3 # total period (years)

N_t = 1001 # approximately continuous case
t = numpy.linspace(0,T,N_t) 
dt = numpy.diff(t)[0]

Ns = numpy.arange(10,50,1) # different numbers of samples

fig2 = mp.figure()
ims = []

for g in range(0,len(Ns)):
    N = Ns[g]
    tg = numpy.linspace(0,T,N)
    dtg = numpy.diff(tg)[0]
    
    gee = mp.plot(t,f(t),'b')
    gee += mp.plot(t,sincf(f(tg),t,dtg),'k--')
    gee += mp.plot(tg,f(tg),'r.')   
    ims.append(gee)
    
ims.append(mp.plot())
im_ani = matplotlib.animation.ArtistAnimation(
        fig2, ims, interval=100, repeat=True, blit=True)

mp.xlim((t[0],T))
mp.show()